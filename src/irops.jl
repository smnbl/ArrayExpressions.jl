const CC = Core.Compiler
using Core.Compiler: iterate
using CUDA

using Core: SSAValue

# extract expression that resolves to #indx SSA value
# TODO: handle phi nodes -> atm: stop at phi nodes (diverging control flow)
#       -> look at implementing gated SSA, for full body analysis?
# visited: set of already visited notes
function _extract_slice!(ir::IRCode, loc::SSAValue; visited=Int64[], modify_ir=true, intr_outputs=Dict{Any, Vector{IntrinsicInstance}}(), intrinsic=nothing)
    inst = ir.stmts[loc.id][:inst]
    type = CC.widenconst(ir.stmts[loc.id][:type])

    # inference boundaries
    pure = !(Base.isexpr(inst, :new) || Base.isexpr(inst, :static_parameter))

    # doesn't work that well atm
    # pure = ir.stmts[loc.id][:flag] & CC.IR_FLAG_EFFECT_FREE != 0
    # if statement is not pure we can not extract the instruction
    if (!pure)
        return visited, Input(loc, type)
    end

    # TODO: check this!!
    # check if we're not crossing an intrinsic which modifies 
    crossing_intrinsic = arg -> haskey(intr_outputs, arg) && any(el -> el < loc.id, intr_outputs[arg])

    # get last intrinsic that modified the arg
    last_intrinsic = arg -> begin
        last = findlast(el -> el < loc.id, intr_outputs[arg])
        if (last === nothing)
            return nothing
        end
        return intr_outputs[arg][last]
    end

    if intrinsic !== nothing
        args = getargs(ir, intrinsic)
        inst = Expr(:intrinsic, intrinsic, args...)
    end

    if inst isa Expr
        args_op = []

        push!(visited, loc.id)

        for (idx, arg) in enumerate(inst.args)
            # keep track of the visited statements (kind off redundant as they will be marked for deletion (nothing))

            # TODO: v1.8 already has a version that works on IRCode, without explicitly passing sptypes & argtypes
            arg_type = CC.widenconst(CC.argextype(arg, ir))

            if arg isa SSAValue
                if crossing_intrinsic(arg)
                    last = last_intrinsic(arg)

                    visited, subtree = _extract_slice!(ir, SSAValue(last.location), visited=visited, intr_outputs=intr_outputs, intrinsic=last)
                    
                    push!(args_op, subtree)
                else
                    # TODO: stop at functions with side effects?
                    # TODO: look at Julia's native purity modeling infra (part of Julia v1.8): https://github.com/JuliaLang/julia/pull/43852
                    visited, arexpr = _extract_slice!(ir, arg, visited=visited, intr_outputs=intr_outputs)
                    push!(args_op, arexpr)
                end
            elseif arg isa Output
                push!(args_op, Output(arg.val, arg_type))
            else
                push!(args_op, Input(arg, arg_type))
            end
        end

        return visited, ArrayExpr(inst.head, [args_op...], type)

    elseif inst isa SSAValue
        push!(visited, loc.id)

        if crossing_intrinsic(inst)
            last = last_intrinsic(inst)
            visited, subtree = _extract_slice!(ir, last.possible, visited=visited, intr_outputs=intr_outputs, intrinsic=last)
            return visited, subtree
        end

        return _extract_slice!(ir, inst, visited=visited, intr_outputs=intr_outputs)

    elseif inst isa CC.GlobalRef
        # keep track of the visited statements (kind off redundant as they will be marked for deletion (nothing))
        push!(visited, loc.id)
        return visited, Input(inst, type)

    else # PhiNodes, etc...
        return visited, Input(loc, type)
    end
end

function extract_slice!(ir::IRCode, loc::SSAValue)
    _, arir, _ = _extract_slice!(ir, loc, visited=Int64[])
    return arir
end

function replace!(ir::IRCode, visited, first, call_expr, type, output_map)
    loc = first.id

    let compact = CC.IncrementalCompact(ir, true)
        # insert tuple call right before first use
        ssa_tuple = CC.insert_node!(compact, first, CC.non_effect_free(CC.NewInstruction(call_expr, type)))

        next = CC.iterate(compact)
        while true
            ((old_idx, idx), stmt) = next[1]

            if (old_idx == loc)
                expr = Expr(:call, GlobalRef(Base, :getfield), ssa_tuple, 1)
                CC.fixup_node(compact, expr)
                CC.setindex!(compact.result[idx], expr, :inst)
            end

            next = CC.iterate(compact, next[2])
            next != nothing || break
        end

        for old_idx in visited
            if old_idx != loc
                idx = compact.ssa_rename[old_idx].id
                compact.used_ssas[idx] -= 1

                println("$old_idx: used $(compact.used_ssas[idx])")
                # TODO: this is broken
                # intrinsics will have used_ssas < 0
                if compact.used_ssas[idx] <= 0
                    CC.setindex!(compact.result[idx], nothing, :inst)
                end
            end
        end

        ir = CC.finish(compact)
    end
end

# Array optimizations pass
function (aro::ArrOptimPass)(ir::IRCode, mod::Module)
    stmts = ir.stmts
    visited = Int64[]
    exprs = ArrayIR[]

    first = nothing

    correct_rettype(rettype) = rettype <: aro.element_type && rettype != Union{}

    intr_outputs = Dict{Any, Vector{IntrinsicInstance}}() # arg -> lines that update this arg (indirectly)
    intrinsic_instances = []

    # Forward Pass
    for idx in 1:length(ir.stmts)
        # lower invokes to calls
        inst = ir.stmts.inst[idx]
        if (CC.isexpr(inst, :invoke))
            ir.stmts.inst[idx] = Expr(:call, inst.args[2:end]...)
            inst = ir.stmts.inst[idx]
        end

        # fix intrinsics
        instance = inintrinsics(inst, aro.intrinsics, idx)
        if !isnothing(nothing)
            # check if kwfunc
            op = inst.args[1]

            # offset from where the arguments start
            args_offset = instance.args_offset

            for arg in instance.intrinsic.outputs
                push!(get!(intr_outputs, inst.args[arg + args_offset], []), instance)
                push!(intrinsic_instances, instance)
            end
            println(intr_outputs)
        end
    end


    # TODO:
    # iterate over basic blocks

    # 1. collect expressions
    for idx in length(stmts):-1:1
        inst = stmts.inst[idx]

        loc = SSAValue(idx)

        idx ∉ visited || continue
        inst isa Expr || continue

        # check if return type is StubArray and use this to confirm array ir
        rettype = CC.widenconst(stmts[idx][:type])
        iscopyto_array = iscall(inst, GlobalRef(Main, :copyto!)) && correct_rettype(CC.widenconst(CC.argextype(inst.args[2], ir)))

        rule = inrules(inst, aro.extra_rules)
        intrinsic = inintrinsics(inst, aro.intrinsics, idx)

        # TODO: bench hom much speedup due to inrules!
        rule || intrinsic !== nothing || continue 
        # (correct_type #=&& inrules(inst, aro.extra_rules)=# || iscopyto_array) || continue

        if (rule)
           # TODO: optimize whole ir body iteratively
            # extract & optimize array expression that returns
            # TODO: work with basic blocks and CFG?
            visited, arexpr = _extract_slice!(ir, loc, visited=visited, intr_outputs=intr_outputs)

            # todo: isn't this last?
            if first == nothing
                first = CC.OldSSAValue(idx)
            end

            push!(exprs, arexpr)

            break
        else
            # TODO: start at intrinsics
            # TODO: idea to handle intrinsics:
            # wrap in sth like (virtually?): f(outputs, inputs) = { intrinsic(outputs, inputs); return outputs }
            # ? requires forward pass

            # if intrinsic:

            println("INTRINSIC:")
            println(inst.args[1])

            op = inst.args[intrinsic.intrinsic.operation_pos + intrinsic.args_offset - 1]
            op_type = CC.widenconst(CC.argextype(op, ir))
            println("operation: $(op_type)")

            visited, tree = _extract_slice!(ir, loc, visited=visited, intr_outputs=intr_outputs, intrinsic=intrinsic)

            # todo: isn't this last?
            if first == nothing
                first = CC.OldSSAValue(idx)
            end

            push!(exprs, tree)

            # stop at first tree; as rest is broken
            break
        end
    end

    # if nothing found, just return ir
    # TODO: add outputmap
    if (length(exprs) > 1 || isempty(exprs) || all(map(el -> el isa Input, exprs)))
        return ir
    end

    # 2. construct output tuple
    # wrap arexpr in output function
    # this is to make expressions that only consist of 1 SSAValue, (e.g. arexpr = %4)
    # equivalent to expressions that only contain SSAValues inside its arguments
    tuple_type = Tuple{map(x -> x.type, exprs)...}
    tuple = ArrayExpr(:tuple, exprs, tuple_type)
    output = ArrayExpr(:output, [tuple], tuple.type)
    println(output)
    println(">>>")

    # 3. jointly optimize output tuple
    op = simplify(output, extra_rules=aro.extra_rules)
    println("simplified = $op")
    println("---")

    # 4. insert optimized expression back
    expr, input_map = codegen_expr!(op.args[1], length(ir.argtypes))

    println("compiling opaque closure:")
    println(expr)

    # note: world age changes here
    # TODO: investigate if problem, read world age paper
    
    # outlines in separate function,
    # TODO: might switch to opaque closure (but has world age problems)
    # idea: run this in an older world age, decompose the eval so that it does not update the world age counter?
    # oc = Core.eval(mod, Expr(:opaque_closure, expr))
    oc = Core.eval(mod, expr)

    arguments = Array{Any}(undef, length(input_map))

    for (val, idx) in pairs(input_map)
        if val isa SSAValue
            # wrap in OldSSA for compacting
            val = CC.OldSSAValue(val.id)
        end
        arguments[idx] = val
    end

    # TODO: should we use invokelatest here?
    # TODO: try ... catch ... pattern for faster call invocations
    # doesn't seem to work with opaque closures :(
    call_expr =  Expr(:call, oc, arguments...)

    println(visited)

    # TODO: add output map
    # Have to set output type to (Any, ...), otherwise segmentation faults -> TODO: investigate, seems to be the case if there is a mismatch in the return types
    ir = replace!(ir, visited, first, call_expr, NTuple{length(exprs), Any}, nothing)

    return ir
end
