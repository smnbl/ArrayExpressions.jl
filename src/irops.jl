const CC = Core.Compiler
using Core.Compiler: iterate
using CUDA

using Core: SSAValue

# extract expression that resolves to #indx SSA value
# TODO: handle phi nodes -> atm: stop at phi nodes (diverging control flow)
#       -> look at implementing gated SSA, for full body analysis?
# visited: set of already visited notes
function _extract_slice!(ir::IRCode, loc::SSAValue; visited=Int64[], modify_ir=true)
    inst = ir.stmts[loc.id][:inst]
    type = CC.widenconst(ir.stmts[loc.id][:type])

    # inference boundarys
    pure = !(Base.isexpr(inst, :new) || Base.isexpr(inst, :static_parameter))

    # doesn't work that well atm
    # pure = ir.stmts[loc.id][:flag] & CC.IR_FLAG_EFFECT_FREE != 0
    # if statement is not pure we can not extract the instruction
    if (!pure)
        return visited, Input(loc, type)
    end

    if inst isa Expr
        args_op = []

        # keep track of the visited statements (kind off redundant as they will be marked for deletion (nothing))
        push!(visited, loc.id)

        for (idx, arg) in enumerate(inst.args)
            # TODO: v1.8 already has a version that works on IRCode, without explicitly passing sptypes & argtypes
            arg_type = CC.widenconst(CC.argextype(arg, ir))

            if arg isa SSAValue
                # TODO: stop at functions with side effects?
                # TODO: look at Julia's native purity modeling infra (part of Julia v1.8): https://github.com/JuliaLang/julia/pull/43852
                visited, arexpr = _extract_slice!(ir, arg, visited=visited)
                push!(args_op, arexpr)
            else
                push!(args_op, Input(arg, arg_type))
            end
        end

        return visited, ArrayExpr(inst.head, [args_op...], type)
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
    loc = maximum(visited)

    println(loc)
    println(first.id)

    @assert loc == first.id

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
                if compact.used_ssas[idx] == 0
                    CC.setindex!(compact.result[idx], nothing, :inst)
                end
            end

        end

        ir = CC.finish(compact)
    end
end

export ArrOptimPass

struct ArrOptimPass
    extra_rules::Vector{Metatheory.AbstractRule}

    ArrOptimPass(; extra_rules=[]) = return new(extra_rules)
end


correct_rettype(rettype) = rettype <: ValueTypes && rettype != Union{}

# Array optimizations pass
function (aro::ArrOptimPass)(ir::IRCode, mod)
    # check meta tag
    perform_array_opt = CC._any(@nospecialize(x) -> CC.isexpr(x, :meta) && x.args[1] === :array_opt, ir.meta)
    if (false && !perform_array_opt)
        println("skipping")
        # do nothing
        return ir
    end

    stmts = ir.stmts
    visited = Int64[]
    exprs = ArrayIR[]

    first = nothing

    # 1. collect expressions
    for idx in length(stmts):-1:1
        inst = stmts.inst[idx]
        
        loc = SSAValue(idx)

        idx ∉ visited || continue
        inst isa Expr || continue

        # check if return type is StubArray and use this to confirm array ir
        rettype = CC.widenconst(stmts[idx][:type])
    
        iscopyto_array = iscall(inst, GlobalRef(Main, :copyto!)) && correct_rettype(CC.widenconst(CC.argextype(inst.args[2], ir)))

        (correct_rettype(rettype) || iscopyto_array) || continue

        println(inst.args)

        line = stmts[idx][:line]

        # TODO: optimize whole ir body iteratively
        # extract & optimize array expression that returns
        # TODO: work with basic blocks and CFG?
        visited, arexpr = _extract_slice!(ir, loc, visited=visited)

        if first == nothing
            first = CC.OldSSAValue(idx)
        end

        push!(exprs, arexpr)
    end

    # if nothing found, just return ir
    if (isempty(exprs))
        return ir
    end

    println("arr optimizing")
    println(ir)

    # 2. construct output tuple
    # wrap arexpr in output function
    # this is to make expressions that only consist of 1 SSAValue, (e.g. arexpr = %4)
    # equivalent to expressions that only contain SSAValues inside its arguments
    tuple_type = Tuple{map(x -> x.type, exprs)...}
    println(tuple_type)
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

    oc = Core.eval(mod, Expr(:opaque_closure, expr))

    arguments = Array{Any}(undef, length(input_map))
    for (val, idx) in pairs(input_map)
        if val isa SSAValue
            # wrap in OldSSA for compacting
            val = CC.OldSSAValue(val.id)
        end
        arguments[idx] = val
    end

    println(arguments)

    call_expr =  Expr(:call, oc, arguments...)

    # TODO: add output map
    ir = replace!(ir, visited, first, call_expr, tuple_type, nothing)
    
    println(ir)


    return ir
end
