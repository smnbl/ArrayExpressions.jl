const CC = Core.Compiler
using Core.Compiler: iterate
using CUDA

using Core: SSAValue
using Core.Compiler: OldSSAValue, NewSSAValue


# extract expression that resolves to #indx SSA value
# TODO: handle phi nodes -> atm: stop at phi nodes (diverging control flow)
#       -> look at implementing gated SSA, for full body analysis?
# visited: set of already visited notes
function _extract_slice!(ir::IRCode, loc::SSAValue; visited=Int64[], latest_ref=0)
    inst = ir.stmts[loc.id][:inst]
    # might include const information
    type = ir.stmts[loc.id][:type]

    # inference boundaries
    pure = !(Base.isexpr(inst, :new) || Base.isexpr(inst, :static_parameter) || Base.isexpr(inst, :foreigncall))

    # doesn't work that well atm
    # pure = ir.stmts[loc.id][:flag] & CC.IR_FLAG_EFFECT_FREE != 0
    # if statement is not pure we can not extract the instruction
    if (!pure)
        return visited, Input(loc, type), latest_ref
    end

    if inst isa Expr
        args_op = []

        push!(visited, loc.id)

        for (idx, arg) in enumerate(inst.args)
            # keep track of the visited statements (kind off redundant as they will be marked for deletion (nothing))

            # TODO: v1.8 already has a version that works on IRCode, without explicitly passing sptypes & argtypes
            # might include Const information
            arg_type = CC.argextype(arg, ir)

            if arg isa SSAValue
                latest_ref = max(latest_ref, loc.id)

                # TODO: stop at functions with side effects?
                # TODO: look at Julia's native purity modeling infra (part of Julia v1.8): https://github.com/JuliaLang/julia/pull/43852
                visited, arexpr, latest_ref = _extract_slice!(ir, arg, visited=visited, latest_ref=latest_ref)

                push!(args_op, arexpr)
            elseif arg isa Output
                push!(args_op, Output(arg.val, arg_type))
            else
                push!(args_op, Input(arg, arg_type))
            end
        end
        return visited, ArrayExpr(inst.head, [args_op...], type), latest_ref
    elseif inst isa SSAValue
        latest_ref = max(latest_ref, loc.id)

        push!(visited, loc.id)
        return _extract_slice!(ir, inst, visited=visited, latest_ref=latest_ref)

    elseif inst isa CC.GlobalRef
        # keep track of the visited statements (kind off redundant as they will be marked for deletion (nothing))
        push!(visited, loc.id)
        return visited, Input(inst, type), latest_ref

    else # PhiNodes, etc...
        return visited, Input(loc, type), latest_ref
    end
end

function extract_slice!(ir::IRCode, loc::SSAValue)
    _, arir, _ = _extract_slice!(ir, loc, visited=Int64[])
    return arir
end

function replace!(ir::IRCode, visited, todo_opt::Dict, output_map)
    # maps getfields locations to the tuple locations
    tuple_loc = Dict{OldSSAValue, Any}()
    rename = Dict{Int, OldSSAValue}()

    dont_delete = Int[]

    let compact = CC.IncrementalCompact(ir, true)
        # insert tuple calls right before the locations
        for (loc, (call_expr, type, getfield_locs)) in todo_opt
            ssa_tuple = CC.insert_node!(compact, SSAValue(loc.id), CC.non_effect_free(CC.NewInstruction(call_expr, type)))
            tuple_loc[loc] = ssa_tuple

            for (field_index, getfield_ssa) in enumerate(getfield_locs)
                expr = Expr(:call, GlobalRef(Base, :getfield), ssa_tuple, field_index)
                new_ssa = CC.insert_node!(compact, ssa_tuple, CC.non_effect_free(CC.NewInstruction(expr, Any)), true)
                rename[getfield_ssa.id] = new_ssa
                CC.setindex!(compact, new_ssa, getfield_ssa.id)
                push!(dont_delete, getfield_ssa.id)
            end
        end

        next = CC.iterate(compact)
        while true
            (((idx, new_idx), stmt), (next_idx, next_bb)) = next
            next = CC.iterate(compact, (next_idx, next_bb))
            next != nothing || break
        end

        # can remove all the old instructions as they are now covered by the inserted expression
        for old_idx in visited
            old_idx ∉ dont_delete || continue

            idx = compact.ssa_rename[old_idx].id

            # TODO: check this
            if (idx < length(compact.used_ssas))
                compact.used_ssas[idx] -= 1
            

                # println("$old_idx: used $(compact.used_ssas[idx])")
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

# TODO: create structs for the custom types, tuples and pairs can get a bit smelly

struct TodoItem
    exprs::Dict{OldSSAValue, ArrayIR} # extraction loc -> expression
    visited::Vector{Int64}
end

# Array optimizations pass
function (aro::ArrOptimPass)(ir::IRCode, mod::Module)
    open("irdump_before.ir", "w") do io
           print(io, ir)
    end

    expression_log = open("expression_log", "w")

    stmts = ir.stmts
    visited = Int64[]

    first = nothing

    correct_rettype(rettype) = rettype <: aro.element_type && rettype != Union{}

    for idx in 1:length(ir.stmts)
        # lower invokes to calls
        # also done to make metatheory matching work
        # TODO: move this to custom inlining
        inst = ir.stmts.inst[idx]
        if (CC.isexpr(inst, :invoke))
            ir.stmts.inst[idx] = Expr(:call, inst.args[2:end]...)
            inst = ir.stmts.inst[idx]
        end
    end

    # loc of insertion, expr tree to optimize
    todo = Pair{OldSSAValue, TodoItem}[]
    current_bb = CC.block_for_inst(ir.cfg, length(stmts))

    # loc start of expr => extracted expression
    exprs = Dict{OldSSAValue, ArrayIR}()
    insert_loc = nothing

    # latest referenced SSAValue
    latest_ref = 0

    # 1. collect expressions
    for idx in length(stmts):-1:1
        # switched to a new expression when hitting a basic block / crossing the latest ref
        if (current_bb != CC.block_for_inst(ir.cfg, idx) || idx == latest_ref)
            if !isempty(exprs)
                if isnothing(insert_loc)
                    throw("insert loc should be set!") 
                end
                push!(todo, Pair(insert_loc, TodoItem(exprs, visited)))
            end

            visited = Int64[]
            exprs = Dict{OldSSAValue, ArrayIR}()
            insert_loc = nothing
            current_bb = CC.block_for_inst(ir.cfg, idx)
            latest_ref = 0
        end

        inst = stmts.inst[idx]
        loc = SSAValue(idx)

        idx ∉ visited || continue
        inst isa Expr || continue

        # check if return type is StubArray and use this to confirm array ir
        #iscopyto_array = iscall(inst, GlobalRef(Main, :copyto!)) && correct_rettype(CC.widenconst(CC.argextype(inst.args[2], ir)))

        rule = inrules(ir, inst, aro.extra_rules)
        # TODO: bench hom much speedup due to inrules!
        rule || continue 

        # check if correct return type -> does not work when working with tuples!
        # rettype = CC.widenconst(stmts[idx][:type])
        # correct_rettype(rettype) || continue

        # TODO: optimize whole ir body iteratively
        # extract & optimize array expression that returns
        # TODO: work with basic blocks and CFG?

        visited, arexpr, latest_ref = _extract_slice!(ir, loc, visited=visited, latest_ref=latest_ref)

        # always insert at the tuple at the location furthest down
        if isnothing(insert_loc)
            insert_loc = OldSSAValue(idx)
        end

        exprs[OldSSAValue(idx)] = arexpr
    end
    # push from last bb
    if(!isnothing(insert_loc))
        push!(todo, Pair(insert_loc, TodoItem(exprs, visited)))
    end

    # if nothing found, just return ir
    # TODO: add outputmap
    if (isempty(todo)) # TODO: all(map(el -> el isa Input, exprs)))
        return ir
    end

    todo_opt = Dict{OldSSAValue, Tuple{Expr, Type, Any}}()

    # visited used to clean up
    visited_clean = Int64[]

    # TODO: do all in 1 iteration over the intsts (1 compacting iteration)
    for (loc, item) in todo
        exprs = item.exprs
        # 2. construct output tuple
        # wrap arexpr in output function
        # this is to make expressions that only consist of 1 SSAValue, (e.g. arexpr = %4)
        # equivalent to expressions that only contain SSAValues inside its arguments
        tuple_type = Tuple{map(x -> x.type, values(exprs))...}
        tuple_expr = ArrayExpr(:tuple, collect(values(exprs)), tuple_type)
        output = ArrayExpr(:output, [tuple_expr], tuple_expr.type)
        #println(output)
        #println(">>>")

        # 3. jointly optimize output tuple
        simplified = simplify(output, extra_rules=aro.extra_rules)
        #println("simplified = $op")
        #println("---")

        println(expression_log, "before: $output")
        println(expression_log, "after: $simplified")

        # TODO: broken as resp visited nodes are not removed
        if fingerprint(output) == fingerprint(simplified)
            println(expression_log, "skipping injection")
            # trees most likely stayed the same
            continue
        end

        # 4. insert optimized expression back
        expr, input_map = codegen_expr!(simplified.args[1], length(ir.argtypes))

        println(expression_log, "compiling expression:")
        println(expression_log, expr)

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

        union!(visited_clean, item.visited)
        push!(todo_opt, Pair(loc, (call_expr, tuple_type, collect(keys(exprs)))))
    end

    # TODO: add output map
    # Have to set output type to (Any, ...), otherwise segmentation faults -> TODO: investigate, seems to be the case if there is a mismatch in the return types
    ir = replace!(ir, visited_clean, todo_opt, nothing)

    close(expression_log)

    open("irdump.ir", "w") do io
           print(io, ir)
    end

    return ir
end

function (aro::ArrOptimPass)(code_info::CodeInfo, method_instance::MethodInstance)
    ir = CC.inflate_ir(code_info, method_instance)

    ir = aro(ir, method_instance.def.module)

    println("replacing code newstyle")
    CC.replace_code_newstyle!(code_info, ir, Int64(method_instance.def.nargs))
 end
