const CC = Core.Compiler
using Core.Compiler: iterate

using Core: SSAValue

# extract expression that resolves to #indx SSA value
# TODO: handle phi nodes -> atm: stop at phi nodes (diverging control flow)
# visited: set of already visited notes
function _extract_slice!(ir::IRCode, loc::SSAValue; visited=Set(), modify_ir=true)
    inst = ir.stmts[loc.id][:inst]
    type = ir.stmts[loc.id][:type]

    # keep track of the visited statements (kind off redundant as they will be marked for deletion (nothing))
    push!(visited, loc)

    # doesn't work that well atm
    # pure = ir.stmts[loc.id][:flag] & CC.IR_FLAG_EFFECT_FREE != 0
    pure = true

    # if statement is not pure we can not extract the instruction
    if (!pure)
        return visited, ArrayExpr(:call, [:input, loc], CC.widenconst(type))
    end

    # delete visited statement
    if (modify_ir)
        ir.stmts.inst[loc.id] = nothing
    end

    if inst isa Expr
        args_op = []

        for (idx, arg) in enumerate(inst.args)
            # TODO: v1.8 already has a version that works on IRCode, without explicitly passing sptypes & argtypes
            arg_type = CC.widenconst(CC.argextype(arg, ir))

            if arg isa SSAValue && arg_type <: Union{AbstractGPUArray, Number}
                # TODO: stop at functions with side effects?
                # TODO: look at Julia's native purity modeling infra (part of Julia v1.8): https://github.com/JuliaLang/julia/pull/43852
                visited, arexpr = _extract_slice!(ir, arg, visited=visited)
                push!(args_op, arexpr)
            else
                # function matcher only supports, function name Symbols or function objects (not GlobalRef)
                # TODO: add support for matching GlobalRef's to Metatheory.jl?
                if inst.head == :call && idx == 1 && arg isa GlobalRef ||
                    inst.head == :invoke && idx == 2 && arg isa GlobalRef
                    println("replacing with func object")
                    arg = getproperty(arg.mod, arg.name)
                end

                push!(args_op, ArrayExpr(:call, [:input, arg], arg_type))
            end
        end

        return visited, ArrayExpr(inst.head, [args_op...], CC.widenconst(type))

    elseif inst isa CC.GlobalRef || inst isa CC.PhiNode
        return visited, ArrayExpr(:call, [:input, inst], type)
    else
        println("not an expr, but: $inst::$(typeof(inst))")
        return visited, ArrayExpr(:unknown, [], Union{})
    end
end

function extract_slice!(ir::IRCode, loc::SSAValue)
    _, arir, _ = _extract_slice!(ir, loc, visited=Set())
    return arir
end

# Array optimizations pass
function arroptim_pass(ir::IRCode)
    stmts = ir.stmts
    visited = Set()

    # start backwards
    for idx in length(stmts):-1:1
        inst = stmts.inst[idx]
        
        loc = SSAValue(idx)

        # loc ∉ visited || continue
        inst isa Expr || continue

        # check if return type is StubArray and use this to confirm array ir
        rettype = CC.widenconst(stmts[idx][:type])
        rettype <: ValueTypes || continue

        # TODO: optimize whole ir body iteratively
        # extract & optimize array expression that returns
        visited, arexpr = _extract_slice!(ir, loc, visited=visited)

        # wrap arexpr in output function
        # this is to make expressions that only consist of 1 SSAValue, (e.g. arexpr = %4)
        # equivalent to expressions that only contain SSAValues inside its arguments
        arexpr = ArrayExpr(:output, [arexpr], rettype)

        println(arexpr)

        println(">>>")

        op = simplify(arexpr)
        println("simplified = $op")
        println("---")

        new_ssa = codegen_ssa!(op)

        println("insert optimized instructions")

        ssaval = nothing

        compact = CC.IncrementalCompact(ir, true)

        # TODO: can be faster by just determining constant offset and incrementing each SSAValue
        # TODO: verify if this is all correct
        ssa_map = Dict()
        ssaval = SSAValue(-1)
        for (idx, stmt) in enumerate(new_ssa[1:end - 1])
            # replace with final SSAValue
            for (arg_idx, arg) in enumerate(stmt.args)
                if arg isa TempSSAValue
                    stmt.args[arg_idx] = ssa_map[arg.id]
                elseif arg isa SSAValue # mark for renaming
                    stmt.args[arg_idx] = CC.OldSSAValue(arg.id)
                end
            end

            if (stmt.head == :call && stmt.args[1] == :input)
                # remove input label
                expr = Expr(stmt.args[2])
            else
                expr = convert(Expr, stmt)
            end

            if (idx == lastindex(new_ssa) - 1) # last statement (don't input output statement)
                stmts.inst[loc.id] = expr
            elseif (idx == 1) # first statement / insert before loc
                ssaval = CC.insert_node!(compact, loc, CC.effect_free(CC.NewInstruction(expr, stmt.type)), false)
            else
                ssaval = CC.insert_node!(compact, ssaval, CC.effect_free(CC.NewInstruction(expr, stmt.type)), true)
            end

            ssa_map[idx] = ssaval
        end

        next = CC.iterate(compact)
        while (next != nothing)
            next = CC.iterate(compact, next[2])
        end

        ir = CC.finish(compact)

        # for now only replace 1
        return ir
    end

    return ir
end
