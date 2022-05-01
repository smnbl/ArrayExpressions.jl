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
    type = ir.stmts[loc.id][:type]


    # doesn't work that well atm
    # pure = ir.stmts[loc.id][:flag] & CC.IR_FLAG_EFFECT_FREE != 0
    pure = true

    # if statement is not pure we can not extract the instruction
    if (!pure)
        return visited, ArrayExpr(:call, [:input, loc], CC.widenconst(type))
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
                # function matcher only supports, function name Symbols or function objects (not GlobalRef)
                # TODO: add support for matching GlobalRef's to Metatheory.jl?
                if inst.head == :call && idx == 1 && arg isa GlobalRef ||
                    inst.head == :invoke && idx == 2 && arg isa GlobalRef
                    arg = getproperty(arg.mod, arg.name)
                end

                push!(args_op, ArrayExpr(:call, [:input, arg], arg_type))
            end
        end

        return visited, ArrayExpr(inst.head, [args_op...], CC.widenconst(type))
    elseif inst isa CC.GlobalRef
        # keep track of the visited statements (kind off redundant as they will be marked for deletion (nothing))
        push!(visited, loc.id)

        return visited, ArrayExpr(:call, [:input, inst], type)
    else # PhiNodes, etc...
        return visited, ArrayExpr(:call, [:input, loc], type)
    end
end

function extract_slice!(ir::IRCode, loc::SSAValue)
    _, arir, _ = _extract_slice!(ir, loc, visited=Int64[])
    return arir
end

function replace!(ir::IRCode, visited, expr)
    loc = maximum(visited)

    let compact = CC.IncrementalCompact(ir, true)
        next = CC.iterate(compact)
        while true
            ((old_idx, idx), stmt) = next[1]

            if (old_idx == loc)
                println("changing")
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
                    println("deleting")
                    CC.setindex!(compact.result[idx], nothing, :inst)
                end
            end

        end

        ir = CC.finish(compact)
    end
end

# Array optimizations pass
function arroptim_pass(ir::IRCode, mod)
    # check meta tag
    perform_array_opt = CC._any(@nospecialize(x) -> CC.isexpr(x, :meta) && x.args[1] === :array_opt, ir.meta)
    if (!perform_array_opt)
        # do nothing
        return ir
    end
    println("arr optimizing")

    println(ir)
    
    stmts = ir.stmts
    visited = Int64[]

    # start backwards
    for idx in length(stmts):-1:1
        inst = stmts.inst[idx]
        
        loc = SSAValue(idx)

        # loc ∉ visited || continue
        inst isa Expr || continue

        println(inst.args[1])

        # check if return type is StubArray and use this to confirm array ir
        rettype = CC.widenconst(stmts[idx][:type])
    
        iscopyto_array = inst.head == :call && inst.args[1] == GlobalRef(Main, :copyto!) && CC.widenconst(CC.argextype(inst.args[2], ir)) <: ValueTypes

        rettype <: ValueTypes || iscopyto_array || continue

        line = stmts[idx][:line]

        # TODO: optimize whole ir body iteratively
        # extract & optimize array expression that returns
        # TODO: work with basic blocks and CFG?
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

        call_expr =  Expr(:call, oc, arguments...)

        ir = replace!(ir, visited, call_expr)
        
        println(ir)

        return ir
    end

    return ir
end
