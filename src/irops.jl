const CC = Core.Compiler
using Core.Compiler: iterate
using CUDA

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

            if arg isa SSAValue # && arg_type <: Union{CuArray, Number}
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
function arroptim_pass(ir::IRCode, mod)
    perform_array_opt = CC._any(@nospecialize(x) -> CC.isexpr(x, :meta) && x.args[1] === :array_opt, ir.meta)
    if (!perform_array_opt)
        # do nothing
        return ir
    end
    println("arr optimizing")

    println(ir)
    
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

        expr, input_map = codegen_expr!(op.args[1], length(ir.argtypes))

        println("compiling opaque closure:")

        oc = Core.eval(mod, Expr(:opaque_closure, expr))

        arguments = Array{Any}(undef, length(input_map))
        for (val, idx) in pairs(input_map)
            println("$(val), $(typeof(val))")
            arguments[idx] = val
        end
        
        compact = CC.IncrementalCompact(ir, true)                

        call_expr =  Expr(:call, oc, arguments...)
        ssaval = CC.insert_node!(compact, loc, CC.non_effect_free(CC.NewInstruction(call_expr, rettype)), true)
        ssaval = CC.insert_node!(compact, loc, CC.non_effect_free(CC.NewInstruction(CC.ReturnNode(ssaval), rettype)), true)

        next = CC.iterate(compact)
        while (next != nothing)
            next = CC.iterate(compact, next[2])
        end

        ir = CC.finish(compact)

        println(ir)

        return ir
    end

    return ir
end
