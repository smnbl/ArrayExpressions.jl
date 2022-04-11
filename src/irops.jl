const CC = Core.Compiler
using Core.Compiler: iterate

# types of values that are considered intermediate
const ValueTypes = Union{AbstractGPUArray, Base.Broadcast.Broadcasted, CC.SSAValue, Matrix, Number}

# extract expression that resolves to #indx SSA value
# TODO: handle phi nodes -> atm: stop at phi nodes (diverging control flow)
# visited: set of already visited notes
function _extract_slice(ir::IRCode, loc::SSAValue; visited=Set())
    inst = ir.stmts[loc.id][:inst]
    type = ir.stmts[loc.id][:type]

    push!(visited, loc)
    inputs = Set{InputTypes}()

    if CC.isexpr(inst, :call) || CC.isexpr(inst, :invoke)
        args_op = []
        start = if inst.head == :call 2 else 3 end
        args = inst.args[start:end]

        for arg in args
            # TODO: v1.8 already has a version that works on IRCode, without explicitly passing sptypes & argtypes
            type = CC.widenconst(CC.argextype(arg, ir, ir.sptypes, ir.argtypes))

            if arg isa SSAValue # arg isa Core.Argument
                # TODO: stop at functions with side effects?
                # TODO: look at Julia's native purity modeling infra (part of Julia v1.8): https://github.com/JuliaLang/julia/pull/43852
                visited, arir, extra_inputs = _extract_slice(ir, arg, visited=visited)
                push!(args_op, arir.op_expr)
                union!(inputs, extra_inputs)
            else
                push!(args_op, resolve(arg))
                push!(inputs, arg)
            end
        end

        func = if inst.head == :call inst.args[1] else inst.args[2] end
        return visited, ArrayIR(ArrayExpr(:call, [resolve(func), args_op...], type)), inputs
    # inference barrier, inputs are SSAValues
    elseif inst isa CC.GlobalRef || inst isa CC.PhiNode
        push!(inputs, loc)
        type = CC.widenconst(CC.argextype(inst, ir, ir.sptypes, ir.argtypes))
        return visited, loc, inputs
    elseif inst isa CC.SlotNumber

    else
        println("not an expr, but: $inst::$(typeof(inst))")
        return visited, ArrayIR(ArrayExpr(:call, [:invalid], Union{})), inputs
    end
end

# NOTE: atm this is not used!
# delete all the instructions that are parted of the expression that gets replaced, up until the points of input
# goes up the use chain to replace all the instructions with nops
function _delete_expr_ir!(ir::IRCode, loc::SSAValue, inputs::Set{InputTypes})
    inst = ir.stmts[loc.id][:inst]
    start = if inst.head == :call 2 else 3 end

    for arg in inst.args[start:end]
        # if arg is GlobalRef, we can assume it is already ∈ inputs
        if arg isa SSAValue && arg ∉ inputs
            _delete_expr_ir!(ir, arg, inputs)
        end
    end

    # TODO: check if this is a sensible way to nop instructions, it seems so as simple_dce does this?
    # TODO: decrease number of uses, remove if uses == 0 ~ DCE pass
    ir.stmts.inst[loc.id] = nothing
    # ir.stmts.type[loc.id] = Nothing
end

function extract_slice(ir::IRCode, loc::SSAValue)
    _, arir, _ = _extract_slice(ir, loc, visited=Set())
    return arir
end

function optimize_ir(ir::IRCode)
    stmts = ir.stmts

    # start backwards
    for idx in length(stmts):-1:1
        loc = SSAValue(idx)

        loc ∉ visited || continue
        stmt isa Expr || continue

        # check if return type is StubArray and use this to confirm array ir
        rettype = CC.widenconst(stmts[idx][:type])
        rettype <: ValueTypes || continue

        # TODO: optimize whole ir body iteratively
    end
end

