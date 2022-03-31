const CC = Core.Compiler

# types of values that are considered intermediate
const ValueTypes = Union{AbstractGPUArray, Base.Broadcast.Broadcasted, CC.SSAValue, Matrix, Number}

# extract expression that resolves to #indx SSA value
# TODO: handle phi nodes -> atm: stop at phi nodes (diverging control flow)
# visited: set of already visited notes
function _extract_slice(ir::IRCode, loc::SSAValue; visited=Set())
    inst = ir.stmts[loc.id][:inst]
    type = ir.stmts[loc.id][:type]

    push!(visited, loc.id)
    inputs = Set{InputTypes}()

    if CC.isexpr(inst, :call) || CC.isexpr(inst, :invoke)
        args_op = []
        start = if inst.head == :call 2 else 3 end

        for arg in inst.args[start:end]
            # TODO: v1.8 already has a version that works on IRCode, without explicitly passing sptypes & argtypes
            type = CC.widenconst(CC.argextype(arg, ir, ir.sptypes, ir.argtypes))

            if arg isa SSAValue # arg isa Core.Argument
                # TODO: stop at functions with side effects?
                # TODO: look at Julia's native purity modeling infra (part of Julia v1.8): https://github.com/JuliaLang/julia/pull/43852
                visited, arir, extra_inputs = _extract_slice(ir, arg, visited=visited)
                push!(args_op, arir.op_expr)
                union!(inputs, extra_inputs)
            else
                push!(args_op, ArrayExpr(:call, [:input, arg], type))
                push!(inputs, arg)
            end
        end

        # widenconst: converst Const(Type) -> Type ?
        # TODO: GlobalRefs have to be resolved (to Functions?)
        func = if inst.head == :call inst.args[1] else inst.args[2] end
        return visited, ArrayIR(ArrayExpr(:call, [resolve(func), args_op...], CC.widenconst(type))), inputs
    elseif inst isa CC.PhiNode || inst isa CC.GlobalRef
        push!(inputs, loc)
        type = CC.widenconst(CC.argextype(inst, ir, ir.sptypes, ir.argtypes))
        return visited, ArrayIR(ArrayExpr(:call, [:input, inst], type)), inputs
    elseif inst isa CC.SlotNumber

    else
        println("not an expr, but: $inst::$(typeof(inst))")
        return visited, ArrayIR(ArrayExpr(:call, [:invalid], Union{})), inputs
    end
end

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

    # TODO: check if this is a sensible way to nop instructions?
    # TODO: decrease number of uses, remove if uses == 0 ~ DCE pass
    # ir.stmts.inst[loc.id] = nothing
end

function extract_slice(ir::IRCode)
    stmts = ir.stmts

    # set  of already visited array SSA values (as end results)
    visited = Set()
    exprs = ArrayIR[]

    # start backwards
    for idx in length(stmts):-1:1
        stmt = stmts[idx][:inst]
        loc = SSAValue(idx)

        stmt isa Expr || continue
        stmt.head == :invoke || stmt.head == :call || continue
        # check if return type is StubArray and use this to confirm array ir
        rettype = CC.widenconst(stmts[idx][:type])

        rettype <: ValueTypes || continue

        print("$(idx) = ");
        visited, arir, inputs = _extract_slice(ir, loc, visited=visited)

        println(arir.op_expr)

        println(">>>")

        op = simplify(arir.op_expr)
        println("simplified = $op")
        println("---")

        println(op)

        println("codegen:")
        expr = codegen(op, inputs)

        # Note: not necessary to delete anything, we should be able to rely on the DCE pass (part of the compact! routine)
        # but it seems that the lack of a proper escape analysis? makes DCE unable to delete unused array expressions so we implement our own routine?
        _delete_expr_ir!(ir, loc, inputs)

         println("insert optimized instructions")
         #TODO: make this runnable; eg replace with an OC or similar
         stmts.inst[idx] = expr

         return expr
    end
end

