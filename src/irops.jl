# extract expression that resolves to #indx SSA value
# TODO: handle phi nodes
function _extract_ir(ir::IRCode, value::SSAValue; visited=Set())
    inst = ir.stmts.inst[value.id]
    type = ir.stmts.type[value.id]

    push!(visited, value.id)

    if (inst isa Expr)
        args_op = []
        args_type = []
        start = if inst.head == :call 2 else 3 end
        for arg in inst.args[start:end]
            if arg isa SSAValue
                # TODO: stop at functions with side effects?
                visited, arir = _extract_ir(ir, arg, visited=visited)
                push!(args_op, arir.op_expr)
                push!(args_type, arir.type_expr)
            else
                println("not ssa but: $(typeof(arg))")
            end
        end

        # widenconst: converst Const(Type) -> Type ?
        # TODO: GlobalRefs have to be resolved (to Functions?)
        func = if inst.head == :call inst.args[1] else inst.args[2] end
        return visited, ArrayIR(Expr(:call, resolve(func), args_op...), Expr(:call, Symbol(CC.widenconst(type)), args_type...))
    elseif inst isa CC.PhiNode
        # TODO, atm we stop at phi-nodes
        println("phi node")
        println(inst)
        return visited, ArrayIR(Expr(:phinode), Expr(:phinode))
    elseif inst isa CC.GlobalRef
        return visited, ArrayIR(Expr(:GlobalRef), Expr(:GlobalRef))
    else
        println("not an expr, but: $(typeof(inst))")
        return visited, ArrayIR(Expr(:invalid), Expr(:invalid))
    end
end

function extract_array_ir(ir::IRCode)
    inst_stream = ir.stmts

    # keep track of seen array values (used in some chain)
    visited = Set()
    exprs = ArrayIR[]

    # start backwards
    for idx in length(inst_stream.inst):-1:1
        stmt = inst_stream.inst[idx]

        stmt isa Expr || continue
        stmt.head == :invoke || stmt.head == :call || continue
        # check if return type is StubArray and use this to confirm array ir
        rettype = CC.widenconst(ir.stmts.type[idx])
        rettype <: StubArray || continue

        print("$(idx) = ");
        visited, arir = _extract_ir(ir, SSAValue(idx), visited=visited)
        println(arir.op_expr)
        println("of type:")
        println(arir.type_expr)


        op = simplify(arir)
        println("simplified = $op")
        println("---")

        argtypes = Type[]
        argidx = SSAValue[]
    end

    println(visited)
end

