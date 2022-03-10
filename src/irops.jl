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
        args_type = []
        start = if inst.head == :call 2 else 3 end

        for arg in inst.args[start:end]
            # TODO: v1.8 already has a version that works on IRCode, without explicitly passing sptypes & argtypes
            type = CC.widenconst(CC.argextype(arg, ir, ir.sptypes, ir.argtypes))
            if !(type <: AbstractGPUArray)
                push!(args_op, Expr(:input, arg))
                push!(args_type, Expr(:input, type))
                push!(inputs, arg)
            else
                # TODO: stop at functions with side effects?
                # TODO: look at Julia's native purity modeling infra (part of Julia v1.8): https://github.com/JuliaLang/julia/pull/43852
                visited, arir, extra_inputs = _extract_slice(ir, arg, visited=visited)
                push!(args_op, arir.op_expr)
                push!(args_type, arir.type_expr)
                union!(inputs, extra_inputs)
            end
        end

        # widenconst: converst Const(Type) -> Type ?
        # TODO: GlobalRefs have to be resolved (to Functions?)
        func = if inst.head == :call inst.args[1] else inst.args[2] end
        return visited, ArrayIR(Expr(:call, resolve(func), args_op...), Expr(:call, Symbol(CC.widenconst(type)), args_type...)), inputs
    elseif inst isa CC.PhiNode || inst isa CC.GlobalRef
        push!(inputs, loc)
        type = CC.widenconst(CC.argextype(inst, ir, ir.sptypes, ir.argtypes))
        return visited, ArrayIR(Expr(:input, inst), Expr(:input, type)), inputs
    else
        println("not an expr, but: $inst::$(typeof(inst))")
        return visited, ArrayIR(Expr(:invalid), Expr(:invalid)), inputs
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

# opaque closuers are supported from Julia v1.8 onwards
function OC(ir::IRCode, arg1::Any)
    src = ccall(:jl_new_code_info_uninit, Ref{CodeInfo}, ())
    src.slotflags = UInt8[]
    src.slotnames = Symbol[]
    nargs = length(ir.argtypes)
    Core.Compiler.replace_code_newstyle!(src, ir, nargs)
    Core.Compiler.widen_all_consts!(src)
    src.inferred = true

    m = ccall(:jl_make_opaque_closure_method, Ref{Method}, (Any, Any, Any, Any, Any),
              @__MODULE__, nothing, nargs, Core.LineNumberNode(0, nothing), src)

    rarg1 = Ref{Any}(arg1)
    ccall(:jl_new_opaque_closure, Any, (Any, Any, Any, Any, Any, Any, Csize_t),
          Tuple{ir.argtypes[2:end]...}, false, Union{}, Any, m, rarg1, 1)::Core.OpaqueClosure
end

function extract_slice(ir::IRCode)
    stmts = ir.stmts

    # set  of already visited array SSA values (as end results)
    visited = Set()
    exprs = ArrayIR[]

    println(ir.argtypes)

    #println(ir)

    # start backwards
    for idx in length(stmts):-1:1
        stmt = stmts[idx][:inst]
        loc = SSAValue(idx)

        stmt isa Expr || continue
        stmt.head == :invoke || stmt.head == :call || continue
        # check if return type is StubArray and use this to confirm array ir
        rettype = CC.widenconst(stmts[idx][:type])
        rettype <: AbstractGPUArray || continue

        print("$(idx) = ");
        visited, arir, inputs = _extract_slice(ir, loc, visited=visited)

        println(arir.op_expr)
        println("of type:")
        println(arir.type_expr)


        println(">>>")
        println("$inputs")

        op = simplify(arir)
        println("simplified = $op")
        println("---")


        # Note: not necessary to delete anything, we should be able to rely on the DCE pass (part of the compact! routine)
        # but it seems that the lack of a proper escape analysis? makes DCE unable to delete unused array expressions so we implement our own routine?

        println("insert optimized instructions")
        # TODO: make this runnable; eg replace with an OC or similar
        stmts.inst[idx] = Expr(:call, :get_jl_array)

        # TODO: compact & verify IR :)
        println("compacting ir")
        ir = CC.compact!(ir, true)
        println(ir)

        # this throws if sth is wrong with the ir
        CC.verify_ir(ir)

        argtypes = Type[]
        argidx = SSAValue[]
    end

    println(visited)
end

