using Core.Compiler
using Metatheory
using Metatheory: AbstractRule

const CC = Core.Compiler


using Core.Compiler:
    IRCode,
    isexpr,
    getindex
# custom inlining passconst

const EMPTY_DICT = Base.ImmutableDict{Int, Any}()

function ismatch(inst, rule::AbstractRule)
    left = rule.left
    match = Metatheory.matcher(left)
    
    success(bindings, n) = n == 1 

    try
        is = !isnothing(match(success, (inst,), EMPTY_DICT))
    catch err
        throw("rewriterule match error")
    end
end

# TODO: solve the add scalar problem!
# somehow differentiate between the different types of broadcasts
# OR only want to optimize the end result?
gpu_intrinsics = [Intrinsic(GlobalRef(Base, :*), 1, [1]),
                  Intrinsic(GlobalRef(Base, :+), 1, [1])]

# check if the head is in the rules
function inrules(ir::IRCode, inst, rules)
    istree(inst) || return false
    #=
    if (operation(inst) == GlobalRef(Base, :setindex!) || operation(inst) == GlobalRef(Base, :getindex))
        return true
    end
    =#
    op = operation(inst)
    op_type = CC.widenconst(CC.argextype(op, ir))
    #println("$op::$op_type")
    return any(rule -> op == operation(rule.left) || op_type == typeof(resolve(operation(rule.left))), rules)
end

function inintrinsics(ir::IRCode, inst, intrinsics)
    istree(inst) || return false
    #=
    if (operation(inst) == GlobalRef(Base, :setindex!) || operation(inst) == GlobalRef(Base, :getindex))
        return true
    end
    =#
    # PROBLEM: sometimes * -> NNlibCUDA.:* instead of Base.:* (resolve the function objects?)
    op = operation(inst)
    op_type = CC.widenconst(CC.argextype(op, ir))
    #println("$op::$op_type")
    return any(intrinsic -> op == intrinsic.pattern || op_type == typeof(resolve(intrinsic.pattern)), intrinsics)
end

# check if instruction matches one of the rules
function inrulesexact(inst, rules)
    return any(rule -> ismatch(inst, rule) , rules)
end

function inintrinsics(inst, intrinsics, idx)
    # TODO: support more besides call instructions
    if !(CC.isexpr(inst, :call))
        return nothing
    end

    for intrinsic in intrinsics
        if (CC.isexpr(inst, :call))
            op = inst.args[1]
            if (op isa GlobalRef)
                if intrinsic.pattern == op
                    return IntrinsicInstance(idx, 1, intrinsic)
                end
            elseif op isa Function
                name = nameof(op)
                name_str = string(name)

                if (name_str[end-1:end] === "kw")
                    if op == Core.kwfunc(resolve(intrinsic.pattern))
                        return IntrinsicInstance(idx, 3, intrinsic)
                    end
                end

                # TODO: is this fuzzy match ok?
                if parentmodule(op) == intrinsic.pattern.mod &&
                    compare(name_str, string(intrinsic.pattern.name))
                    return IntrinsicInstance(idx, 1, intrinsic)
                end
            end
        end
    end

    return nothing
end

function compare(canonical_name::String, derived_name::String)
    pred = x -> x == '#' || isnumeric(x)
    return canonical_name == strip(pred, derived_name)
end

# -> not inlining when flagged as intrinsic
function custom_assemble_inline_todo!(ir::IRCode, state::CC.InliningState, aro)
    # todo = (inline_idx, (isva, isinvoke, na), method, spvals, inline_linetable, inline_ir, lie)
    todo = Pair{Int, Any}[]
    et = state.et

    for idx in 1:length(ir.stmts)
        simpleres = CC.process_simple!(ir, idx, state, todo)
        simpleres === nothing && continue
        stmt, sig = simpleres

        info = ir.stmts[idx][:info]

        # Check whether this call was @pure and evaluates to a constant
        if info isa CC.MethodResultPure
            CC.inline_const_if_inlineable!(getindex(ir, SSAValue(idx))) && continue
            info = info.info
        end
        if info === false
            # Inference determined this couldn't be analyzed. Don't question it.
            continue
        end

        flag = ir.stmts[idx][:flag]

        if isa(info, CC.OpaqueClosureCallInfo)
            result = info.result
            if isa(result, CC.InferenceResult)
                CC.handle_const_opaque_closure_call!(
                    ir, idx, stmt, result, flag,
                    sig, state, todo)
            else
                if isa(result, CC.ConstResult)
                    item = CC.const_result_item(result, state)
                else
                    item = CC.analyze_method!(info.match, sig.argtypes, flag, state)
                end
                CC.handle_single_case!(ir, idx, stmt, item, todo, state.params)
            end
            continue
        end

        # custom check if it is an intrinsic
        if inrules(ir, stmt, (aro.extra_rules)) || inintrinsics(ir, stmt, gpu_intrinsics)
            continue
        end

        # Handle invoke
        if sig.f === Core.invoke
            if isa(info, CC.InvokeCallInfo)
                CC.inline_invoke!(ir, idx, stmt, info, flag, sig, state, todo)
            end
            continue
        end

        # if inference arrived here with constant-prop'ed result(s),
        # we can perform a specialized analysis for just this case
        if isa(info, CC.ConstCallInfo)
            CC.handle_const_call!(
                ir, idx, stmt, info, flag,
                sig, state, todo)
            continue
        end

        # Ok, now figure out what method to call
        if isa(info, CC.MethodMatchInfo)
            infos = CC.MethodMatchInfo[info]
        elseif isa(info, CC.UnionSplitInfo)
            infos = info.matches
        else
            continue # isa(info, ReturnTypeCallInfo), etc.
        end

        CC.analyze_single_call!(ir, idx, stmt, infos, flag, sig, state, todo)
    end
    todo
end

function custom_ssa_inlining_pass!(ir::IRCode, linetable::Vector{CC.LineInfoNode}, state::CC.InliningState, propagate_inbounds::Bool, aro)
    # Go through the function, performing simple ininlingin (e.g. replacing call by constants
    # and analyzing legality of inlining).
    todo = custom_assemble_inline_todo!(ir, state, aro)
    CC.isempty(todo) && return ir
    # Do the actual inlining for every call we identified
    ir = CC.batch_inline!(todo, ir, linetable, propagate_inbounds, state.params)
    return ir
end
