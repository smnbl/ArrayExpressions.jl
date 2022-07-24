using Core.Compiler

const CC = Core.Compiler


using Core.Compiler:
    IRCode,
    isexpr,
    getindex
# custom inlining pass

# check if instruction matches one of the rules
function inrules(inst, rules)
    # TODO: support more besides call instructions
    if !(CC.isexpr(inst, :call) || (CC.isexpr(inst, :invoke) && inst.args[2] isa GlobalRef))
        return false
    end

    for rule in rules
        if (CC.isexpr(inst, :call))
            if rule.left.operation == inst.args[1]
                return true
            end
        else
            canonical = rule.left.operation
            op = inst.args[2]
            if canonical.mod == op.mod &&
               compare(string(canonical.name), string(op.name))
                return true
            end
        end
    end

    return false
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
        # custom check if it is an intrinsic
        inst = ir.stmts[idx][:inst]
        if (isexpr(inst, :call))
            if inrules(inst, aro.extra_rules)
                println(inst)
                continue
            end
            instance = inintrinsics(inst, aro.intrinsics, idx)
            if !isnothing(instance)
                println(instance)
                continue
            end
        end

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
