# interface with the julia compiler

using Core.Compiler: MethodInstance, CodeInfo, IRCode

const C = Core
const CC = Core.Compiler

export @array_opt

# JULIA IR
function emit_julia(f, @nospecialize(atype), sparams::C.SimpleVector)
    sig = CC.signature_type(f, atype)
    meth::CC.MethodMatch = Base._which(sig)

    mi::MethodInstance = CC.specialize_method(meth)
    return mi
end

# INFERENCE + CODEGEN
struct ArrayInterpreter <: CC.AbstractInterpreter
    # cache of inference results
    cache::Vector{CC.InferenceResult}

    # the world age we're working inside of
    world::UInt

    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams

    function ArrayInterpreter(
            world::UInt = CC.get_world_counter();
            inf_params = CC.InferenceParams(),
            opt_params = CC.OptimizationParams(
                            inlining = false,
                                              ),
        )

        @assert world <= CC.get_world_counter()

        return new(
                   Vector{CC.InferenceResult}(),
                   world,
                   inf_params,
                   opt_params,
                  )
    end
end


CC.InferenceParams(ai::ArrayInterpreter) = ai.inf_params
# Quickly and easily satisfy the AbstractInterpreter API contract
CC.OptimizationParams(ai::ArrayInterpreter) = ai.opt_params
CC.get_world_counter(ai::ArrayInterpreter) = ai.world
CC.get_inference_cache(ai::ArrayInterpreter) = ai.cache

# re-use the global cache, but limit the view using WorldView
# WorldView limits range of valid cache contents to cache entries that have world_age == world_counter
# TODO: built own cache to not infere with NativeInterpreter's global code_cache
CC.code_cache(ai::ArrayInterpreter) = CC.WorldView(CC.GLOBAL_CI_CACHE, CC.get_world_counter(ai))

# hints that a certain MethodInstance is in inference to prevent inferring the same thing multiple times?
CC.lock_mi_inference(ai::ArrayInterpreter, mi::MethodInstance) = (mi.inInference = true; nothing)
CC.unlock_mi_inference(ai::ArrayInterpreter, mi::MethodInstance) = (mi.inInference = false; nothing)

CC.may_optimize(::ArrayInterpreter) = true
CC.may_compress(::ArrayInterpreter) = true
CC.may_discard_trees(::ArrayInterpreter) = true
CC.verbose_stmt_info(::ArrayInterpreter) = false

# TODO: can be used to add analysis remarks during inference for the current line
function CC.add_remark!(ai::ArrayInterpreter, sv::CC.InferenceState, msg)
    @debug "Inference remark during array compilation of $(sv.linfo): $msg"
end

function CC.optimize(interp::ArrayInterpreter, opt::CC.OptimizationState,
                  params::CC.OptimizationParams, caller::CC.InferenceResult)

    # TODO: support v1.8 interface
    ci = opt.src
    sv = opt

    # run some passes, but not the inlining pass
    ir = CC.convert_to_ircode(ci, sv)
    ir = CC.slot2reg(ir, ci, sv) # convert to SSA form
    ir = CC.compact!(ir)

    CC.finish(interp, opt, params, ir, caller)
end

function OC(ir::IRCode, nargs::Int, isva::Bool, env...)
    if (isva && nargs > length(ir.argtypes)) || (!isva && nargs != length(ir.argtypes))
        throw(ArgumentError("invalid argument count"))
    end
    src = ccall(:jl_new_code_info_uninit, Ref{CodeInfo}, ())
    src.slotflags = UInt8[]
    src.slotnames = fill(:none, nargs+1)
    Core.Compiler.replace_code_newstyle!(src, ir, nargs+1)
    Core.Compiler.widen_all_consts!(src)
    src.inferred = true
    # NOTE: we need ir.argtypes[1] == typeof(env)

    ccall(:jl_new_opaque_closure_from_code_info, Any, (Any, Any, Any, Any, Any, Cint, Any, Cint, Cint, Any),
          Tuple{ir.argtypes[2:end]...}, Union{}, Any, @__MODULE__, src, 0, nothing, nargs - 1, isva, env)
end

function optimize(output::Symbol, f, atype, sparams::C.SimpleVector; extra_rules=Metatheory.AbstractRule[])
    @info "Emitting Julia"
    mi = emit_julia(f, atype, sparams)
    output == :julia && return Base.uncompressed_ir(mi.def)

    @info "Performing Inference"
    # perform inference & optimizations using ArrayInterpreter
    interp = ArrayInterpreter()

    optimize=true
    sig = CC.signature_type(f, atype)
    match::CC.MethodMatch = Base._which(sig)
    (ci::CodeInfo, ty) = Core.Compiler.typeinf_code(interp, match.method, match.spec_types, match.sparams, optimize)

    # TODO: add tests for this
    if !(ty <: AbstractArray)
        throw("Function does not return <: AbstractGPUArray")
    end

    ir = CC.inflate_ir(ci)

    println(ir)

    # extract & optimize array expression that returns
    arir = extract_slice(ir, SSAValue(length(ir.stmts) - 1))

    println(arir.op_expr)

    println(">>>")

    op = simplify(arir.op_expr, extra_rules=extra_rules)
    println("simplified = $op")
    println("---")

    println(op)

    expr = codegen_expr!(op, length(ir.argtypes))
    return expr

    #=
    # Note: not necessary to delete anything, we should be able to rely on the DCE pass (part of the compact! routine)
    # but it seems that the lack of a proper escape analysis? makes DCE unable to delete unused array expressions so we implement our own routine?
    _delete_expr_ir!(ir, loc, inputs)

    new_ssa = codegen_ssa!(op)

    println("insert optimized instructions")
    println(new_ssa)

    ssaval = nothing

    compact = CC.IncrementalCompact(ir, true)
    for stmt in new_ssa[1:end-1]
        ssaval = CC.insert_node!(compact, loc, CC.effect_free(CC.NewInstruction(stmt, Any)))
        # TODO: do sth with ssaval
    end

    ir.stmts.inst[idx] = new_ssa[end]

    next = CC.iterate(compact)
    while (next != nothing)
        next = CC.iterate(compact, next[2])
    end

    ir = CC.finish(compact)

    # will throw on errors
    CC.verify_ir(ir)

    oc = OC(ir, length(ir.argtypes), false)
    return oc
    =#
end
