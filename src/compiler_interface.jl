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

macro array_opt(ex)
    esc(isa(ex, Expr) ? Base.pushmeta!(ex, :array_opt) : ex)
end

function CC.optimize(interp::ArrayInterpreter, opt::CC.OptimizationState, params::CC.OptimizationParams, caller::CC.InferenceResult)
    ci = opt.src
    sv = opt

    # run some passes, but not the inlining pass
    ir = CC.convert_to_ircode(ci, sv)
    ir = CC.slot2reg(ir, ci, sv)
    ir = CC.compact!(ir)

    perform_array_opt = CC._any(@nospecialize(x) -> CC.isexpr(x, :meta) && x.args[1] === :array_opt, ir.meta)

    if (perform_array_opt)
        # canonicalize to SSA IR
        # TODO: interface changed in 1.8
        println("array_opt")
        ir = extract_slice(ir)
    end

    # TODO: interface changed in 1.8
    ir = CC.compact!(ir)

    # verify ir
    CC.verify_ir(ir)

    CC.finish(interp, opt, params, ir, caller)
end

function codegen(output::Symbol, f, atype, sparams::C.SimpleVector)
    @info "Emitting Julia"
    mi = emit_julia(f, atype, sparams)
    output == :julia && return Base.uncompressed_ir(mi.def)

    @info "Performing Inference"
    # perform inference & optimizations using ArrayInterpreter
    interp = ArrayInterpreter()

    # perform inference & optimizations using ArrayInterpreter
    interp = ArrayInterpreter()
    
    ci::CodeInfo = CC.typeinf_ext(interp, mi)
    println(ci)

    output == :typed && return ci

    # get code instance & replace inferred src with (populate the cache)
    # TODO: this seems wrong?; might seem to work because compilation is only done after calling the CodeInstance object?
    # TODO: replace this by an opaque closuer performing the operations / a GPU kernel performing the operations
    code_instance::CC.CodeInstance = CC.getindex(CC.code_cache(interp), mi)
    code_instance.inferred = ci
end
