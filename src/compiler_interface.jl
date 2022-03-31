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

function CC.optimize(interp::ArrayInterpreter, opt::CC.OptimizationState, params::CC.OptimizationParams, @nospecialize(result))
    # TODO: support v1.8 interface
    ci = opt.src
    sv = opt
    nargs = Int(opt.nargs) - 1
    preserve_coverage = CC.coverage_enabled(sv.mod)

    # run some passes, but not the inlining pass
    ir = CC.convert_to_ircode(ci, CC.copy_exprargs(ci.code), preserve_coverage, nargs, sv)
    ir = CC.slot2reg(ir, ci, nargs, sv)
    ir = CC.compact!(ir)

    CC.finish(interp, opt, params, ir, result)
end

function optimize(output::Symbol, f, atype, sparams::C.SimpleVector)
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
    if !(ty <: AbstractGPUArray)
        throw("Function does not return <: AbstractGPUArray")
    end

    ir = CC.inflate_ir(ci)

    println(ir)

    # extract & optimize array expression
    expr = extract_slice(ir)
    println(expr)

    # evaluate expession
    return expr
end
