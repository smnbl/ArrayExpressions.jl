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

# Quickly and easily satisfy the AbstractInterpreter API contract
CC.InferenceParams(ai::ArrayInterpreter) = ai.inf_params
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

function CC.optimize(interp::ArrayInterpreter, opt::CC.OptimizationState, params::CC.OptimizationParams, @nospecialize(result))
    nargs = Int(opt.nargs) - 1

    # TODO: interface changed in 1.8
    ir = CC.run_passes(opt.src, nargs, opt)

    perform_array_opt = CC._any(@nospecialize(x) -> CC.isexpr(x, :meta) && x.args[1] === :array_opt, ir.meta)

    if (perform_array_opt)
        println("array_opt")
        extract_slice(ir)
    end

    CC.finish(interp, opt, params, ir, result)
end

## LLVM IR Generation

## TODO: idea to trap inference -> reset typeinf_func using jl_set_typeinf_func
## HACK: you're not supposed to to it this way i guess :P
typeinf(mi::MethodInstance, world::UInt) = CC.typeinf_ext_toplevel(ArrayInterpreter(world), mi)
function inject_typeinf()
    ccall(:jl_set_typeinf_func, Cvoid, (Any,), typeinf)
end

function codegen(output::Symbol, f, atype, sparams::C.SimpleVector)
    @info "Emitting Julia"
    mi = emit_julia(f, atype, sparams)

    output == :julia && return Base.uncompressed_ir(mi.def)

    @info "Performing Inference"

    # perform inference & optimizations using ArrayInterpreter
    interp = ArrayInterpreter()
    src::CodeInfo = CC.typeinf_ext(interp, mi)

    output == :typed && return src

    # get code instance & replace inferred src with 
    # TODO: this seems wrong?; might seem to work because compilation is only done after calling the CodeInstance object?
    # TODO: replace this by an opaque closuer performing the operations / a GPU kernel performing the operations
    code::CC.CodeInstance = CC.getindex(CC.code_cache(interp), mi)
    code.inferred = src

    # TODO:
    # 1. populate cache with code
    # 2. generate native IR
end
