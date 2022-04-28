# interface with the julia compiler

using Core.Compiler: MethodInstance, CodeInfo, IRCode, @timeit

using Metatheory

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
    extra_rules::Vector{Metatheory.AbstractRule}
    
    global_cache::Union{CodeCache, CC.InternalCodeCache}
    
    # cache of inference results
    local_cache::Vector{CC.InferenceResult}

    # the world age we're working inside of
    world::UInt

    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams

    optimize::Bool

    function ArrayInterpreter(
            global_cache,
            extra_rules = Metatheory.AbstractRule[];
            world::UInt = CC.get_world_counter(),
            inf_params = CC.InferenceParams(),
            opt_params = CC.OptimizationParams(
                inlining = false,
            ),
            optimize = true
        )

        @assert world <= CC.get_world_counter()

        return new(
            extra_rules,
            global_cache,
            Vector{CC.InferenceResult}(),
            world,
            inf_params,
            opt_params,
            optimize
        )
    end
end


CC.InferenceParams(ai::ArrayInterpreter) = ai.inf_params
# Quickly and easily satisfy the AbstractInterpreter API contract
CC.OptimizationParams(ai::ArrayInterpreter) = ai.opt_params

# TODO: hack
CC.get_world_counter(ai::ArrayInterpreter) =  ai.world
CC.get_inference_cache(ai::ArrayInterpreter) = ai.local_cache

CC.code_cache(ai::ArrayInterpreter) = CC.WorldView(ai.global_cache, CC.get_world_counter(ai))

# no need to lock, not putting anything in the runtime cache
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

    @timeit "optimizer" ir = run_passes(interp, opt.src, opt, caller)

    CC.finish(interp, opt, params, ir, caller)
end

function run_passes(interp::ArrayInterpreter, ci::CodeInfo, sv::CC.OptimizationState, caller::CC.InferenceResult)
    @timeit "convert"   ir = CC.convert_to_ircode(ci, sv)
    @timeit "slot2reg"  ir = CC.slot2reg(ir, ci, sv)
    @timeit "compact 1" ir = CC.compact!(ir)

    # only optimize tagged function bodies
    perform_array_opt = CC._any(@nospecialize(x) -> CC.isexpr(x, :meta) && x.args[1] === :array_opt, ir.meta)
    if (perform_array_opt && interp.optimize)
        # Get module of the method definition
        mod = ci.parent.def.module

        # Array expression optimizing pass
        @timeit "arroptim" ir = arroptim_pass(ir, mod)

    end

    @timeit "compact 1" ir = CC.compact!(ir)
    
    @timeit "Inlining"  ir = CC.ssa_inlining_pass!(ir, ir.linetable, sv.inlining, ci.propagate_inbounds)
    # @timeit "verify 2" verify_ir(ir)
    @timeit "compact 2" ir = CC.compact!(ir)
    @timeit "SROA"      ir = CC.sroa_pass!(ir)
    @timeit "ADCE"      ir = CC.adce_pass!(ir)
    @timeit "type lift" ir = CC.type_lift_pass!(ir)
    @timeit "compact 3" ir = CC.compact!(ir)

    # TODO: make it depend on debug option
    if (true || Base.JLOptions().debug_level == 2)
        @timeit "verify 3" (CC.verify_ir(ir); CC.verify_linetable(ir.linetable))
    end

    return ir
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

export @array_opt

macro array_opt(ex)
    esc(isa(ex, Expr) ? Base.pushmeta!(ex, :array_opt) : ex)
end

function optimize(f, atype, sparams::C.SimpleVector; extra_rules=Metatheory.AbstractRule[], cache=CodeCache())
    @info "Emitting Julia"
    mi = emit_julia(f, atype, sparams)

    # create a new code cache

    @info "Performing Inference"
    # perform inference & optimizations using ArrayInterpreter
    # TODO: create new cache for Array interpreted code?
    interp = ArrayInterpreter(cache, extra_rules)

    optimize=true
    sig = CC.signature_type(f, atype)
    match::CC.MethodMatch = Base._which(sig)

    mi::CC.MethodInstance = CC.specialize_method(match)

    world_age = Base.get_world_counter()

    # compile or get from cache
    #=
    if ci_cache_lookup(cache, mi, world_age, typemax(Cint)) === nothing
        ci_cache_populate(interp, cache, mi, world_age, typemax(Cint))
    end

    code_instance = ci_cache_lookup(cache, mi, world_age, typemax(Cint))
    src = code_instance.inferred
    =#

    # ci::CodeInfo = CC.typeinf_ext_toplevel(interp, mi)
    #
    
    src = Core.Compiler.typeinf_ext_toplevel(interp, mi)
    
    # decompress if Vector{UInt8}
    if !isa(src, CodeInfo)
        src = ccall(:jl_uncompress_ir, Any, (Any, Ptr{Cvoid}, Any), mi.def, C_NULL, src::Vector{UInt8})::CodeInfo
    end

    return src

    #=
    # HACK: this is not a proper way of working with world ages (should equal the world age of the calling code?)
    =#

    # get CI via code cache, ...

    # TODO: add tests for this
    #=
    if !(ty <: AbstractArray)
        throw("Function does not return <: AbstractGPUArray")
    end
    =#

    #=
    # Note: not necessary to delete anything, we should be able to rely on the DCE pass (part of the compact! routine)
    # but it seems that the lack of a proper escape analysis? makes DCE unable to delete unused array expressions so we implement our own routine?
    =#
end
