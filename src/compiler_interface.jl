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
