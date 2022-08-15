using GPUCompiler
using GPUCompiler:
    cached_compilation,
    AbstractCompilerParams,
    GPUInterpreter
using ArrayAbstractions
using LLVM

using Core.Compiler
const CC = Core.Compiler

using Base: llvmcall

# the GPU runtime library
module TestRuntime
    # dummy methods
    signal_exception() = return
    malloc(sz) = ccall("extern malloc", llvmcall, Csize_t, (Csize_t,), sz)
    report_oom(sz) = return
    report_exception(ex) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return
end

struct TestCompilerParams <: AbstractCompilerParams end
GPUCompiler.runtime_module(::CompilerJob{<:Any,TestCompilerParams}) = TestRuntime

Base.@kwdef struct ArrayNativeCompilerTarget <: GPUCompiler.AbstractCompilerTarget
    cpu::String=(LLVM.version() < v"8") ? "" : unsafe_string(LLVM.API.LLVMGetHostCPUName())
    features::String=(LLVM.version() < v"8") ? "" : unsafe_string(LLVM.API.LLVMGetHostCPUFeatures())
    always_inline::Bool=false # will mark the job function as always inline
    jlruntime::Bool=true# Use Julia runtime for throwing errors, instead of the GPUCompiler support
    aro::ArrOptimPass = ArrOptimPass()
    cache = AA.CodeCache()
end

GPUCompiler.llvm_triple(::ArrayNativeCompilerTarget) = Sys.MACHINE

function GPUCompiler.llvm_machine(target::ArrayNativeCompilerTarget)
    triple = GPUCompiler.llvm_triple(target)

    t = GPUCompiler.Target(triple=triple)

    tm = GPUCompiler.TargetMachine(t, triple, target.cpu, target.features)
    GPUCompiler.asm_verbosity!(tm, true)

    return tm
end

function GPUCompiler.process_entry!(job::CompilerJob{ArrayNativeCompilerTarget}, mod::LLVM.Module, entry::LLVM.Function)
    ctx = GPUCompiler.context(mod)
    if job.target.always_inline
        push!(function_attributes(entry), EnumAttribute("alwaysinline", 0; ctx))
    end
    invoke(GPUCompiler.process_entry!, Tuple{CompilerJob, LLVM.Module, LLVM.Function}, job, mod, entry)
end

## job
GPUCompiler.runtime_slug(job::CompilerJob{ArrayNativeCompilerTarget}) = "native_$(job.target.cpu)-$(hash(job.target.features))$(job.target.jlruntime ? "-jlrt" : "")"
GPUCompiler.uses_julia_runtime(job::CompilerJob{ArrayNativeCompilerTarget}) = job.target.jlruntime

function native_job_with_pass(@nospecialize(func), @nospecialize(types), aro::ArrOptimPass; kernel::Bool=false, entry_abi=:specfunc, kwargs...)
    source = FunctionSpec(func, Base.to_tuple_type(types), kernel)
    target = ArrayNativeCompilerTarget(aro = aro, always_inline=true)
    params = TestCompilerParams()

    function end_pass(code_info::CC.CodeInfo, mi::CC.MethodInstance)
        ir = CC.inflate_ir(code_info, mi)
        ir = aro(ir, mi.def.module)

        CC.replace_code_newstyle!(code_info, ir, Int64(mi.def.nargs))

        return code_info
    end
    CompilerJob(target, source, params, entry_abi, end_pass=end_pass, always_inline=false), kwargs
end

# use fresh code cache for each job
GPUCompiler.ci_cache(job::CompilerJob{ArrayNativeCompilerTarget}) = job.target.cache

GPUCompiler.get_interpreter(job::CompilerJob{ArrayNativeCompilerTarget}) =
    ArrayInterpreter(job.target.aro, job.source.world; ci_cache = GPUCompiler.ci_cache(job))

using GPUCompiler:
@timeit_debug, compile_method_instance, functions, @compiler_assert, safe_name, ModulePassManager, linkage!, internalize!, always_inliner!, can_throw, add!, add_lowering_passes!, run!, process_module!, process_entry!, mangle_call

# irgen with extra end pass
function GPUCompiler.irgen(job::CompilerJob{ArrayNativeCompilerTarget}, method_instance::Core.MethodInstance;
               ctx::Context)
    mod, compiled = @timeit_debug to "emission" compile_method_instance(job, method_instance; ctx)

    println("performing at end pass")

    ci = compiled[method_instance].ci

    src = if ci.inferred isa Vector{UInt8}
        ccall(:jl_uncompress_ir, Any, (Any, Ptr{Cvoid}, Any),
                method_instance.def, C_NULL, ci.inferred)
    else
        ci.inferred
    end

    # TODO: compress back?
    ci.inferred = job.target.aro(src, method_instance) 

    if job.entry_abi === :specfunc
        entry_fn = compiled[method_instance].specfunc
    else
        entry_fn = compiled[method_instance].func
    end

    # clean up incompatibilities
    @timeit_debug to "clean-up" begin
        for llvmf in functions(mod)
            # only occurs in debug builds
            delete!(function_attributes(llvmf), EnumAttribute("sspstrong", 0; ctx))

            if Sys.iswindows()
                personality!(llvmf, nothing)
            end

            # remove the non-specialized jfptr functions
            # TODO: Do we need to remove these?
            if job.entry_abi === :specfunc
                if startswith(LLVM.name(llvmf), "jfptr_")
                    unsafe_delete!(mod, llvmf)
                end
            end
        end

        # remove the exception-handling personality function
        if Sys.iswindows() && "__julia_personality" in functions(mod)
            llvmf = functions(mod)["__julia_personality"]
            @compiler_assert isempty(uses(llvmf)) job
            unsafe_delete!(mod, llvmf)
        end
    end

    # target-specific processing
    process_module!(job, mod)
    entry = functions(mod)[entry_fn]

    # sanitize function names
    # FIXME: Julia should do this, but apparently fails (see maleadt/LLVM.jl#201)
    for f in functions(mod)
        LLVM.isintrinsic(f) && continue
        llvmfn = LLVM.name(f)
        startswith(llvmfn, "julia.") && continue # Julia intrinsics
        startswith(llvmfn, "llvm.") && continue # unofficial LLVM intrinsics
        llvmfn′ = safe_name(llvmfn)
        if llvmfn != llvmfn′
            @assert !haskey(functions(mod), llvmfn′)
            LLVM.name!(f, llvmfn′)
        end
    end

    # rename and process the entry point
    if job.source.name !== nothing
        LLVM.name!(entry, safe_name(string("julia_", job.source.name)))
    end
    if job.source.kernel
        LLVM.name!(entry, mangle_call(entry, job.source.tt))
    end
    entry = process_entry!(job, mod, entry)
    if job.entry_abi === :specfunc
        func = compiled[method_instance].func
        specfunc = LLVM.name(entry)
    else
        func = LLVM.name(entry)
        specfunc = compiled[method_instance].specfunc
    end

    compiled[method_instance] =
        (; compiled[method_instance].ci, func, specfunc)

    # minimal required optimization
    @timeit_debug to "rewrite" ModulePassManager() do pm
        global current_job
        current_job = job

        linkage!(entry, LLVM.API.LLVMExternalLinkage)

        # internalize all functions, but keep exported global variables
        exports = String[LLVM.name(entry)]
        for gvar in globals(mod)
            push!(exports, LLVM.name(gvar))
        end
        internalize!(pm, exports)

        # inline llvmcall bodies
        always_inliner!(pm)

        can_throw(job) || add!(pm, ModulePass("LowerThrow", lower_throw!))

        add_lowering_passes!(job, pm)

        run!(pm, mod)
    end

    return mod, compiled
end

function compile_with_gpucompiler(func, argtype, args; eltype=AbstractArray, extra_rules=[], intrinsics=[])
    pass = ArrOptimPass(eltype, extra_rules=extra_rules, intrinsics=intrinsics)

    # don't run a pass in the loop
    job, _ = native_job_with_pass(func, (argtype), pass; kernel=false)
    mi, _ = GPUCompiler.emit_julia(job)
    println("emitting julia done...")

    ctx = JuliaContext()

    println("inferring & emitting llvm code")
    ir, ir_meta = GPUCompiler.emit_llvm(job, mi; ctx, libraries=false)

    println("emitting llvm done")

    compiled = ir_meta[2]
    rettype = compiled[mi].ci.rettype

    fn = LLVM.name(ir_meta.entry)
    @assert !isempty(fn)
    
    quote
        Base.@inline
        Base.llvmcall(($(string(ir)), $fn), $rettype, $argtype, $(args...))
    end
end

# IDEA: stop inlining when coming across an 'intrinsic'
function compile_expression(func, argtype, args; eltype=AbstractArray, extra_rules=[], intrinsics=[])
    pass = ArrOptimPass(eltype, extra_rules=extra_rules, intrinsics=intrinsics)

    world_count = Base.get_world_counter()

    # for internal passes -> need for custom interpreter
    interpreter = ArrayInterpreter(pass)
    code_info, ty = Base.code_typed(func, argtype, interp = interpreter)[1]

    # get method instance
    meth = which(func, argtype)
    sig = Base.signature_type(func, argtype)::Type
    (ti, env) = ccall(:jl_type_intersection_with_env, Any,
                        (Any, Any), sig, meth.sig)::Core.SimpleVector
    meth = Base.func_for_method_checked(meth, ti, env)
    method_instance = ccall(:jl_specializations_get_linfo, Ref{Core.MethodInstance},
                    (Any, Any, Any, UInt), meth, ti, env, world_count)


    #println("performing opt pass...")
    ir = CC.inflate_ir(code_info, method_instance)

    println(code_info)

    println("performing the pass at the end!")
    ir = pass(ir, method_instance.def.module)

    println("replacing code newstyle")
    CC.replace_code_newstyle!(code_info, ir, Int64(method_instance.def.nargs))

    println(code_info)

    println("done!")

    return code_info
end

# compile and put inside an opaque closure
function compile_oc()

end

#=
function hello(x)
    println("hello world :) $x")
end

@eval f(x) = $(compile(hello, Tuple{{Int64}, [:x]))

f(11)
=#
