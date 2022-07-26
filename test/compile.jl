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
    jlruntime::Bool=true # Use Julia runtime for throwing errors, instead of the GPUCompiler support
    aro::ArrOptimPass = ArrOptimPass()
end

GPUCompiler.llvm_triple(::ArrayNativeCompilerTarget) = Sys.MACHINE

function GPUCompiler.llvm_machine(target::ArrayNativeCompilerTarget)
    triple = GPUCompiler.llvm_triple(target)

    t = GPUCompiler.Target(triple=triple)

    tm = GPUCompiler.TargetMachine(t, triple, target.cpu, target.features)
    GPUCompiler.asm_verbosity!(tm, true)

    return tm
end

function process_entry!(job::CompilerJob{ArrayNativeCompilerTarget}, mod::LLVM.Module, entry::LLVM.Function)
    ctx = context(mod)
    if job.target.always_inline
        push!(function_attributes(entry), EnumAttribute("alwaysinline", 0; ctx))
    end
    invoke(process_entry!, Tuple{CompilerJob, LLVM.Module, LLVM.Function}, job, mod, entry)
end

## job
runtime_slug(job::CompilerJob{ArrayNativeCompilerTarget}) = "native_$(job.target.cpu)-$(hash(job.target.features))$(job.target.jlruntime ? "-jlrt" : "")"
uses_julia_runtime(job::CompilerJob{ArrayNativeCompilerTarget}) = job.target.jlruntime

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


GPUCompiler.get_interpreter(@nospecialize(job::CompilerJob{ArrayNativeCompilerTarget})) =
    ArrayInterpreter(job.target.aro, job.source.world; ci_cache = GPUCompiler.ci_cache(job))

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
    #ir = CC.inflate_ir(code_info, method_instance)

    #open("irdump","w") do f
        #print(f, ir)
    #end

    # perform custom inlining
    # ir = pass(ir, method_instance.def.module)

    #println("replacing code newstyle")
    #CC.replace_code_newstyle!(code_info, ir, Int64(method_instance.def.nargs))

    println("done!")

    return code_info
end

#=
function hello(x)
    println("hello world :) $x")
end

@eval f(x) = $(compile(hello, Tuple{{Int64}, [:x]))

f(11)
=#
