using GPUCompiler
using GPUCompiler: cached_compilation, AbstractCompilerParams
using ArrayAbstractions
using LLVM

using Base: llvmcall

# the GPU runtime library
module TestRuntime
    # dummy methods
    signal_exception() = return
    # HACK: if malloc returns 0 or traps, all calling functions (like jl_box_*)
    #       get reduced to a trap, which really messes with our test suite.
    malloc(sz) = Ptr{Cvoid}(Int(0xDEADBEEF))
    report_oom(sz) = return
    report_exception(ex) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return
end

struct TestCompilerParams <: AbstractCompilerParams end
GPUCompiler.runtime_module(::CompilerJob{<:Any,TestCompilerParams}) = TestRuntime

function native_job_with_pass(@nospecialize(func), @nospecialize(types), extra_passes; kernel::Bool=false, entry_abi=:specfunc, kwargs...)
    source = FunctionSpec(func, Base.to_tuple_type(types), kernel)
    target = NativeCompilerTarget(always_inline=true) 
    params = TestCompilerParams()
    CompilerJob(target, source, params, entry_abi, extra_passes=extra_passes), kwargs
end


function compile(func, argtype, args)
    job, _ = native_job_with_pass(func, (argtype), [ArrayAbstractions.arroptim_pass])
    mi, _ = GPUCompiler.emit_julia(job)

    ctx = JuliaContext()

    ir, ir_meta = GPUCompiler.emit_llvm(job, mi; ctx)

    compiled = ir_meta[2]
    rettype = compiled[mi].ci.rettype

    fn = LLVM.name(ir_meta.entry)
    @assert !isempty(fn)
    rettype = rettype
    ctx = JuliaContext()
    
    quote
        Base.@inline
        Base.llvmcall(($(string(ir)), $fn), $rettype, $argtype, $(args...))
    end
end

#=
function hello(x)
    println("Hello $x")
end

println(compile(hello, [Int32], [:x]))

@eval f(x) = $(compile(hello, Tuple{Int32}, [:x]))

f(Int32(10))
=#
