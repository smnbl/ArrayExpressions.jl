using ArrayAbstractions
using Metatheory
using GPUArrays
using Symbolics: unwrap, @variables

using Core.Compiler
using Core.Compiler: IRCode, CodeInfo

const CC = Core.Compiler

const AA = ArrayAbstractions

# implementation of AbstractGPUArray on CPU
include("jlarray.jl")
using .JLArrays

@info "testing GEMM optim"
const (M, N, K) = (10, 20, 30)

# temporary hack
function get_jl_array()
    println("replacement called!")
    C = JLArray(rand(Float32,(M, N)))
    return C
end

function Gemm(A, B, C)
    println("gemming...")
end

@array_opt function f_opt()
    A = JLArray(rand(Float32,(M, K)))
    B = JLArray(rand(Float32,(K, N)))
    C = JLArray(rand(Float32,(M, N)))

    return C + A * B
end

# opaque closuers are supported from Julia v1.8 onwards
function OC(ir::CC.IRCode, arg1::Any)
    src = ccall(:jl_new_code_info_uninit, Ref{CodeInfo}, ())
    src.slotflags = UInt8[]
    src.slotnames = Symbol[]
    nargs = length(ir.argtypes)
    CC.replace_code_newstyle!(src, ir, nargs)
    CC.widen_all_consts!(src)
    src.inferred = true

    m = ccall(:jl_make_opaque_closure_method, Ref{Method}, (Any, Any, Any, Any, Any),
              @__MODULE__, nothing, nargs, Core.LineNumberNode(0, nothing), src)

    rarg1 = Ref{Any}(arg1)
    ccall(:jl_new_opaque_closure, Any, (Any, Any, Any, Any, Any, Any, Csize_t),
          Tuple{ir.argtypes[2:end]...}, false, Union{}, Any, m, rarg1, 1)::Core.OpaqueClosure
end


ci = AA.codegen(:cache, f_opt, (), Core.svec())

f_opt()

println(Base.code_typed(f_opt))

# TODO: put source in generated function (see tests for examples on how to do (use OpaqueClosure?)


