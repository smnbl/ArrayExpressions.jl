using ArrayAbstractions
using Metatheory
using GPUArrays

const AA = ArrayAbstractions

# implementation of AbstractGPUArray on CPU
include("jlarray.jl")
using .JLArrays

#ArrayAbstractions.inject_typeinf()

@info "testing GEMM optim"
const (M, N, K) = (10, 20, 30)

@array_opt function f_opt()
    A = JLArray(rand(Float32,(M, K)))
    B = JLArray(rand(Float32,(K, N)))
    C = JLArray(rand(Float32,(M, N)))
    repeat = x

    T = A * B
    if (rand() > 0.5)
        C = x*C
    end

    return T + C
end

AA.codegen(:inject, f_opt, (), Core.svec())
