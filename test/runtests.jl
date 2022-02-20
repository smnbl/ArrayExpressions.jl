using ArrayAbstractions
using Metatheory
using CUDA

const AA = ArrayAbstractions

#ArrayAbstractions.inject_typeinf()

@info "testing GEMM optim"
(M, N, K) = (10, 20, 30)

@array_opt function f_opt()
    A = CuArray(rand(Float32, (M, K)))
    B = CuArray(rand(Float32, (K, N)))
    C = CuArray(rand(Float32, (M, N)))
    T = A * B
    D = T + C
    return D
end

using InteractiveUtils

println(InteractiveUtils.@code_typed f_opt())

AA.codegen(:typed, f_opt, (), Core.svec())
