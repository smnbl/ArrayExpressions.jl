using ArrayAbstractions
using Core.Compiler

using CUDA
using GemmKernels
using BenchmarkTools
using LinearAlgebra

## test native target
include("GPUCompiler.jl/test/definitions/native.jl")

const (M, N, K) = (256, 256, 256)

@inline function Gemm(A::CuArray, B::CuArray, C::CuArray)
    # return LinearAlgebra.mul!(C, A, B, 1.0, 1.0)
    return GemmWithEpilogue(A, B, C, identity)
end

@inline function Gemm!(A, B, C)
    GemmWithEpilogue!(A, B, C, identity)
end

# TODO:
# add prologue?
@inline function GemmWithEpilogue(A, B, C, epi)
    D = similar(C)
    GemmInterface(A, B, C, D, epi)
    return D
end

@inline function GemmWithEpilogue!(A, B, C, epi)
    GemmInterface(A, B, C, C, epi)
end

@inline function GemmInterface(A, B, C, D, epi)
    m = size(A, 1)
    k = size(A, 2)
    n = size(B, 2)

    if m != size(C, 1) || n != size(C, 2) || k != size(B, 1)
        throw(DimensionMismatch("Dimensions do not match"))
    end

    a_layout = GemmKernels.BLAS.global_layout(typeof(A), Val(false))
    b_layout = GemmKernels.BLAS.global_layout(typeof(B), Val(false))

    conf = GemmKernels.get_config(
            gemm_shape = (M = m, N = n, K = k),
            # TODO: gemmkernels interface changes here in latest version: .., eltype(C)}
            operator = Operator.WMMAOp{16, 16, 16},

            global_a_layout = a_layout,
            global_b_layout = b_layout,
            global_c_layout = GemmKernels.BLAS.global_layout(typeof(C), Val(false)),
            global_d_layout = GemmKernels.BLAS.global_layout(typeof(C), Val(false)),

            shared_a_layout = GemmKernels.BLAS.shared_layout_ab(typeof(A), Val(false)),
            shared_b_layout = GemmKernels.BLAS.shared_layout_ab(typeof(B), Val(false)),
            shared_c_layout = GemmKernels.BLAS.shared_layout_cd(typeof(C), Val(false)),
            shared_d_layout = GemmKernels.BLAS.shared_layout_cd(typeof(C), Val(false)),

            is_a_col_major = false,
            is_b_col_major = false
                                )
    GemmKernels.matmul(A, B, C, D, conf;
                       #transform_shared_to_regs_a = ...
                       transform_regs_to_shared_d = GemmKernels.Transform.Elementwise(epi),
                       kernel = GemmKernels.BLAS.kernel(a_layout, b_layout)
                      )
end

@array_opt function gemm_fusion(A, B, C)
    for _ in 1:1000
        copyto!(C, A * B + C)
    end
    return C
end

relu(x) = max(0.0, x)

@array_opt function gemm_fusion_scalar_add(A, B, C)
    for i in 1:100
        T = A * B
        if (i > 100)
            println("test")
        end
        T += C
        T = T .+ i
        copyto!(C, relu.(T))
    end
    return C
end

A = CuArray(rand(Float16, (M, K)))
B = CuArray(rand(Float16, (K, N)))
C = CuArray(rand(Float32, (M, N)))

cache = ArrayAbstractions.CodeCache()

args = Core.typeof.([A, B, C])

ci  = AA.optimize(gemm_fusion, (args), Core.svec(); cache = cache)
# TODO: add check if indeed replaced
println(ci)

@generated generated_fusion(A, B, C) = ci
CUDA.@sync println(isapprox(Array(generated_fusion(A, B, C)), Array(gemm_fusion(A, B, C)), rtol=1.0))

println("gemm: before:")
@time CUDA.@sync gemm_fusion(A, B, C)

println("gemm: after:")
@time CUDA.@sync generated_fusion(A, B, C)


ci  = AA.optimize(gemm_fusion_scalar_add, (args), Core.svec(); cache = cache)
# TODO: add check if indeed replaced
println(ci)

@generated generated_fusion_scalar_add(A, B, C) = ci

println("benchmarking...")

CUDA.@sync println(isapprox(Array(generated_fusion_scalar_add(A, B, C)), Array(gemm_fusion_scalar_add(A, B, C)), rtol=1.0))

println("epi: before:")
@time CUDA.@sync gemm_fusion_scalar_add(A, B, C)

println("epi: after")
@time CUDA.@sync generated_fusion_scalar_add(A, B, C)


#=
GC.@preserve args begin
        kernel_args = cudaconvert.(args)
        kernel_tt = Tuple{Core.Typeof.(kernel_args)...}
        kernel = cufunction(test3, kernel_tt, extra_passes=[ArrayAbstractions.arroptim_pass])
        kernel(kernel_args...)
end
=#
