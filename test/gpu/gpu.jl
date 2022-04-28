using ArrayAbstractions
using Core.Compiler

using CUDA
using GemmKernels
using BenchmarkTools

## test native target
include("GPUCompiler.jl/test/definitions/native.jl")

const (M, N, K) = (512, 512, 512)


function Gemm(A::CuArray, B::CuArray, C::CuArray)
    return GemmWithEpilogue(A, B, C, identity)
end

# TODO:
# add prologue?
function GemmWithEpilogue(A, B, C, epi)
    println(epi)
    m = size(A, 1)
    k = size(A, 2)
    n = size(B, 2)

    if m != size(C, 1) || n != size(C, 2) || k != size(B, 1)
        throw(DimensionMismatch("Dimensions do not match"))
    end

    a_layout = GemmKernels.BLAS.global_layout(typeof(A), Val(false))
    b_layout = GemmKernels.BLAS.global_layout(typeof(B), Val(false))

    D = similar(C)

    conf = GemmKernels.get_config(
            gemm_shape = (M = m, N = n, K = k),
            operator = Operator.WMMAOp{16, 16, 16, eltype(C)},

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
                       transform_shared_to_regs_c = GemmKernels.Transform.Elementwise(epi),
                       kernel = GemmKernels.BLAS.kernel(a_layout, b_layout)
                      )
    return D
end

@array_opt function gemm_fusion(A, B, C)
    return A * B + C
end

@array_opt function gemm_fusion_scalar_add(A, B, C)
    return A * B + C .* 2.0
end

A = CuArray(rand(Float16, (M, K)))
B = CuArray(rand(Float16, (K, N)))
C = CuArray(rand(Float16, (M, N)))

cache = ArrayAbstractions.CodeCache()

args = Core.typeof.([A, B, C])

ci  = AA.optimize(gemm_fusion, (args), Core.svec(); cache = cache)
# TODO: add check if indeed replaced

@generated generated_fusion(A, B, C) = ci
CUDA.@sync println(isapprox(Array(generated_fusion(A, B, C)), Array(gemm_fusion(A, B, C))))

println("before:")
CUDA.@time CUDA.@sync gemm_fusion(A, B, C)
println("after:")
CUDA.@time CUDA.@sync generated_fusion(A, B, C)

# SECOND test
A = CuArray(rand(Float16, (M, K)))
B = CuArray(rand(Float16, (K, N)))
C = CuArray(rand(Float16, (M, N)))

ci  = AA.optimize(gemm_fusion_scalar_add, (args), Core.svec(); cache = cache)
# TODO: add check if indeed replaced

@generated generated_fusion_scalar_add(A, B, C) = ci
CUDA.@sync println(isapprox(Array(generated_fusion_scalar_add(A, B, C)), Array(gemm_fusion_scalar_add(A, B, C))))
A = CuArray(rand(Float16, (M, K)))
B = CuArray(rand(Float16, (K, N)))
C = CuArray(rand(Float16, (M, N)))
println("before:")
CUDA.@time CUDA.@sync gemm_fusion_scalar_add(A, B, C)
A = CuArray(rand(Float16, (M, K)))
B = CuArray(rand(Float16, (K, N)))
C = CuArray(rand(Float16, (M, N)))
println("after")
CUDA.@time CUDA.@sync generated_fusion_scalar_add(A, B, C)


#=
GC.@preserve args begin
        kernel_args = cudaconvert.(args)
        kernel_tt = Tuple{Core.Typeof.(kernel_args)...}
        kernel = cufunction(test3, kernel_tt, extra_passes=[ArrayAbstractions.arroptim_pass])
        kernel(kernel_args...)
end
=#
