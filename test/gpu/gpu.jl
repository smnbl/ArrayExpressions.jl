using ArrayAbstractions
using Core.Compiler

using CUDA
using GemmKernels
using BenchmarkTools
using LinearAlgebra

include("../compile.jl")

const (M, N, K) = (256, 256, 256)

# for a fair comparison
function (Base.:*)(A::CuMatrix, B::CuMatrix)
    C = CuMatrix{Float32}(undef, size(A))
    # C = 1.0 * A*B + 0.0 * C
    Gemm!(A, B, C, 1.0, 0.0)
    return C
end

@inline function Gemm(A::CuMatrix, B::CuMatrix, C::CuMatrix, alpha=1.0, beta=1.0)
    # return LinearAlgebra.mul!(C, A, B, 1.0, 1.0)
    return GemmWithEpilogue(A, B, C, identity, alpha, beta)
end

@inline function Gemm!(A, B, C, alpha=1.0, beta=1.0)
    GemmWithEpilogue!(A, B, C, identity, alpha, beta)
end

# TODO:
# add prologue?
@inline function GemmWithEpilogue(A, B, C, epi, alpha=1.0, beta=1.0)
    D = similar(C)
    GemmInterface(A, B, C, D, epi, alpha, beta)
    return D
end

@inline function GemmWithEpilogue!(A, B, C, epi, alpha=1.0, beta=1.0)
    GemmInterface(A, B, C, C, epi, alpha, beta)
end

@inline function GemmInterface(A, B, C, D, epi, alpha=1.0, beta=1.0)
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
                       # TODO: prologues, bias stuff
                       #transform_shared_to_regs_a = ...
                       transform_regs_to_shared_d = GemmKernels.Transform.Elementwise(epi),
                       kernel = GemmKernels.BLAS.kernel(a_layout, b_layout)
                      )
end

@array_opt function gemm_fusion(A, B, C)
    for i in 1:1000
        copyto!(C, A * B + C)
    end
    return C
end

relu(x) = max(0.0, x)

@array_opt function gemm_fusion_scalar_add(A, B, C)
    for i in 1:1000
        T = A * B
        T += C
# TODO: investigate what happens here?
        copyto!(C, relu.(T))
    end
    return C
end

@array_opt function gemm_multi(A, B, C)
    X = A * B + C
    Y = A * B + C
    return X, Y
end

A = CuArray(rand(Float16, (M, K)))
B = CuArray(rand(Float16, (K, N)))
C = CuArray(rand(Float32, (M, N)))

cache = ArrayAbstractions.CodeCache()

argtype = Tuple{Core.typeof.([A, B, C])...}

# TODO: add check if indeed replaced


# normal gemm

# TODO: this trick using generated functions seems fragile!!!
# interpolation happens not as you'd imagine :(
@eval generated_fusion(A, B, C) = $(compile(gemm_fusion, argtype, [:A, :B, :C]))

# TODO: fix this, why not same values???
CUDA.@sync println(isapprox(Array(generated_fusion(A, B, C)), Array(gemm_fusion(A, B, C)), rtol=1.0))

println("optimized, benching...")

println("gemm: before:")
@time CUDA.@sync gemm_fusion(A, B, C)
println("gemm: after:")
@time CUDA.@sync generated_fusion(A, B, C)

#=
# w scalar_add
# @eval generated_fusion_scalar_add(A, B, C) = $(compile(gemm_fusion_scalar_add, argtype, [:A, :B, :C]))
# CUDA.@sync println(isapprox(Array(generated_fusion_scalar_add(A, B, C)), Array(gemm_fusion_scalar_add(A, B, C)), rtol=1.0))

println("benchmarking...")

println("epi: before:")
@time CUDA.@sync gemm_fusion_scalar_add(A, B, C)

println("epi: after")
@time CUDA.@sync generated_fusion_scalar_add(A, B, C)

# generated multi
# @eval generated_multi(A, B, C) = $(compile(gemm_multi, argtype, [:A, :B, :C]))
=#
