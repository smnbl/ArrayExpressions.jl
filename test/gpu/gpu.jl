using ArrayAbstractions
using Core.Compiler

using CUDA
using GemmKernels
using BenchmarkTools
using LinearAlgebra
using ArrayAbstractions: App, Lambda

include("../compile.jl")
include("./gpu_rules.jl")

const (M, N, K) = ntuple(i -> 512, 3)


# for a fair comparison
#=
@inline function (Base.:*)(A::CuMatrix, B::CuMatrix)
    C = CuArray{Float32}(undef, (size(A, 1), size(B, 2)))
    # C = 1.0 * A*B + 0.0 * C
    Gemm!(A, B, C, alpha=1.0, beta=0.0)
    return C
end
=#

@inline function Gemm(A, B, C; alpha=1.0, beta=1.0)
    # return LinearAlgebra.mul!(C, A, B, 1.0, 1.0)
    return GemmWithEpilogue(A, B, C, identity, alpha=alpha, beta=beta)
end

@inline function Gemm!(A, B, C; alpha=1.0, beta=1.0)
    GemmWithEpilogue!(A, B, C, identity, alpha=alpha, beta=beta)
end

# TODO:
# add prologue?
@inline function GemmWithEpilogue(A, B, C, transform; alpha=1.0, beta=1.0)
    D = similar(C)
    GemmInterface(A, B, C, D, transform, alpha=alpha, beta=beta)
    return D
end

@inline function GemmWithAdd(A, B, C, d; alpha=1.0, beta=1.0)
    transform = epi(+, d)
    D = similar(C)
    GemmInterface(A, B, C, D, transform, alpha=alpha, beta=beta)
    return D
end

@inline function GemmWithEpilogue!(A, B, C, transform; alpha=1.0, beta=1.0)
    GemmInterface(A, B, C, C, transform, alpha=alpha, beta=beta)
end

@inline function GemmBias(A, B, C, transform, bias)
    D = similar(C)
    GemmInterface(A, B, C, D, transform, alpha=alpha, beta=beta, bias=bias)
    return D
end

@inline function GemmInterface(A, B, C, D, transform; alpha=1.0, beta=1.0, bias=nothing)
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

            is_a_col_major = true,
            is_b_col_major = true
                                )
    GemmKernels.matmul(A, B, C, D, conf;
                       # TODO: prologues, bias stuff

                       transform_shared_to_regs_a = Transform.Elementwise(x -> x * alpha),
                       transform_shared_to_regs_c = Transform.Elementwise(x -> x * beta),

                       transform_regs_to_shared_d = Transform.Elementwise(transform),
                       epilogue = if (bias != nothing) GemmKernels.Epilogue.Bias(pointer(bias)) else GemmKernels.Epilogue.Default() end,
                       kernel = GemmKernels.BLAS.kernel(a_layout, b_layout)
                      )
end

eltype = CuArray

# intergration tests
gemmcompile(func, argtype, args) = compile_expression(func, argtype, args; eltype=eltype, extra_rules=gemm_properties ∪ AA.canonicalize_broadcasting, intrinsics=gpu_intrinsics)

function subcall(A, B)
    return subsubcall(A, B)
end

function subsubcall(A, B)
    return A + B
end

function gemm(A, B, C)
    return subcall(A * B, C)
end

relu(x) = max(0.0, x)

function gemm_fusion_scalar_add(A, B, C)
    T = A * B
    T += C
    # NOTE: how make this more flexible that other floating types are supported?
    T = T .+ Float32(0.2)
    T = relu.(T)
    return T
end

function gemm_multi(A, B, C)
    X = A * B + C
    Y = A * B + C
    return X, Y
end

A = CuArray(rand(Float16, (M, K)))
B = CuArray(rand(Float16, (K, N)))
C = CuArray(rand(Float32, (M, N)))

cache = ArrayAbstractions.CodeCache()

argtype = Tuple{Core.typeof.([A, B, C])...}

# demo function
@eval gemm_opt(A, B, C) = $(gemmcompile(gemm, argtype, [:A, :B, :C]))
# warmup & check
@test isapprox(Array(gemm(A, B, C)), Array(gemm_opt(A, B, C)), rtol=1.0, nans=true)

println("optimized, benching...")
println("epi: before:")
@time CUDA.@sync begin
    for _ in 1:10000
        copyto!(C, gemm(A, B, C))
    end
end

println("epi: after")
@time CUDA.@sync begin
    for _ in 1:10000
        copyto!(C, gemm_opt(A, B, C))
    end
end

# w scalar_add
@eval gemm_fusion_scalar_add_opt(A, B, C) = $(gemmcompile(gemm_fusion_scalar_add, argtype, [:A, :B, :C]))
@test isapprox(Array(gemm_fusion_scalar_add_opt(A, B, C)), Array(gemm_fusion_scalar_add(A, B, C)), rtol=1.0, nans=true)

iterations = 10000

println("benchmarking...")
println("epi: before:")
@time CUDA.@sync begin
    for _ in 1:iterations
        copyto!(C, gemm_fusion_scalar_add(A, B, C))
    end
end

println("epi: after")
@time CUDA.@sync begin
    for _ in 1:iterations
        copyto!(C, gemm_fusion_scalar_add_opt(A, B, C))
    end
end

# generated multi
# @eval generated_multi(A, B, C) = $(compile(gemm_multi, argtype, [:A, :B, :C]))
