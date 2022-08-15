using ArrayAbstractions
using Core.Compiler

using CUDA
using GemmKernels
using BenchmarkTools
using LinearAlgebra
using ArrayAbstractions: App, Lambda

const AA = ArrayAbstractions

include("../compile.jl")
include("./gpu_rules.jl")

const (M, N, K) = ntuple(i -> 512, 3)


eltype = CuArray

# intergration tests

macro gemmcompile(target, func, args, argtype)
    println(args)
    quote
        $(esc(target))($(eval(args)...)) = $(compile_with_gpucompiler(func, argtype, args; eltype=eltype, extra_rules=gemm_properties ∪ AA.canonicalize_broadcasting, intrinsics=gpu_intrinsics))
    end
end

function subcall(A, B, C)
    return subsubcall(A, B) + C
end

function subsubcall(A, B)
    @noinline return CUDA.:*(A, B)
end

function gemm(A, B, C)
    subcall(A, B, C)
end

relu(x) = max(Float32(0.0), x)

function scalar_kernel(T)
    T = T .+ Float32(0.2)
    T = relu.(T)
    T = T .* 3.0

    return T
end

function gemm_fusion_scalar_add(A, B, C)
    T = A * B
    T += C
    T = scalar_kernel(T)

    return T
end

function gemm_multi(A, B, C)
    T = A * B
    T += C
    T = element_kernel(T)

    X = adjoint(B) * adjoint(A) + adjoint(C)

    return T, X
end

A = CuArray(rand(Float16, (M, K)))
B = CuArray(rand(Float16, (K, N)))
C = CuArray(rand(Float16, (M, N)))

cache = ArrayAbstractions.CodeCache()

args = [A, B, C]
argtype = Tuple{typeof.(args)...}

# demo function
ci = compile_expression(gemm, argtype, [:A, :B, :C]; eltype=eltype, extra_rules=gemm_properties ∪ AA.canonicalize_broadcasting)
@generated gemm_opt(A, B, C) = ci

@eval gemm_opt(A, B, C) = $(compile_with_gpucompiler(gemm, argtype, [:A, :B, :C]; eltype=eltype, extra_rules=gemm_properties ∪ AA.canonicalize_broadcasting))

# warmup & check
#@test isapprox(Array(gemm(A, B, C)), Array(gemm_opt(A, B, C)), rtol=1.0, nans=true)

#iterations = 10000

#=
println("benching gemm")
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
=#

# w scalar_add
#=
@gemmcompile gemm_fusion_scalar_add_opt gemm_fusion_scalar_add [:A, :B, :C] argtype
@test isapprox(Array(gemm_fusion_scalar_add_opt(A, B, C)), Array(gemm_fusion_scalar_add(A, B, C)), rtol=1.0, nans=true)

println("benchmarking fusion_scalar_add")
println("epi: before:")

CUDA.@sync gemm_fusion_scalar_add(A, B, C)

@time CUDA.@sync begin
    for _ in 1:iterations
        copyto!(C, gemm_fusion_scalar_add(A, B, C))
    end
end

println("epi: after")
CUDA.@sync gemm_fusion_scalar_add_opt(A, B, C)

@time CUDA.@sync begin
    for _ in 1:iterations
        copyto!(C, gemm_fusion_scalar_add_opt(A, B, C))
    end
end

@gemmcompile gemm_multi_opt gemm_multi [:A, :B, :C] argtype

a1, a2 = gemm_multi_opt(A, B, C)
b1, b2 = gemm_multi(A, B, C)
@test isapprox(Array(a1), Array(b1), rtol=1.0, nans=true)
@test isapprox(Array(b2), Array(b2), rtol=1.0, nans=true)

println("benchmarking multi")
println("multi: before:")


@time CUDA.@sync begin
    for _ in 1:iterations
        gemm_multi(A, B, C)
    end
end

println("multi: after")
@time CUDA.@sync begin
    for _ in 1:iterations
        gemm_multi_opt(A, B, C)
    end
end
=#
