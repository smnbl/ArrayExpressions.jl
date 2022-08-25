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

const (M, N, K) = (512, 512, 512)

eltype = CuArray

# intergration tests

macro gemmcompile(target, func, args, argtype)
println(args)
quote
    $(esc(target))($(eval(args)...)) = $(compile_with_gpucompiler(func, argtype, args; eltype=eltype, extra_rules=gemm_properties ∪ AA.canonicalize_broadcasting, intrinsics=gpu_intrinsics))
end
end

function subcall(A, B, C)
    return broadcast(+, subsubcall(A, B), C)
end

function subsubcall(A, B)
    A * B
end

@array_opt function gemm!(D, A, B, C)
    copyto!(D, subcall(A, B, C))
    # problem at the boundary between the deferred call?
    return nothing
end

function gemm_multi(A, B, C)
    T = A * B
    T += C
    T = element_kernel(T)

    X = adjoint(B) * adjoint(A) + adjoint(C)

    return T, X
end

A = CuArray(rand(Float32, (M, K)))
B = CuArray(rand(Float32, (K, N)))
C = CuArray(rand(Float32, (M, N)))
C2 = CuArray(rand(Float32, N))

cache = ArrayAbstractions.CodeCache()

args = [A, B, C]
argtype = Tuple{typeof.(args)...}

# demo function
# ci = compile_expression(gemm, argtype, [:A, :B, :C], cost_function; eltype=eltype, extra_rules=gemm_properties ∪ AA.canonicalize_broadcasting)
# @generated gemm_opt(A, B, C) = ci

#@eval gemm_opt(A, B, C) = $(compile_with_gpucompiler(gemm, argtype, [:A, :B, :C], cost_function; eltype=eltype, extra_rules=gemm_properties ∪ AA.canonicalize_broadcasting))
#gemm_opt(A, B, C)

D1 = similar(C)
D2 = similar(C)
gemm!(D1, A, B, C2)
@eval gemm_opt!(D, A, B, C2) = $(compile_deferred(gemm!, (D2, A, B, C2), cost_function, eltype=eltype, extra_rules=gemm_properties ∪ AA.canonicalize_broadcasting))

gemm_opt!(D2, A, B, C2)

# warmup & check
@test isapprox(Array(D1), Array(D2), rtol=1.0, nans=true)

println("benchmarking gemm_broadcast replacement")
println("epi: before:")
CUDA.@sync gemm!(D1, A, B, C2) # warmup
bench = @benchmark CUDA.@sync gemm!(D1, A, B, C2)
println(bench)

println("epi: after")
CUDA.@sync gemm_opt!(D2, A, B, C2) # warmup
bench = @benchmark CUDA.@sync gemm_opt!(D2, A, B, C2)
println(bench)

################################################################################################################################

D1 = similar(C)
D2 = similar(C)
gemm!(D1, A, B, C)
@eval gemm_opt!(D, A, B, C) = $(compile_deferred(gemm!, (D2, A, B, C), cost_function, eltype=eltype, extra_rules=gemm_properties ∪ AA.canonicalize_broadcasting))

gemm_opt!(D2, A, B, C)

# warmup & check
@test isapprox(Array(D1), Array(D2), rtol=1.0, nans=true)

println("benchmarking gemm replacement")
println("epi: before:")
CUDA.@sync gemm!(D1, A, B, C) # warmup
bench = @benchmark CUDA.@sync gemm!(D1, A, B, C)
println(bench)

println("epi: after")
CUDA.@sync gemm_opt!(D2, A, B, C) # warmup
bench = @benchmark CUDA.@sync gemm_opt!(D2, A, B, C)
println(bench)

#################################################################################################################################

function scalar_kernel(T)
    relu(x) = max(Float32(0.0), x)

    T = T .+ Float32(0.2)
    T = relu.(T)
    T = T .* Float32(3.0)
    T = relu.(T)
    T = T .* Float32(3.0)
    T = relu.(T)
    T = T .* Float32(3.0)
    return T
end

@array_opt function gemm_fusion(A, B, C)
    T = A * B
    T += C
    T = scalar_kernel(T)
    
    copyto!(C, T)
    return nothing
end

gemm_fusion(A, B, C)

@eval gemm_fusion_opt(A, B, C) = $(compile_deferred(gemm_fusion, (A, B, C), cost_function, eltype=eltype, extra_rules=gemm_properties ∪ AA.canonicalize_broadcasting))

CUDA.@sync gemm_fusion(A, B, C) # warmup
CUDA.@sync gemm_fusion_opt(A, B, C) # warmup

println("benchmarking fusion_scalar_add")
println("epi: before")
bench = @benchmark CUDA.@sync gemm_fusion(A, B, C)
println("$bench")

println("epi: after")
bench =  @benchmark CUDA.@sync gemm_fusion_opt(A, B, C)
println("$bench")

#################################################################################################################################

#=
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
