using ArrayAbstractions
using Core.Compiler

using JLD2

using CUDA
using GemmKernels
using BenchmarkTools
using LinearAlgebra
using ArrayAbstractions: App, Lambda

const AA = ArrayAbstractions

include("../compile.jl")
include("./gpu_rules.jl")

const (M, N, K) = (512, 512, 512)

cache = ArrayAbstractions.CodeCache()

T = Float16

function a()
    CuArray(rand(T, (M, K)))
end

function b()
    CuArray(rand(T, (K, N)))
end

function c()
    CuArray(rand(T, (M, N)))
end

function d()
    CuArray(rand(T, (M, N)))
end

# simple gemm design example
function bench(before, a, b, c, d)
    A = a()
    B = b()
    C = c()
    D1 = d()
    D2 = d()

    @CUDA.sync before(D1, A, B, C)

    # race condition loading the CUDA libraries?
    println("$(Array(D1))")

    eltype = CuArray
    args = [A, B, C]
    argtype = Tuple{typeof.(args)...}

    # TODO: push array opt?
    @eval after(D2, A, B, C) = $(compile_deferred(before, (D2, A, B, C), cost_function, eltype=eltype, extra_rules=gemm_properties ∪ AA.canonicalize_broadcasting))

    Base.invokelatest(after, D2, A, B, C)

    # warmup & check
    # @assert isapprox(Array(D1), Array(D2), rtol=1.0, nans=true)

    println("benchmarking $before")
    println("epi: before:")
    before_bench = @benchmark CUDA.@sync Base.invokelatest($before, $(d()), $(a()), $(b()), $(c()))

    println("epi: after")
    after_bench = @benchmark CUDA.@sync Base.invokelatest($after, $(d()), $(a()), $(b()), $(c()))

    return before_bench, after_bench
end

#################################################################################################################################
@array_opt function gemm_design_example(D, A, B, C)
    q = Float32(0.1)
    T = A * B
    T += C
    copyto!(D, T .+ q)
    nothing
end

@array_opt function gemm_design_example_man_opt(A, B, C)
    copyto!(C, GemmWithEpilogue(A, B, C, el -> el + Float32(0.1)))
    nothing
end

@array_opt function gemm_design_example_man_gemm(A, B, C)
    T = Gemm(A, B, C)
    T .+ Float32(0.1)
    copyto!(C, T)
    nothing
end

@array_opt function gemm_design_example_man_transform(A, B, C)
    copyto!(C, GemmWithEpilogue(A, B, C, el -> el + Float32(0.1)))
    nothing
end

@array_opt function gemm_design_example_man_all(A, B, C)
    GemmWithEpilogue!(A, B, C, el -> el + Float32(0.1))
    nothing
end

@array_opt function gemm_design_complex(D, A, B, C)
    q = Float32(0.1)
    T = A * B
    T += C
    copyto!(D, T .^ 2 .^ 2 .^ 2 .^ 2)
    nothing
end

###################################################################################################################################
A = a()
B = b()
C = c()
C2 = c()
C1 = c()
# broadcasted
function gemm_broadcasted()
    D1 = similar(C)
    D2 = similar(C)
    gemm!(D1, A, B, C2)
    @eval gemm_opt!(D, A, B, C2) = $(compile_deferred(gemm!, (D2, A, B, C2), cost_function, extra_rules=gemm_properties ∪ AA.canonicalize_broadcasting))

    Base.invokelatest(gemm_opt!, D2, A, B, C2)

    # warmup & check
    @assert isapprox(Array(D1), Array(D2), rtol=sqrt(eps(Float16)), nans=true)

    println("benchmarking gemm_broadcast replacement")
    println("epi: before:")
    CUDA.@sync gemm!(D1, A, B, C2) # warmup
    before = @elapsed CUDA.@sync gemm!(D1, A, B, C2)

    println("epi: after")
    CUDA.@sync Base.invokelatest(gemm_opt!, D2, A, B, C2) # warmup
    after = @elapsed CUDA.@sync Base.invokelatest(gemm_opt!, (D2, A, B, C2))

    return before, after
end

function test_broadcast()
    A = a()
    B = b()
    C_flat = similar(A, size(A, 1))

    D1 = broadcast(+, A * B, C_flat)

    D2 = similar(D1)
    GemmInterface(A, B, C_flat, D2, identity)

    println(D1 - D2)
    @assert isapprox(Array(D1), Array(D2), rtol=sqrt(sqrt(eps(Float16))), nans=true)
end

function test_gemm()
    A = a()
    B = b()
    C = c()

    D1 = A * B + C

    D2 = similar(D1)
    GemmInterface(A, B, C, D2, identity)

    println(D1 - D2)
    @assert isapprox(Array(D1), Array(D2), rtol=sqrt(sqrt(eps(Float16))), nans=true)
end


################################################################################################################################
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

function gemm_gemm_replacement()
    D1 = similar(C)
    D2 = similar(C)
    gemm!(D1, A, B, C)
    @eval gemm_opt!(D, A, B, C) = $(compile_deferred(gemm!, (D2, A, B, C), cost_function, eltype=eltype, extra_rules=gemm_properties ∪ AA.canonicalize_broadcasting))

    gemm_opt!(D2, A, B, C)

    # warmup & check
    @assert isapprox(Array(D1), Array(D2), rtol=1.0, nans=true)

    println("benchmarking gemm replacement")
    println("epi: before:")
    CUDA.@sync gemm!(D1, A, B, C) # warmup
    before = @benchmark CUDA.@sync gemm!(D1, A, B, C)

    println("epi: after")
    CUDA.@sync gemm_opt!(D2, A, B, C) # warmup
    after = @benchmark CUDA.@sync gemm_opt!(D2, A, B, C)

    return befor, after
end


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

function gemm_scalar_fusion()
    gemm_fusion(A, B, C)

    @eval gemm_fusion_opt(A, B, C) = $(compile_deferred(gemm_fusion, (A, B, C), cost_function, eltype=eltype, extra_rules=gemm_properties ∪ AA.canonicalize_broadcasting))

    CUDA.@sync gemm_fusion(A, B, C) # warmup
    CUDA.@sync gemm_fusion_opt(A, B, C) # warmup

    println("benchmarking fusion_scalar_add")
    println("epi: before")
    before = @benchmark CUDA.@sync gemm_fusion(A, B, C)

    println("epi: after")
    after =  @benchmark CUDA.@sync gemm_fusion_opt(A, B, C)

    return before, after
end

#################################################################################################################################


#benchs = load("benchmakrs.jld2")

# before = @benchmark CUDA.@sync gemm_design_example($(a()), $(b()), $(c()))
# gemm =  @benchmark CUDA.@sync gemm_design_example_man_opt($(a()), $(b()), $(c()))
# transform =  @benchmark CUDA.@sync gemm_design_example_man_transform($(a()), $(b()), $(c()))
# all =  @benchmark CUDA.@sync gemm_design_example_man_all($(a()), $(b()), $(c()))

# bench(gemm_design_example, a, b, c, d)
# bench(gemm_design_example, a, b, c, d)

# benchs["design_simple"] = before, gemm, transform, all
# benchs["design_complex"] = bench(gemm_design_complex, a, b, c, d)
# benchs["gemm_fusion"] = bench(gemm_fusion, a, b, c, d)

#=
function gemm_multi(A, B, C)
    T = A * B
    T += C
    T = element_kernel(T)

    X = adjoint(B) * adjoint(A) + adjoint(C)

    return T, X
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
