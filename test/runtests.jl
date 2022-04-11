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

function Gemm(A, B, C)
    println("gemming...")
    return A*B + C
end

A = JLArray(rand(Float32,(M, K)))
B = JLArray(rand(Float32,(K, N)))
C = JLArray(rand(Float32,(M, N)))

function gemm_replacement(A::JLArray, B::JLArray, C::JLArray)
    return C + A * B
end

function gemm_fusing_scalar_add(A::JLArray, B::JLArray, C::JLArray)
    return A * B + C .+ 2
end

function gemm_fusing_scalar_mul(A::JLArray, B::JLArray, C::JLArray)
    return (C + A * B) .* 2
end

function gemm_fusing_scalar_addmul(A::JLArray, B::JLArray, C::JLArray)
    return (C + A * B) .* 2 .+ 2
end

function strength_reduction(A::JLArray, B::JLArray, C::JLArray)
    return (C + C)
end

function multi_strength_reduction(A::JLArray, B::JLArray, C::JLArray)
    return (C + C + C + C + C + C)
end

function assignments(A, B, C)
    X = A
    Y = B
    Z = C

    return X * Y + C
end

function if_statement(A::JLArray, B::JLArray, C::JLArray)
    # TODO
    if x == 10
        C = C + C
    end

    return A * B + C
end

function test_ssa_codegen(A, B, C)
    if x == 10
        C = C + C
    end

    D = A * B + C # GEMM

    B = 2 * D

    return B
end

#=
expr  = AA.optimize(:cache, gemm_replacement, (JLArray{Float32, 2}, JLArray{Float32, 2}, JLArray{Float32, 2}), Core.svec())

eval(:(lambda = $expr;
        println(typeof(lambda));
        println(lambda);
        println(lambda(A, B, C))))
=#

# GEMM:
expr = AA.optimize(:cache, gemm_replacement, (JLArray{Float32, 2}, JLArray{Float32, 2}, JLArray{Float32, 2}), Core.svec())

println(expr)

#expr  = AA.optimize(:cache, gemm_fusing_scalar_add, (JLArray{Float32, 2}, JLArray{Float32, 2}, JLArray{Float32, 2}), Core.svec())
#expr  = AA.optimize(:cache, gemm_fusing_scalar_mul, (JLArray{Float32, 2}, JLArray{Float32, 2}, JLArray{Float32, 2}), Core.svec())
#expr  = AA.optimize(:cache, gemm_fusing_scalar_addmul, (JLArray{Float32, 2}, JLArray{Float32, 2}, JLArray{Float32, 2}), Core.svec())

# Matrix arithmetic
#=
expr  = AA.optimize(:cache, strength_reduction, (JLArray{Float32, 2}, JLArray{Float32, 2}, JLArray{Float32, 2}), Core.svec())

# assignments
expr  = AA.optimize(:cache, assignments, (JLArray{Float32, 2}, JLArray{Float32, 2}, JLArray{Float32, 2}), Core.svec())

# if
expr  = AA.optimize(:cache, if_statement, (JLArray{Float32, 2}, JLArray{Float32, 2}, JLArray{Float32, 2}), Core.svec())

# SSA codegen
# expr  = AA.optimize(:cache, test_ssa_codegen, (JLArray{Float32, 2}, JLArray{Float32, 2}, JLArray{Float32, 2}), Core.svec())
=#

#=
eval(:(lambda = $expr;
        println(typeof(lambda));
        println(lambda);
        println(lambda(A, B, C))))
=#
