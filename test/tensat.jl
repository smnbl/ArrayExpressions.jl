using ArrayAbstractions
using BenchmarkTools
using CUDA

include("compile.jl")

# in these tests we try to apply some rewrite rules used in the TENSAT (equality saturation for tensor graph superoptimization) paper

# define some of the operators
split(A) = A[:, 1:size(l, 2)÷2]
matmul(A, B) = A * B
concat(A, B) = hcat(A, B)

toople(A, B) = tuple(A, B)

# the optimization rule:
rules = @array_theory w1 x x1 x2 w2 begin
    # TODO: don't have to add these trivial rules somehow
    matmul(x1, w1) == matmul(x1, w1)

    # TODO: dynamic checks?

    # TODO: fix tuple rewrites; works with toople though :P
    toople(matmul(x1, w1), matmul(x2, w2)) == split(matmul(x1, concat(w1, w2)), x2)
end

# tests

function test(w1, w2, x)
    A = matmul(x, w1)
    B = matmul(x, w2)

    return toople(A, B)
end

size = 10000

x = CuArray(randn(size, size))
w1 = CuArray(randn(size, size))
w2 = CuArray(randn(size, size))


# with optimizations
args = [x, w1, w2]
argtype = Tuple{typeof.(args)...}

@eval test_normal(x, w1, w2) = $(compile_with_gpucompiler(test, argtype, [:x, :w1, :w2]))
@eval test_opt(x, w1, w2) = $(compile_with_gpucompiler(test, argtype, [:x, :w1, :w2], extra_rules=rules))

test_normal(x, w1, w2)
test_opt(x, w1, w2)

# without optimizations
@time CUDA.@sync test_normal(x, w1, w2)

# with optimizations
@time CUDA.@sync test_opt(x, w1, w2)
       
# without optimizations
@time CUDA.@sync test_normal(x, w1, w2)

# with optimizations
@time CUDA.@sync test_opt(x, w1, w2)
