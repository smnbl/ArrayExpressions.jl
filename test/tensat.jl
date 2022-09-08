using ArrayAbstractions
using BenchmarkTools
using CUDA

include("compile.jl")
include("gpu/gpu_rules.jl")

# in these tests we try to apply some rewrite rules used in the TENSAT (equality saturation for tensor graph superoptimization) paper

# define some of the operators
split(A, j) = view(A, :, 1:j), view(A, :, j+1:size(A,2)) # split at column j
matmul(A, B) = A * B
concat(A, B) = hcat(A, B)

toople(A, B) = tuple(A, B)

function split_matmul_expr(x1, x2, w1, w2)
    if hash(x1) != hash(x2)
        println("nothing")
        return nothing
    end
    return ArrayExpr(:call, [split, ArrayExpr(:call, [matmul, x1, ArrayExpr(:call, [concat, w1, w2])]), ArrayExpr(:call, [size, w1, 2])])
end
    

# the optimization rule:
tensat_rules = @array_theory w1 x x1 x2 w2 begin
    # TODO: don't have to add these trivial rules somehow
    matmul(x1, w1) == matmul(x1, w1)
    toople(matmul(x1, w1), matmul(x2, w2)) => split_matmul_expr(x1, x2, w1, w2)
end

normal_rules = @array_theory A B begin
    A * B == A * B
end

operations = Dict{Any, Any}()
operations[matmul] = 10000
operations[tuple] = 0
operations[toople] = 0
operations[:tuple] = 0

# closures
operations[:->] = 0
operations[:block] = 0

using Metatheory
using Metatheory: AbstractAnalysis, operation, arguments, ENodeTerm, ENodeLiteral, EGraph, hasdata

function cost_function(n::ENodeTerm, g::EGraph, an::Type{<:AbstractAnalysis})
    op = operation(n)
    if op isa Input
        if op.type isa Core.Const
            op = op.type.val
        else
            op = op.val
        end
    end


    cost = get(operations, op, 1000)

    for id in arguments(n)
        eclass = g[id]
        # if the child e-class has not yet been analyzed, return +Inf
        !hasdata(eclass, an) && (cost += Inf; break)
        cost += last(getdata(eclass, an))
    end

    # interesting illustration:
    # TOOD: add debug levels?

    return cost
end

cost_function(n::ENodeLiteral, g::EGraph, an::Type{<:AbstractAnalysis}) = 100
# tests

@array_opt function test(w1, w2, x)
    A = matmul(x, w1)
    B = matmul(x, w2)
    nothing
end

@array_opt function test_opt(w1, w2, x)
    A, B = split(matmul(x, concat(w1, w1)), size(w1, 2))
    nothing
end

msize = 512

x(msize=msize) = CuArray(randn(Float32, msize, msize))
w1(msize=msize) = CuArray(randn(Float32, msize, msize))
w2(msize=msize) = CuArray(randn(Float32, msize, msize))

X = x()
W1 = w1()
W2 = w2()

# with optimizations
args = [X, W1, W2]
argtype = Tuple{typeof.(args)...}

rules = AA.canonicalize_broadcasting ∪ tensat_rules

test(x(), w1(), w2())

println("compiling...")

@eval test_normal(x, w1, w2) = $(compile_deferred(test, (X, W1, W2), cost_function, extra_rules=normal_rules))
@eval test_opt(x, w1, w2) = $(compile_deferred(test, (X, W1, W2), cost_function, extra_rules=rules))

println("waiting...")

test_opt(x(), w1(), w2())

for i in (32, 128, 1024, 4096)
    println("$i")
    # without optimizations
    b1 = @benchmark CUDA.@sync test($(x(i)), $(w1(i)), $(w2(i)))
    println(b1)

    # with optimizations
    b2 = @benchmark CUDA.@sync test_opt($(x(i)), $(w1(i)), $(w2(i)))
    println(b2)
end
