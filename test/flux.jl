using Flux
using NNlib
using CUDA

include("compile.jl")
include("gpu/gpu_rules.jl")

const bias = cu(randn(Float32, 7))

xs = rand(Float32, 1000, 1000, 3, 50)

const layer = Conv((5,5), 3 => 7, relu; bias = bias)

const chain = Chain(
    Conv((5, 5), 3 =>6, relu),
    Dropout(Float32(0.4)),
    MaxPool((2, 2)),
    Conv((5, 5), 6=>16, relu),
    MaxPool((2, 2)),
          )

# investigate what goes wrong, and why needed to wrap inside other function
function f(xs)
    return chain(xs)
end

println("testing non-optimised")
f(xs)

function replace(f, a, x, w, cdims, b)
    println("replacing...")

    return ArrayExpr(:call, [GlobalRef(NNlib, :conv_bias_act), x, w, cdims, b, a])
end

function conv_bias_act(x, w, cdims, b, a)
    println("bias")
    bz = CUDA.zeros(size(b))
    NNlib.conv_bias_act(x, w, cdims, bz, a)
end

flux_conv_fusing_rules = @array_theory begin
    # TODO fix this with current graphs (Flux.conv seeems different)
    Main.broadcast(~a, Main.broadcast(~f, Flux.conv(~x, ~w, ~cdims), ~b)) => replace(~f, ~a, ~x, ~w, ~cdims, ~b) where istype(~f, typeof(Flux.:+))k
end

const rules = flux_conv_fusing_rules ∪ AA.canonicalize_broadcasting ∪ gemm_properties

args = [xs]
argtype = Tuple{typeof.(args)...}

# @eval f_opt(xs) = compile_with_gpucompiler(f, argtype, [:xs], eltype=CuArray, extra_rules=rules)
ci = compile_expression(f, argtype, [:xs], eltype=CuArray, extra_rules=rules)
@generated f_opt(xs) = ci

println("testing layer_opt")
println(isapprox(Array(f(xs)), Array(f_opt(xs)), rtol=1.0, nans=true))

println("flux before:")
@time CUDA.@sync begin
    f(xs)
end

println("flux after:")
@time CUDA.@sync begin
    f_opt(xs)
end

println(":)")

