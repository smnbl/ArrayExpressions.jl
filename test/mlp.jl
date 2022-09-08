using Flux
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay
using Flux: onehotbatch, onecold, flatten
using Flux.Losses: logitcrossentropy
using Flux: AbstractRNG, rng_from_array, nfan
using NNlib
using CUDA
using BenchmarkTools
using JLD2

using ArrayAbstractions
const AA = ArrayAbstractions

include("compile.jl")
include("gpu/gpu_rules.jl")

function mlp(; imgsize=(28,28,1), nclasses=10) 
    return Chain(
                 flatten,
                 Dense(prod(imgsize), 32, relu),
                 Dense(32, nclasses))
end

xs_f(;img_size=32) = CuArray(rand(Float32, img_size, img_size, 1, 50))
xs = xs_f()

# Float16 version of Flux.glorot_uniform, necessary as otherwirse the weight matrices get initialized to Float32 (GemmKernels.jl doesn't support mixed precision gemm atm)
function glorot_uniform_16(rng::AbstractRNG, dims::Integer...; gain::Real=1)
  scale = Float16(gain) * Float16(sqrt(24.0f0 / sum(nfan(dims...))))
  (rand(rng, Float16, dims...) .- Float16(0.5f0)) .* scale
end
glorot_uniform_16(dims::Integer...; kw...) = glorot_uniform_16(rng_from_array(), dims...; kw...)
glorot_uniform_16(rng::AbstractRNG=rng_from_array(); init_kwargs...) = (dims...; kwargs...) -> glorot_uniform_16(rng, dims...; init_kwargs..., kwargs...)

function replace_conv_bias_act(f, a, x, w, cdims, b)
    println("replacing conv_bias_act...")

    return ArrayExpr(:call, [GlobalRef(NNlib, :conv_bias_act), x, w, cdims, b, a])
end

flux_conv_fusing_rules = @array_theory a f x w cdims b begin
    # fix to keep in ir
    Flux.conv(x, w, cdims) == Flux.conv(x, w, cdims)
    # TODO fix this with current graphs (Flux.conv seeems different)
    broadcast(a, broadcast(f, Flux.conv(x, w, cdims), b)) => replace_conv_bias_act(f, a, x, w, cdims, b) where istype(f, typeof(Flux.:+))
end

rules = AA.canonicalize_broadcasting ∪ gemm_properties ∪ flux_conv_fusing_rules

nclasses = 10

const chain = fmap(cu, mlp(imgsize=size(xs)[1:end-1], nclasses=nclasses))

println(chain)

args = [xs]
argtype = Tuple{typeof.(args)...}

c = similar(xs, (nclasses, size(xs)[end]))

@array_opt function f(c, x)
    x = chain(x)
    copyto!(c, x)
    return nothing
end

println("warming up")
@CUDA.sync f(c, xs)

println("optimizing")
#f_opt_wo(c, xs) = compile_deferred(f, (c, xs), cost_function)
@eval f_opt_gemm(c, xs) = $(compile_deferred(f, (c, xs), cost_function; extra_rules=gemm_properties ∪ AA.canonicalize_broadcasting))
@eval f_opt_gemm2(c, xs) = $(compile_deferred(f, (c, xs), cost_function; extra_rules=gemm_properties ∪ AA.canonicalize_broadcasting))
#ci = compile_expression(f, argtype, [:xs], cost_function; extra_rules=rules)
#@generated f_opt(xs) = return ci

#println(code_typed(f_opt, argtype))
# Base.invokelatest(f_opt, xs)

println("warming up and testing correctness")
c1 = similar(c)
c2 = similar(c)

#println(f_opt_wo(c1, xs))

# @test isapprox(Array(c1), Array(c2), rtol=1.0, nans=true)

# without rules
#@eval f_norules(xs) = $(compile_with_gpucompiler(f, argtype, [:xs]))

#println("testing w/o rules")
#f_norules(xs)

benchs = load("benchmarks.jld2")
println(benchs)

println("flux before:")
# benchs["flux_before"] = @time CUDA.@sync f($c2, $(xs_f()))
for img_size = [64, 128, 512, 1024]
    let x = xs_f(img_size = img_size)
        samples = []

        CUDA.@sync f(c1, x)
        for i = 1:10
            append!(samples, @elapsed CUDA.@sync f(c1, x))
        end

        benchs["mlp before $img_size"] = samples
    end
end

println("flux after gemm:")
#CUDA.@sync f_opt_gemm(c1, xs)
for img_size = [64, 128, 512, 1024]
    let x = xs_f(img_size = img_size)
        CUDA.@sync f_opt_gemm(c1, x)
        samples = []
        for i = 1:10
            append!(samples, @elapsed CUDA.@sync f_opt_gemm(c1, x))
        end

        benchs["mlp after $img_size"] = samples
    end
end

println(benchs)
save("benchmarks.jld2", benchs)

#=
println("flux after gemm + conv_bias_act")
CUDA.@sync f_opt_gemm_cba(c1, xs)
@benchmark CUDA.@sync f_opt_gemm_cba(c1, xs)
benchs["flux_after_gemm+cba"]
=#

