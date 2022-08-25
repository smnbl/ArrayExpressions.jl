using Flux
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay
using Flux: onehotbatch, onecold, flatten
using Flux.Losses: logitcrossentropy
using Flux: AbstractRNG, rng_from_array, nfan
using NNlib
using CUDA
using BenchmarkTools

using ArrayAbstractions
const AA = ArrayAbstractions

include("compile.jl")
include("gpu/gpu_rules.jl")

function LeNet5(; imgsize=(28,28,1), nclasses=10) 
    out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)
    
    return Chain(
            Conv((5, 5), imgsize[end]=>6, relu; init=glorot_uniform_16),
            MaxPool((2, 2)),
            Conv((5, 5), 6=>16, relu; init=glorot_uniform_16),
            MaxPool((2, 2)),
            flatten,
            # elementwise operations?
            # added as a test
            Dense(prod(out_conv_size), 120, relu; init=glorot_uniform_16), 
            Dense(120, 84, relu; init=glorot_uniform_16), 
            Dense(84, nclasses; init=glorot_uniform_16)
    )
end

xs = CuArray(rand(Float16, 256, 256, 10, 50))

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

const chain = fmap(cu, LeNet5(imgsize=size(xs)[1:end-1], nclasses=nclasses))

args = [xs]
argtype = Tuple{typeof.(args)...}

c = similar(xs, (nclasses, size(xs)[end]))

@array_opt function f(c, x)
    x = chain.layers[1](x)
    x = chain.layers[2](x)
    x = chain.layers[3](x)
    x = chain.layers[4](x)
    x = chain.layers[5](x)
    x = chain.layers[6](x)
    x = chain.layers[7](x)
    x = chain.layers[8](x)
    copyto!(c, x)
    return nothing
end

f(c, xs)

@eval f_opt_wo(c, xs) = $(compile_deferred(f, (c, xs), cost_function))
@eval f_opt_gemm(c, xs) = $(compile_deferred(f, (c, xs), cost_function; extra_rules=gemm_properties ∪ AA.canonicalize_broadcasting))
@eval f_opt_gemm_cba(c, xs) = $(compile_deferred(f, (c, xs), cost_function; extra_rules=gemm_properties ∪ AA.canonicalize_broadcasting ∪ flux_conv_fusing_rules))
#ci = compile_expression(f, argtype, [:xs], cost_function; extra_rules=rules)
#@generated f_opt(xs) = return ci

#println(code_typed(f_opt, argtype))
# Base.invokelatest(f_opt, xs)

println("warming up and testing correctness")
c1 = similar(c)
c2 = similar(c)
c3 = similar(c)

f_opt_wo(c1, xs)
f_opt_gemm(c2, xs)
f_opt_gemm_cba(c2, xs)

@test isapprox(Array(c1), Array(c2), rtol=1.0, nans=true)

# without rules
#@eval f_norules(xs) = $(compile_with_gpucompiler(f, argtype, [:xs]))

#println("testing w/o rules")
#f_norules(xs)


println("flux before:")
CUDA.@sync f_opt_wo(c2, xs)
bench = @benchmark CUDA.@sync f_opt_wo(c2, xs)
println(bench)

println("flux after gemm:")
CUDA.@sync f_opt_gemm(c1, xs)
bench = @benchmark CUDA.@sync f_opt_gemm(c1, xs)
println(bench)

println("flux after gemm + conv_bias_act")
CUDA.@sync f_opt_gemm_cba(c1, xs)
@benchmark CUDA.@sync f_opt_gemm_cba(c1, xs)
println(bench)

