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

# Float16 version of Flux.glorot_uniform, necessary as otherwirse the weight matrices get initialized to Float32 (GemmKernels.jl doesn't support mixed precision gemm atm)
function glorot_uniform_16(rng::AbstractRNG, dims::Integer...; gain::Real=1)
   scale = Float16(gain) * Float16(sqrt(24.0f0 / sum(nfan(dims...))))
     (rand(rng, Float16, dims...) .- Float16(0.5f0)) .* scale
     end
glorot_uniform_16(dims::Integer...; kw...) = glorot_uniform_16(rng_from_array(), dims...; kw...)
glorot_uniform_16(rng::AbstractRNG=rng_from_array(); init_kwargs...) = (dims...; kwargs...) -> glorot_uniform_16(rng, dims...; init_kwargs..., kwargs...)

function mlp(; imgsize=(28,28,1), nclasses=10) 
    return Chain(
                 flatten,
                 Dense(prod(imgsize), 32, relu, init=glorot_uniform_16),
                 Dense(32, nclasses, init=glorot_uniform_16))
end

img_size = parse(Int, ARGS[1])
xs_f(;img_size=32) = CuArray(rand(Float16, img_size, img_size, 1, 50))
xs = xs_f(img_size=img_size)

nclasses = 10

args = [xs]
argtype = Tuple{typeof.(args)...}

c = similar(xs, (nclasses, size(xs)[end]))

const mlp_chain = fmap(cu, mlp(imgsize=size(xs)[1:end-1], nclasses=nclasses))

@array_opt function f(c, x)
    x = lenet_chain.layers[1](x)
    x = lenet_chain.layers[2](x)
    x = lenet_chain.layers[3](x)
    copyto!(c, x)
    nothing
end

function bench_mlp_single(xs, n_samples=100)
    # for a fair comparison and to remove the effects of semi-inlining, compare with a version that is optimized without rewrite rules
    @eval f_normal(c, xs) = $(compile_deferred(f, (c, xs), cost_function))
    @eval f_opt_gemm(c, xs) = $(compile_deferred(f, (c, xs), cost_function; extra_rules=gemm_properties ∪ AA.canonicalize_broadcasting))
    #ci = compile_expression(f, argtype, [:xs], cost_function; extra_rules=rules)
    #@generated f_opt(xs) = return ci

    #println(code_typed(f_opt, argtype))
    # Base.invokelatest(f_opt, xs)

    println("testing: mlp ($(size(xs)))")
    println("warming up")
    c1 = similar(c)
    c2 = similar(c)

    #println(f_opt_wo(c1, xs))

    c1 = similar(xs, (nclasses, size(xs)[end]))

    # before
    println("before")
    CUDA.@sync Base.invokelatest(f_normal, c1, xs)

    
    before = @benchmark CUDA.@sync Base.invokelatest($f_normal, $c1, $xs)
    println(before)

    #before = []
    #for i = 1:n_samples
        #append!(before, @elapsed CUDA.@sync Base.invokelatest(f_normal, c1, xs))
    #end

    println("after")
    c2 = similar(xs, (nclasses, size(xs)[end]))
    CUDA.@sync Base.invokelatest(f_opt_gemm, c2, xs)

    after = @benchmark CUDA.@sync Base.invokelatest($f_opt_gemm, $c2, $xs)
    println(after)
    
    #println(Array(c1) - Array(c2))
    #@assert isapprox(Array(c1), Array(c2), rtol=sqrt(sqrt(eps(Float16))), nans=true)

    #after = []
    #for i = 1:n_samples
        #append!(after, @elapsed CUDA.@sync Base.invokelatest(f_opt_gemm, c1, xs))
    #end

    return before, after
end


before, after = bench_mlp_single(xs)
println("before: $before")
println("after: $after")

benchs = Dict{Any, Any}()
try
    global benchs = load("benchmarks.jld2")

catch
end

benchs["mlp before $img_size"] = before
benchs["mlp after $img_size"] = after

save("benchmarks.jld2", benchs)

#=
println("flux after gemm + conv_bias_act")
CUDA.@sync f_opt_gemm_cba(c1, xs)
@benchmark CUDA.@sync f_opt_gemm_cba(c1, xs)
benchs["flux_after_gemm+cba"]
=#

