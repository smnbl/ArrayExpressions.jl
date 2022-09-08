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

function LeNet5(; imgsize=(28,28,1), nclasses=10) 
    out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)
    
    return Chain(
            Conv((5, 5), imgsize[end]=>6, relu),
            MaxPool((2, 2)),
            Conv((5, 5), 6=>16, relu),
            MaxPool((2, 2)),
            flatten,
            # elementwise operations?
            # added as a test
            Dense(prod(out_conv_size), 120, relu), 
            Dense(120, 84, relu), 
            Dense(84, nclasses)
    )
end

xs_f(;img_size=32) = CuArray(rand(Float32, img_size, img_size, 1, 50))
xs = xs_f()

rules = AA.canonicalize_broadcasting ∪ gemm_properties

nclasses = 10

const chain = fmap(cu, LeNet5(imgsize=size(xs)[1:end-1], nclasses=nclasses))

args = [xs]
argtype = Tuple{typeof.(args)...}

c = similar(xs, (nclasses, size(xs)[end]))

@array_opt function f(c, x)
    x = chain(x)
    copyto!(c, x)
    nothing
end

@CUDA.sync f(c, xs)

#f_opt_wo(c, xs) = compile_deferred(f, (c, xs), cost_function)
@eval f_opt_gemm(c, xs) = $(compile_deferred(f, (c, xs), cost_function; extra_rules=gemm_properties ∪ AA.canonicalize_broadcasting))
#ci = compile_expression(f, argtype, [:xs], cost_function; extra_rules=rules)
#@generated f_opt(xs) = return ci

#println(code_typed(f_opt, argtype))
# Base.invokelatest(f_opt, xs)

println("warming up and testing correctness")
c1 = similar(c)
c2 = similar(c)

#println(f_opt_wo(c1, xs))
println(f_opt_gemm(c2, xs))

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
        println(img_size)
        # warmup
        @time CUDA.@sync f(c1, x)
        @time CUDA.@sync f(c1, x)

        samples = []
        for i = 1:10
            append!(samples, @elapsed CUDA.@sync f(c1, x))
        end

        benchs["lenet before $img_size"] = samples
    end
end

println("flux after gemm:")
#CUDA.@sync f_opt_gemm(c1, xs)
for img_size = [64, 128, 512, 1024]
    let x = xs_f(img_size = img_size)

        println(img_size)

        CUDA.@sync f_opt_gemm(c1, x)
        
        #@assert isapprox(Array(c1), Array(c2), rtol=sqrt(sqrt(eps(Float16))), nans=true)

        samples = []
        for i = 1:10
            append!(samples, @elapsed CUDA.@sync f_opt_gemm(c1, x))
        end

        benchs["lenet after $img_size"] = samples
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
