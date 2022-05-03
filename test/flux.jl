using Flux
using Metatheory
using ArrayAbstractions

const AA = ArrayAbstractions

include("compile.jl")

function test(xs, bias)
    layer = Conv((5,5), 3 => 7, relu; bias = bias)

    layer(xs)
end

xs = rand(Float32, 100, 100, 3, 50)
bias = randn(Float32, 7)

args = [xs, bias]
argtype = Tuple{typeof.(args)...}

flux_conv_fusing_rules = @theory begin
    broadcast(~a, broadcast(+, conv(~x, ~w, ~cdims), ~b)) == conv_bias_act(~x, ~w, ~cdims, ~b, ~a)
end

# expr = AA.optimize(:cache, layer, (typeof(xs),), Core.svec(), extra_rules=flux_conv_fusing_rules)
@eval optim(xs, bias) = $(compile(test, argtype, [:xs, :bias], extra_rules=flux_conv_fusing_rules))

println(isapprox(test(xs, bias), optim(xs, bias)))

println("before:")
@time test(xs, bias)

println("after:")
@time optim(xs, bias)
