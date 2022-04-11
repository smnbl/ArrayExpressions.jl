using Flux
using Metatheory
using ArrayAbstractions

const AA = ArrayAbstractions

xs = rand(Float32, 100, 100, 3, 50)

bias = randn(Float32, 7)

layer = Conv((5,5), 3 => 7, relu; bias = bias)

flux_conv_fusing_rules = @theory begin
    broadcast(~a, broadcast(+, conv(~x, ~w, ~cdims), ~b)) == conv_bias_act(~x, ~w, ~cdims, ~b, ~a)
end

expr = AA.optimize(:cache, layer, (typeof(xs),), Core.svec(), extra_rules=flux_conv_fusing_rules)
