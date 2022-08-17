using ArrayAbstractions
using Metatheory
using GPUArrays
using Test

using Core.Compiler
using Core.Compiler: IRCode, CodeInfo

const CC = Core.Compiler

const AA = ArrayAbstractions

macro test_if(label, expr)
    return quote
        if isempty(ARGS) || $(label) in ARGS
            $(esc(expr))
        else
            nothing
        end
    end
end

# Unit tests
@info "Performing Unit Tests..."
@test_if "rules" include("rules.jl")

#=
@test_if "injection" include("injection.jl")
=#

# Integration tests
@info "Performing Intergration Tests..."
@test_if "gpu" include("gpu/gpu.jl")
# @test_if "flux" include("flux.jl")
# include("tensat.jl")
# include("conv2d.jl")
include("conv_mnist.jl")
