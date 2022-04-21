module ArrayAbstractions

export ArrayIR

using GPUArrays

# types of values that are considered intermediate
const ValueTypes = Union{AbstractGPUArray}

# ArrayIR & TermInterface
include("ir.jl")
include("irutils.jl")

# interface with Core.Compiler
include("codecache.jl")
include("compiler_interface.jl")

# Codegen
include("codegen.jl")

# Equality saturation rules & simplification
include("rules.jl")
include("irops.jl")


end # module
