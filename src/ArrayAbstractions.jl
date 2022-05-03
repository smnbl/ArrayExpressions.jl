module ArrayAbstractions
using CUDA

export ArrayIR

# types of values that are considered intermediate
ValueTypes = CuArray

# ArrayIR & TermInterface
include("ir.jl")
include("utils.jl")

# interface with Core.Compiler
include("codecache.jl")
include("compiler_interface.jl")

# Codegen
include("codegen.jl")

# Equality saturation rules & simplification
include("rules.jl")
include("irops.jl")


end # module
