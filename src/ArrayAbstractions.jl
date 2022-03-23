module ArrayAbstractions

export ArrayIR

# ArrayIR & TermInterface
include("ir.jl")
include("irutils.jl")

# interface with Core.Compiler
include("compiler_interface.jl")

# Equality saturation rules & simplification
include("rules.jl")
include("irops.jl")

end # module
