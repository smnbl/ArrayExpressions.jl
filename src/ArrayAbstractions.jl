module ArrayAbstractions

export ArrayIR

include("array.jl")
include("ir.jl")

include("term_interface.jl")
include("irutils.jl")
include("compiler_interface.jl")

include("rules.jl")
include("irops.jl")

end # module
