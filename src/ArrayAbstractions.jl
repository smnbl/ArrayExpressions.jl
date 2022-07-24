module ArrayAbstractions
using CUDA

export ArrayIR
export ArrayInterpreter

# types of values that are considered intermediate
ValueTypes = CuArray

export Input, ArrayExpr
# ArrayIR & TermInterface
include("ir.jl")
include("utils.jl")

# interface with Core.Compiler
include("codecache.jl")
#include("compiler_interface.jl")

# Codegen
include("codegen.jl")

export @array_theory, @array_rule, istype
# Equality saturation rules & simplification
include("rules.jl")
include("intrinsics.jl")
include("inlining.jl")
include("interpreter.jl")
include("irops.jl")

end # module
