# How to implement ArrayAbstractions in Julia
# step 1: use functions to define ArrayAbstraction IR (similar to XLA.jl); flag as intrinsic to prevent inlining
# step 2: modify compiler to generate array abstractions IR from expressions
# step 3: start pattern matching on ArrayAbstractions (look at macrotools capture macros? maybe?)
