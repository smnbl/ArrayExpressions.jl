using ArrayAbstractions
using Metatheory
using GPUArrays

using Core.Compiler
using Core.Compiler: IRCode, CodeInfo

const CC = Core.Compiler

const AA = ArrayAbstractions

#####

include("gpu/gpu.jl")
