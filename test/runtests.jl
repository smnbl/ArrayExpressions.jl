using ArrayAbstractions
using Metatheory
using GPUArrays
using Symbolics: unwrap, @variables

const AA = ArrayAbstractions

# implementation of AbstractGPUArray on CPU
include("jlarray.jl")
using .JLArrays

#ArrayAbstractions.inject_typeinf()

@info "testing GEMM optim"
const (M, N, K) = (10, 20, 30)

# temporary hack
function get_jl_array()
    println("replacement called!")
    C = JLArray(rand(Float32,(M, N)))
    return C
end

@array_opt function f_opt()
    A = JLArray(rand(Float32,(M, K)))
    B = JLArray(rand(Float32,(K, N)))
    C = JLArray(rand(Float32,(M, N)))

    return C + A * B .+ 2
end

println(AA.codegen(:typed, f_opt, (), Core.svec()))


