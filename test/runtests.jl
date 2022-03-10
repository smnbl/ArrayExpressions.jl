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

    return T + C
end

# AA.codegen(:inject, f_opt, (), Core.svec())

#using InteractiveUtils
#@code_warntype f_opt()
#@code_llvm f_opt()

# f_opt()

@variables X[1:10,1:5] Y[1:5, 1:10] C[1:10,1:10] D[1:10,1:10] d;

t = X*Y + C
t2 = C + C + C
t3 = d .+ C + X*Y
t4 = C + D + C + D
t5 = X * Y + C + D
t6 = C + X * Y + C

println(ArrayAbstractions.simplify(unwrap(t)))
println(ArrayAbstractions.simplify(unwrap(t2)))
println(ArrayAbstractions.simplify(unwrap(t3)))
println(ArrayAbstractions.simplify(unwrap(t4)))
println(ArrayAbstractions.simplify(unwrap(t5)))
println(ArrayAbstractions.simplify(unwrap(t6)))
