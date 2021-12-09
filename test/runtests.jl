using ArrayAbstractions
using ArrayAbstractions: StubArray, ArrayDot, ArrayPlus, ArrayGemm

@info "testing GEMM optim"

A, B, C = ntuple(i -> StubArray{Tuple{10, 10}, Int}(), 3)

function f() 
    X = 10
    a = ArrayPlus(ArrayDot(A, B), C)
    println("lmao")
    return 11
end

optimize(f, ())
