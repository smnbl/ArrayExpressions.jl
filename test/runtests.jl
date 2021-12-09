using ArrayAbstractions
using ArrayAbstractions: StubArray, ArrayDot, ArrayPlus, ArrayGemm

@info "testing GEMM optim"

(M, N, K) = (10, 20, 30)
A = StubArray{Tuple{M, K}, Int}()
B = StubArray{Tuple{K, N}, Int}()
C = StubArray{Tuple{M, N}, Int}()

function f() 
    X = 10
    return ArrayPlus(ArrayDot(A, B), C)
end

ci_new = optimize(f, ())

println(typeof(f()))
println(typeof(ArrayAbstractions.lambda(ci_new)()))
