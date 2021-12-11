using ArrayAbstractions
using ArrayAbstractions: StubArray, ArrayDot, ArrayPlus, ArrayGemm, codegen

@info "testing GEMM optim"

(M, N, K) = (10, 20, 30)
A = StubArray{Tuple{M, K}, Int}()
B = StubArray{Tuple{K, N}, Int}()
C = StubArray{Tuple{M, N}, Int}()

function f() 
    (M, N, K) = (10, 20, 30)
    A = StubArray{Tuple{M, K}, Int}()
    B = StubArray{Tuple{K, N}, Int}()
    C = StubArray{Tuple{M, N}, Int}()
    X = 10
    F = ArrayDot(A, B)
    return ArrayPlus(F, C)
end

typed = codegen(:llvm, f, Tuple{}, Core.svec())

f()
println(typed)
