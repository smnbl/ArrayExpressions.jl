using ArrayAbstractions
using ArrayAbstractions: StubArray, ArrayDot, ArrayPlus, ArrayGemm, codegen
using Metatheory

#ArrayAbstractions.inject_typeinf()

@info "testing GEMM optim"
(M, N, K) = (10, 20, 30)
A = StubArray{Tuple{M, K}, Int}()
B = StubArray{Tuple{K, N}, Int}()
C = StubArray{Tuple{M, N}, Int}()

@array_opt function f()
    (M, N, K) = (10, 20, 30)
    A = StubArray{Tuple{M, K}, Int}()
    B = StubArray{Tuple{K, N}, Int}()
    C = StubArray{Tuple{M, N}, Int}()
    X = 10
    F = ArrayDot(A, B)
    return ArrayPlus(F, C)
end

@array_opt function f_opt()
    F = ArrayDot(A, B)
    println("hello :)")
    println("waat")
    X = ArrayPlus(F, C)
    return X
end

using InteractiveUtils
println(InteractiveUtils.@code_typed f_opt())

codegen(:typed, f_opt, (), Core.svec())
