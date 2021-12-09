# Defines Array IR to model array abstractions

# A * B
@noinline function ArrayDot(A::StubArray{S, T, N}, B::StubArray{S2, T2, N2}) where {S, S2, T, T2, N, N2}
    StubArray{S, T, N}()
end

# A + B
@noinline function ArrayPlus(A::StubArray{S, T, N}, B::StubArray{S2, T2, N2}) where {S, S2, T, T2, N, N2}
    StubArray{S, T, N}()
end

# A * B + C
# TODO: generalise over tensor contractions? (multi-dimensional)
@noinline function ArrayGemm(A::StubArray{Tuple{M, K}, TA, NA},
                             B::StubArray{Tuple{K, N}, TB, NB},
                             C::StubArray{Tuple{M, N}, TC, NC}) where {M, N, K, TA, TB, NA, NB, TC, NC}
    StubArray{Tuple{M, N}, TC, NC}()
end

# transpose(A)
@noinline function ArrayTranspose(A::StubArray)::StubArray
    StubArray{S, T, N}()
end

# from CUDA.jl kernels:
# @noinline function ArrayMapreducedim()
