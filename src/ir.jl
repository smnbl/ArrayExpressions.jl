# Defines Array IR to model array abstractions

# A * B
@noinline function ArrayDot(A::StubArray{Tuple{M, K}, T, N1}, B::StubArray{Tuple{K, N}, T2, N2}) where {M, N, K, T, T2, N1, N2}
    @info "array_dot"
    StubArray{Tuple{M, N}, T, N1}()
end

# A + B
@noinline function ArrayPlus(A::StubArray{Tuple{M, N}, T, N1}, B::StubArray{Tuple{M, N}, T2, N2}) where {M, N, T, T2, N1, N2}
    @info "array plus"
    StubArray{Tuple{M, N}, T, N1}()
end

# A * B + C
# TODO: generalise over tensor contractions? (multi-dimensional)
@noinline function ArrayGemm(A::StubArray{Tuple{M, K}, TA, NA},
                             B::StubArray{Tuple{K, N}, TB, NB},
                             C::StubArray{Tuple{M, N}, TC, NC}) where {M, N, K, TA, TB, NA, NB, TC, NC}
    @info "array_gemm"
    StubArray{Tuple{M, N}, TC, NC}()
end

# transpose(A)
@noinline function ArrayTranspose(A::StubArray)::StubArray
    StubArray{S, T, N}()
end

# from CUDA.jl kernels:
# @noinline function ArrayMapreducedim()
