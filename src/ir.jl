using Core.Compiler: IRCode, SSAValue

const C = Core
const CC = C.Compiler

# functional representation of the array expressions as a Tree-like object
struct ArrayIR
    # expresses the operation & inputs
    op_expr::Expr
    # expresses results of type inference on this expression
    type_expr::Expr
    # TODO add tracking to original code location (for modifications?)
end

struct ArrayIRStream
    ir::Vector{Any}
    args::Vector{Any}
    type::Vector{Any}
end

struct ArrayValue
    type
end

struct ArrayOp
    f::Function
    type
end

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

@noinline function IfElse(condition::Bool, If::StubArray, Else::StubArray)
    if (condition)
        If
    else
        Else
    end
end

# IndexShifts
@noinline function ArrayTranspose(A::StubArray{Tuple{M, N}, T, NA}) where {M, N, T, NA}
    StubArray{Tuple{N, M}, T, N}()
end
