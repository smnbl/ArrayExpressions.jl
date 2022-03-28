using Core.Compiler: IRCode, SSAValue, PhiNode
using GPUArrays

using TermInterface

const C = Core
const CC = C.Compiler
const TI = TermInterface

const InputTypes = Any

struct ArrayExpr
    head::Any
    args::Vector{Any}
    type::Type
end

import Base.(==)
(==)(a::ArrayExpr, b::ArrayExpr) = a.head == b.head && a.args == b.args

convert(::Type{Expr}, expr::ArrayExpr) = Expr(expr.head, expr.args...)

function Base.show(io::IO, e::ArrayExpr)
    start = 1
    if e.head == :call
        print(io, "$(e.args[1])(")
        start = 2
    elseif e.head == :invoke
        print(io, "$(e.args[2])(")
        start = 3
    elseif e.head == :->
        print(io, "$(e.args[1]) -> ")
        start = 2
    else
        print(io, "($(e.head))(")
    end

    for arg in e.args[start:end-1]
        Base.show(io, arg)
        print(io, ", ")
    end
    if length(e.args) > 0
        Base.show(io, e.args[end])
    end
    print(io, ")")
end

# functional representation of the array expressions as a Tree-like object
struct ArrayIR
    # expresses the operation & inputs
    op_expr::ArrayExpr

    # location of all the input locations aka the point at which the inputs are fed to the expression
    # this is used when replacing an expression by removing all the intermediate operations that get replaced by the optimised kernel
    # these can be the location of phi nodes or GlobalRefs to the used array variables
    input_locs::Set{InputTypes}
    ArrayIR(op::ArrayExpr) = new(op, Set())
end

# Make ArrayExpr support TermInterface.jl
# modified from the Expr implementation
TI.istree(x::Type{ArrayExpr}) = true
TI.exprhead(e::ArrayExpr) = e.head

TI.operation(e::ArrayExpr) = expr_operation(e, Val{exprhead(e)}())
TI.arguments(e::ArrayExpr) = expr_arguments(e, Val{exprhead(e)}())

# See https://docs.julialang.org/en/v1/devdocs/ast/
# TODO: safe to make Symbols?
expr_operation(e::ArrayExpr, ::Union{Val{:call}}) = Symbol(e.args[1])
expr_arguments(e::ArrayExpr, ::Union{Val{:call}}) = e.args[2:end]
expr_operation(e::ArrayExpr, ::Union{Val{:invoke}}) = Symbol(e.args[2])
expr_arguments(e::ArrayExpr, ::Union{Val{:invoke}}) = e.args[3:end]

expr_operation(e::ArrayExpr, ::Val{T}) where T = T
expr_arguments(e::ArrayExpr, _) = e.args

# Add type info via metadata
TI.metadata(e::ArrayExpr) = (type = e.type)

# will be fixed in later release of Metatheory
function TI.similarterm(x::ArrayExpr, head, args, symtype = nothing; metadata = nothing, exprhead = :call)
    TI.similarterm(ArrayExpr, head, args, nothing; metadata = nothing, exprhead = :call)
end

function TI.similarterm(x::Type{ArrayExpr}, head, args, symtype = nothing; metadata = nothing, exprhead = :call)
    expr_similarterm(head, args, (metadata == nothing) ? Union{} : metadata, Val{exprhead}())
end

expr_similarterm(head, args, type, ::Val{:call}) = ArrayExpr(:call, [head, args...], type)
expr_similarterm(head, args, type, ::Val{:invoke}) = ArrayExpr(:invoke, [head, args...], type)
expr_similarterm(head, args, type, ::Val{eh}) where {eh} = ArrayExpr(eh, args, type)
