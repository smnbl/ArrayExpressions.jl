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

function Base.show(io::IO, e::ArrayExpr)
    start = 1
    if e.head == :call
        print(io, "$(e.args[1])(")
        start = 2
    elseif e.head == :invoke
        print(io, "$(e.args[2])(")
        start = 3
    elseif e.head == :input
        print(io, "input")
        return
    else
        print(io, "($(e.head))(")
    end

    for arg in e.args[start:end-1]
        Base.show(io, arg)
        print(io, ",")
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
expr_operation(e::ArrayExpr, ::Union{Val{:call}}) = e.args[1]
expr_arguments(e::ArrayExpr, ::Union{Val{:call}}) = e.args[2:end]
expr_operation(e::ArrayExpr, ::Union{Val{:invoke}}) = e.args[2]
expr_arguments(e::ArrayExpr, ::Union{Val{:invoke}}) = e.args[3:end]
expr_operation(e::ArrayExpr, ::Union{Val{:input}}) = :input
expr_arguments(e::ArrayExpr, ::Union{Val{:input}}) = e.args[1:end]

# Add type info via metadata
TI.metadata(e::ArrayExpr) = (type = e.type)

function TI.similarterm(x::Type{ArrayExpr}, head, args, symtype = nothing; metadata = nothing, exprhead = :call)
    ArrayExpr(exprhead, [head, args...], (metadata == nothing) ? Union{} : metadata)
end
