using Core.Compiler: IRCode, SSAValue, PhiNode

using TermInterface

const C = Core
const CC = C.Compiler
const TI = TermInterface

const InputTypes = Any

struct Output
    val::Any
    type::Type
    Output(val::Any) = new(val, Any)
    Output(val::Any, type::Type) = new(val, type)
end

struct Input
    val::Any
    type::Type
    Input(val::Any) = new(val, Any)
    Input(val::Any, type::Type) = new(val, type)
end

Base.isequal(input::Input, other) = input.val == other
Base.isequal(other, input::Input) = input.val == other
Base.isequal(input::Input, input2::Input) = input.val == input2.val

struct ArrayExpr
    head::Any
    args::Vector{Any}
    type::Type
    ArrayExpr(head, args) = new(head, args, Any)
    ArrayExpr(head, args, type) = new(head, args, type)
end

const ArrayIR = Union{Input, ArrayExpr}

import Base.(==)
(==)(a::ArrayExpr, b::ArrayExpr) = a.head == b.head && a.args == b.args
           

convert(::Type{Expr}, expr::ArrayExpr) = Expr(expr.head, expr.args...)

function Lambda(binding::Symbol, body)
    return ArrayExpr(:->, [binding, body], Union{})
end

function App(func, arguments)
    return ArrayExpr(:app, [func, arguments...], Union{})
end

function Phi(edges, values, type=Union{})
    return ArrayExpr(:ϕ, [edges, values], type)
end

function Base.show(io::IO, e::Input)
    print(io, "→$(e.val)")
end

function Base.show(io::IO, e::ArrayExpr)
    start = 1
    if e.head == :call
        println(io, "$(e.args[1])(")
        start = 2
    elseif e.head == :invoke
        print(io, "$(e.args[2])(")
        start = 3
    elseif e.head == :->
        print(io, "$(e.args[1]) -> ")
        start = 2
    elseif e.head == :ϕ
        print(io, "ϕ(")
        # only print the values
        for arg in e.args[2][1:end-1]
            print(io, arg)
            print(io, ", ")
        end
        print(io, e.args[2][end])
        print(io, ")")
        return
    else
        print(io, "($(e.head))(")
    end

    for arg in e.args[start:end-1]
        Base.show(io, arg)
        print(io, ",\n ")
    end
    if length(e.args) > 0
        Base.show(io, e.args[end])
    end
    print(io, ")::$(e.type)")
end


# Make ArrayExpr support TermInterface.jl
# modified from the Expr implementation
TI.istree(x::Type{ArrayExpr}) = true
TI.exprhead(e::ArrayExpr) = e.head

TI.operation(e::ArrayExpr) = expr_operation(e, Val{exprhead(e)}())
TI.arguments(e::ArrayExpr) = expr_arguments(e, Val{exprhead(e)}())

# See https://docs.julialang.org/en/v1/devdocs/ast/
function expr_operation(e::ArrayExpr, ::Union{Val{:call}})
    # TODO: fix this hack by fixing matching with function objects!
    e.args[1]
end

function expr_operation(e::ArrayExpr, ::Union{Val{:invoke}})
    e.args[1]
    #= FIXME BY FIXING MATCHING
    if (e.args[2] isa ArrayExpr && e.args[2].head == :call && e.args[2].args[1] == :input)
        e.args[2].args[2]
    else
        e.args[2]
    end
    =#
end

expr_arguments(e::ArrayExpr, ::Union{Val{:call}}) = e.args[2:end]
expr_arguments(e::ArrayExpr, ::Union{Val{:invoke}}) = e.args[3:end]

expr_operation(e::ArrayExpr, ::Val{T}) where T = T
expr_arguments(e::ArrayExpr, _) = e.args

# Add type info via metadata
TI.metadata(e::ArrayExpr) = (type = e.type)
TI.metadata(e::Input) = (type = e.type)

# will be fixed in later release of Metatheory
function TI.similarterm(x::ArrayExpr, head, args, symtype = nothing; metadata = nothing, exprhead = :call)
    TI.similarterm(ArrayExpr, head, args, symtype; metadata, exprhead = :call)
end

function TI.similarterm(x::Type{ArrayExpr}, head, args, symtype = nothing; metadata = nothing, exprhead = :call)
    expr_similarterm(head, args, isnothing(metadata) ? Union{} : metadata, Val{exprhead}())
end

expr_similarterm(head, args, type, ::Val{:call}) = ArrayExpr(:call, [head, args...], type)

# TODO: we lose invokes -> might result in performance degradation
expr_similarterm(head, args, type, ::Val{:invoke}) = ArrayExpr(:call, [head, args...], type)
expr_similarterm(head, args, type, ::Val{eh}) where {eh} = ArrayExpr(eh, args, type)
