using Symbolics
using Metatheory
using Metatheory.EGraphs

using Core.Compiler
const C = Core
const CC = C.Compiler

ArrayGemm(A, B, C) = error("should still be linked with GemmKernels.jl")

# some mathematical properties of matrices
#
const addition_properties = @theory A B C begin
    # commutativity
    broadcast(+, A, B) == broadcast(+, B, A)
    A + B == B + A

    # associativity
    broadcast(+, broadcast(+, A, B), C) == broadcast(+, A, broadcast(+, C, B))
    (A + B) + C == A + (B + C)

    # lowering of addition with itself, useful?
    broadcast(+, C, C) == C .* 2
    (C + C) == C .* 2

    # TODO: add dynamic identity / neutral element rule
end

const multiplication_properties = @theory A B C d begin
    # distributivity of multiplication
    A * (broadcast(+, B, C)) == broadcast(+, A*B, A*C)
    A * (B + C) == A*B + A*C

    # associativity
    (A*B)*C == A*(B*C)

    # adjoint
    adjoint(A*B) == adjoint(B)*adjoint(A)


    # TODO: product with a scalar; confirm it is indeed a scalar (dynamic rules?)
end

const adjoint_properties = @theory A B begin
    adjoint(adjoint(~A)) == A
end

function possibleGemm(A::EClass, B::EClass, C::EClass)
    a_info = getdata(A, ArrayAnalysis, nothing)
    b_info = getdata(B, ArrayAnalysis, nothing)
    c_info = getdata(C, ArrayAnalysis, nothing)

    println("possible_gemm ($a_info, $b_info, $c_info)")

    if a_info isa Symbol && b_info isa Symbol && c_info isa Symbol
        if a_info == :nonscalar && b_info == :nonscalar && c_info == :nonscalar
            throw("TODO: how to construct expression here??")
        elseif a_info == :nonscalar && b_info == :nonscalar && c_info == :nonscalar
            throw("TODO: possible gemm with epilogue mul")
        end
    end

    return nothing
end

#=
@Symbolics.wrapped function Gemm(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
    @Symbolics.syms i::Int j::Int k::Int
    return @Symbolics.arrayop Gemm(A, B, C) (i, j) A[i, k] * B[k, j] + C[i, j]
end

@Symbolics.wrapped function GemmWithEpilogue(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, ep)
    @Symbolics.syms i::Int j::Int k::Int
    return @Symbolics.arrayop GemmWithEpilogue(A, B, C) (i, j) A[i, k] * B[k, j] + C[i, j] + ep
end
=#

function Gemm(A::EClass, B::EClass, C::EClass)
    return Expr(:call, :Gemm, A, B, C)
end

function GemmWithEpilogue(A::EClass, B::EClass, C::EClass, op::EClass, d::EClass)
    return Expr(:call, :GemmWithEpilogue, A, B, C, op, d)
end

function istype(X::EClass, type)
    return getdata(X, MetadataAnalysis, Union{}) <: type
end

# big rewrite rules with custom implementations
const gemm_properties = @theory A B C op d begin
    # GEMM;
    # TODO: make sure A, B, C is a matrix, and not a scalar!!
    # idea: make mul with scalar separate function? (this supports purely syntactical rewrites)
    # TODO: problem with dynamic rules like this is is that is does not work in the opposite direction
    (A*B) + C => ArrayExpr(:call, [:Gemm, A, B, C], Union{}) where (istype(A, Matrix) && istype(B, Matrix) && istype(C, Matrix))
    # (A*B) + C => ArrayExpr(:call, [:Gemm, A, B, C], Union{}) where (istype(A, Matrix) && istype(B, Matrix) && istype(C, Matrix))

    # merge operations in prologue / epilogue
    # TODO: how to merge with prefix? prologue?
    # TODO: make sure epilogue is scalar
    broadcasted(op, Gemm(A, B, C), d) => GemmWithEpilogue(A, B, C, op, d)  where istype(d, Number)

    # TODO: add gemm alpha rule
end

# TODO: add map(reduce()) -> mapreduce() rule & other map/reduce rules

# Define Metatheory rules
const theory = addition_properties ∪ multiplication_properties ∪ addition_properties ∪ gemm_properties

# dynamic rules: egraph-analyses
abstract type ArrayAnalysis <: AbstractAnalysis end

# input type
const ArrayExprInput = Union{CC.SSAValue, CC.PhiNode}

# base step (literals)
function EGraphs.make(an::Type{ArrayAnalysis}, g::EGraph, n::ENodeLiteral)
    if n.value isa Number # scalar
        return :scalar
    else
        return :nonscalar
    end
end

# inductive step (terms)
function EGraphs.make(an::Type{ArrayAnalysis}, g::EGraph, n::ENodeTerm)
    println(n)
    if exprhead(n) == :input
        child_eclasses = arguments(n)
        type_node = g[child_eclasses[1]][1]

        if type_node isa ENodeLiteral
            type = type_node.value

            return (type <: Number) ? :scalar : :nonscalar
        end
        throw("input not ENodeLiteral")
    end

    if (exprhead(n) == :call || exprhead(n) == :invoke) && arity(n) == 1
        child_eclasses = arguments(n)
        op = operation(n)

        info = getdata(g[child_eclasses[1]], an, nothing)

        if op == Symbol("Core.apply_type")
            print("yas")
            return (info == :scalar) ? :scalar : :nonscalar
        end
    end

    #TODO: support more than binary operations
    if (exprhead(n) == :call || exprhead(n) == :invoke) && arity(n) == 2
        op = operation(n)
        child_eclasses = arguments(n)
        l = g[child_eclasses[1]]
        r = g[child_eclasses[2]]

        linfo = getdata(l, an, nothing)
        rinfo = getdata(r, an, nothing)
        
        # TODO: add more operations; think of a smarter way to propagate :scalar tag
        if linfo isa Symbol && rinfo isa Symbol
            if op == :* || op == :+
                return (linfo == :scalar && rinfo == :scalar) ? :scalar : :nonscalar
            end
        end
    end

    return nothing
end

# merge analysis values
function EGraphs.join(an::Type{ArrayAnalysis}, a, b)
    if (a == b)
        return a
    else
        # this is actually contradictory
        return nothing
    end
end

# TODO: why are we implementing these?
function EGraphs.make(an::Type{MetadataAnalysis}, g::EGraph, n::ENodeTerm)
    return Union{}
end

function EGraphs.join(an::Type{MetadataAnalysis}, a, b)
    if (a == Union{})
        return b
    else
        return a
    end
end

# TODO: should we lower dot & plus to map / reduce combos?
# NOTE: as long as tensor_expr satisfies TermInterface, this will work
function simplify(tensor_expr)
    g = EGraph(tensor_expr; keepmeta = true)
    settermtype!(g, ArrayExpr)

    # saturate graph
    report = saturate!(g, theory)
    # dynamic rewrites using an egraph analysis
    # analyze!(g, ArrayAnalysis)

    # saturate again with the extra analysis info
    report = saturate!(g, theory)

    # TODO: replace with own cost function
    # astsize: cost function that favors smaller expressions in the equivalence classes
    ex = extract!(g, astsize)
    return ex
end
