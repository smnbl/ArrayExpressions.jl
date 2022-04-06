using Symbolics
using Metatheory
using Metatheory: Postwalk, Fixpoint, EGraphs, Chain

using Core.Compiler
const C = Core
const CC = C.Compiler

using TermInterface

ArrayGemm(A, B, C) = error("should still be linked with GemmKernels.jl")

# canonicalize broadcasts using classic term rewriting
const canonicalize_broadcasting = @theory A B op begin
    materialize(A) --> A
    broadcasted(op, A, B) --> broadcast(op, A, B)
end

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
    adjoint(adjoint(A)) == A
end

function Gemm(A::EClass, B::EClass, C::EClass)
    return ArrayExpr(:call, [:Gemm, A, B, C], Union{})
end

function GemmWithEpilogue(A::EClass, B::EClass, C::EClass, epilogue)
    return ArrayExpr(:call, [:GemmWithEpilogue, A, B, C, epilogue], Union{})
end

function Lambda(binding::Symbol, body)
    return ArrayExpr(:->, [binding, body], Union{})
end

function App(func, arguments)
    return ArrayExpr(:app, [func, arguments...], Union{})
end

function istype(X::EClass, type)
    return getdata(X, MetadataAnalysis, Union{}) <: type
end


# big rewrite rules with custom implementations
const gemm_properties = @theory A B C op d epi begin
    # GEMM;
    # TODO: make sure A, B, C is a matrix, and not a scalar!!
    # idea: make mul with scalar separate function? (this supports purely syntactical rewrites)
    # TODO: problem with dynamic rules like this is is that is does not work in the opposite direction
    (A*B) + C => ArrayExpr(:call, [:Gemm, A, B, C], Union{}) where (istype(A, Matrix) && istype(B, Matrix) && istype(C, Matrix))
    # (A*B) + C => ArrayExpr(:call, [:Gemm, A, B, C], Union{}) where (istype(A, Matrix) && istype(B, Matrix) && istype(C, Matrix))

    # merge operations in prologue / epilogue
    # TODO: how to merge with prefix? prologue?
    # TODO: make sure epilogue is scalar
    broadcast(op, Gemm(A, B, C), d) => GemmWithEpilogue(A, B, C, Lambda(:el, App(op, [:el, d])))  where istype(d, Number)

    # fuse operations
    broadcast(op, GemmWithEpilogue(A, B, C, epi), d) => GemmWithEpilogue(A, B, C, Lambda(:el, App(op, [App(epi, [:el]), d])))

    # TODO: add gemm alpha rule
end

# TODO: add map(reduce()) -> mapreduce() rule & other map/reduce ruleshttps://discord.com/channels/442048551342702605/570980600132009987

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

function cost_function(n::ENodeTerm, g::EGraph, an::Type{<:AbstractAnalysis})
    if operation(n) == :GemmWithEpilogue
        cost = 1
    elseif operation(n) == :Gemm
        cost = 10
    elseif !(exprhead(n) == :call || exprhead(n) == :invoke) || operation(n) == :app || operation(n) == :Lambda
        cost = 0
    else
        cost = 100
    end

    for id in arguments(n)
        eclass = g[id]
        # if the child e-class has not yet been analyzed, return +Inf
        !hasdata(eclass, an) && (cost += Inf; break)
        cost += last(getdata(eclass, an))
    end

    return cost
end

cost_function(n::ENodeLiteral, g::EGraph, an::Type{<:AbstractAnalysis}) = 0

# TODO: should we lower dot & plus to map / reduce combos?
# NOTE: as long as tensor_expr satisfies TermInterface, this will work
function simplify(tensor_expr)
    # canonicalize broadcasts (classical rewriting)
    tensor_expr = Postwalk(Chain(canonicalize_broadcasting))(tensor_expr)

    # Equality Saturation
    g = EGraph(tensor_expr; keepmeta = true)

    settermtype!(g, ArrayExpr)

    # saturate graph
    report = saturate!(g, theory)
    # dynamic rewrites using an egraph analysis
    # analyze!(g, ArrayAnalysis)

    # saturate again with the extra analysis info
    # report = saturate!(g, theory)

    # TODO: replace with own cost function
    # astsize: cost function that favors smaller expressions in the equivalence classes
    ex = extract!(g, cost_function)

    return ex
end
