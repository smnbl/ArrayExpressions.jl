using Metatheory
using Metatheory: Postwalk, Fixpoint, EGraphs, Chain, car, islist

using Core.Compiler
const C = Core
const CC = C.Compiler

using TermInterface

# TODO: proper mod handling in matching GlobalRefs
# TODO: overwriting like this is not allowed (find better way) !
# only works for the classical rewriters
function Metatheory.head_matcher(f::Symbol, mod)
    # x will always be GlobalRef
    checkhead = (x) -> isequal(x, f) || (x isa GlobalRef && isequal(x.name, f))

    function head_matcher(next, data, bindings)
        h = car(data)
        if h isa Input
            h = h.val
        end
        if islist(data) && checkhead(h)
            next(bindings, 1)
        else 
            nothing
        end
    end
end

# TODO: egraphs use another way of comparing operations, check the compile_pat! function in ematch compiler

# canonicalize broadcasts using classic term rewriting
const canonicalize_broadcasting = @theory A B op begin
    materialize(A) --> A
    broadcasted(op, A, B) --> broadcast(op, A, B)
    broadcasted(op, A) --> broadcast(op, A)
    # TODO: generic in nr of arguments
end

# some mathematical properties of matrices
#
const addition_properties = @theory A B C begin
    # commutativity
    A + B == B + A

    # associativity
    broadcast(+, broadcast(+, A, B), C) == broadcast(+, A, broadcast(+, C, B))

    # lowering of addition with itself, useful?
    broadcast(+, C, C) == C .* 2

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

function Gemm!(A::EClass, B::EClass, C::EClass)
    return ArrayExpr(:call, [:Gemm!, A, B, C], Union{})
end

function GemmWithEpilogue(A::EClass, B::EClass, C::EClass, epilogue)
    return ArrayExpr(:call, [:GemmWithEpilogue, A, B, C, epilogue], Union{})
end

function GemmWithEpilogue!(A::EClass, B::EClass, C::EClass, epilogue)
    return ArrayExpr(:call, [:GemmWithEpilogue!, A, B, C, epilogue], Union{})
end

function gettype(X::EClass)
    ty = getdata(X, MetadataAnalysis, Union{})
    return ty
end

function istype(X::EClass, type)
    println("istype")
    ty = gettype(X)
    return ty <: type
end

# big rewrite rules with custom implementations
const gemm_properties = @theory A B C op d epi begin
    # GEMM;
    # TODO: make sure A, B, C is a matrix, and not a scalar!!
    # idea: make mul with scalar separate function? (this supports purely syntactical rewrites)
    # TODO: problem with dynamic rules like this is is that is does not work in the opposite direction
    (A*B) + C => Gemm(A, B, C) where (istype(A, ValueTypes) && istype(B, ValueTypes) && istype(C, ValueTypes))
    # (A*B) + C => ArrayExpr(:call, [:Gemm, A, B, C], Union{}) where (istype(A, Matrix) && istype(B, Matrix) && istype(C, Matrix))

    # merge operations in prologue / epilogue
    # TODO: how to merge with prefix? prologue?
    # TODO: make sure epilogue is scalar
    broadcast(op, Gemm(A, B, C), d) => GemmWithEpilogue(A, B, C, Lambda(:el, App(op, [:el, d])))  where istype(d, Number)
    broadcast(op, d, Gemm(A, B, C)) => GemmWithEpilogue(A, B, C, Lambda(:el, App(op, [d, :el])))  where istype(d, Number)

    # TODO: add support for these type of rules
    # C = Gemm(A, B, C) => Gemm!(A, B, C)
    copyto!(C, Gemm(A, B, C)) == Gemm!(A, B, C)

    # fuse operations
    broadcast(op, Gemm(A, B, C)) => GemmWithEpilogue(A, B, C, op)
    broadcast(op, GemmWithEpilogue(A, B, C, epi), d) => GemmWithEpilogue(A, B, C, Lambda(:el, App(op, [App(epi, [:el]), d])))
    broadcast(op, d, GemmWithEpilogue(A, B, C, epi)) => GemmWithEpilogue(A, B, C, Lambda(:el, App(op, [d, App(epi, [:el])])))
    broadcast(op, GemmWithEpilogue(A, B, C, epi)) => GemmWithEpilogue(A, B, C, Lambda(:el, App(op, [App(epi, [:el])])))

    # TODO: add support for these type of rules
    # C = GemmWithEpilogue(A, B, C, epi) => GemmWithEpilogue!(A, B, C, epi)
    copyto!(C, GemmWithEpilogue(A, B, C, epi)) == GemmWithEpilogue!(A, B, C, epi)

    # TODO: add gemm alpha rule
end

# big rewrite rules with custom implementations
const gemm_properties_classical = @theory A B C op d epi begin
    # GEMM;
    # TODO: make sure A, B, C is a matrix, and not a scalar!!
    # idea: make mul with scalar separate function? (this supports purely syntactical rewrites)
    # TODO: problem with dynamic rules like this is is that is does not work in the opposite direction
    (A*B) + C --> Gemm(A, B, C)
    copyto!(C, Gemm(A, B, C)) --> Gemm!(A, B, C)
end

# TODO: add map(reduce()) -> mapreduce() rule & other map/reduce rules 

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
        cost = 1000
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

const theory = addition_properties ∪ multiplication_properties ∪ addition_properties ∪ gemm_properties

# TODO: should we lower dot & plus to map / reduce combos?
# NOTE: as long as tensor_expr satisfies TermInterface, this will work
function simplify(tensor_expr; extra_rules=Metatheory.AbstractRule[])
    @assert tensor_expr.head == :output

    # remove :output wrapper
    type = tensor_expr.type
    tensor_expr = tensor_expr.args[1]

    # TODO: fix this, typing info is lost :(
    tensor_expr = Postwalk(Chain(canonicalize_broadcasting))(tensor_expr)
    tensor_expr = Postwalk(Chain(gemm_properties_classical))(tensor_expr)
    
    # Equality Saturation
    g = EGraph(tensor_expr; keepmeta = true)

    settermtype!(g, ArrayExpr)

    theories = theory ∪ extra_rules

    # saturate graph
    report = saturate!(g, theories)

    # TODO: replace with own cost function
    # astsize: cost function that favors smaller expressions in the equivalence classes
    ex = extract!(g, cost_function)

    return ArrayExpr(:output, [ex], type)
end
