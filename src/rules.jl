using Metatheory
using Metatheory: Postwalk, Fixpoint, EGraphs, Chain, car, islist

using Core.Compiler
const C = Core
const CC = C.Compiler

using TermInterface

# treat as a literal
function makepattern(x, pvars, slots, mod=@__MODULE__, splat=false) 
    if splat 
        x in slots ? Metatheory.Syntax.makesegment(x, pvars) : x
    else
        x in slots ? Metatheory.Syntax.makevar(x, pvars) : x
    end
end

function makepattern(ex::Expr, pvars, slots, mod=@__MODULE__, splat=false)
    head = exprhead(ex)
    op = operation(ex)
    args = arguments(ex)

    #throw(Meta.ParseError("Unsupported pattern syntax $ex"))
    
    if head === :call
        if istree(op)
            op = makepattern(op, pvars, slots, mod)
        elseif op isa Symbol
            # TODO: investigate what to supply as mod ?
            # HACK
            op = QuoteNode(GlobalRef(mod, op))
        end

        if operation(ex) === :(~) # is a variable or segment
            if args[1] isa Expr && operation(args[1]) == :(~)
                # matches ~~x::predicate or ~~x::predicate...
                return Metatheory.Syntax.makesegment(arguments(args[1])[1], pvars)
            elseif splat
                # matches ~x::predicate...
                return Metatheory.Syntax.makesegment(args[1], pvars)
            else
                return Metatheory.Syntax.makevar(args[1], pvars)
            end
        else # is a term
            patargs = map(i -> makepattern(i, pvars, slots, mod), args) # recurse
            return :($PatTerm(:call, $op, [$(patargs...)], $mod))
        end
    elseif head === :... 
        makepattern(args[1], pvars, slots, mod, true)
    elseif head == :(::) && args[1] in slots
        return splat ? Metatheory.Syntax.makesegment(ex, pvars) : Metatheory.Syntax.makevar(ex, pvars)
    elseif head === :ref 
        # getindex 
        patargs = map(i -> makepattern(i, pvars, slots, mod), args) # recurse
        return :($PatTerm(:ref, getindex, [$(patargs...)], $mod))
    elseif head === :$
        return args[1]
    # NEW:
    elseif head === :.
        m = resolveref(args[1], mod)
        # HACK
        return QuoteNode(GlobalRef(m, args[2].value))
    else 
        patargs = map(i -> makepattern(i, pvars, slots, mod), args) # recurse
        return :($PatTerm($(head isa Symbol ? QuoteNode(head) : head), $(op isa Symbol ? QuoteNode(op) : op), [$(patargs...)], $mod))
        # throw(Meta.ParseError("Unsupported pattern syntax $ex"))
    end
end

function resolveref(val, mod)
    if (CC.isexpr(val, :.))
        if(val.args[2] isa QuoteNode)
            val.args[2] = val.args[2].value
        end
        getproperty(resolveref(val.args[1], mod), val.args[2])
    else
        if (val isa QuoteNode)
            val = val.value
        end
        getproperty(mod, val)
    end
end

function addslots(expr, slots)
    if expr isa Expr
        if expr.head === :macrocall && expr.args[1] in [Symbol("@rule"), Symbol("@capture"), Symbol("@slots"), Symbol("@array_theory"), Symbol("@array_rule"), Symbol("@array_theory")]
            Expr(:macrocall, expr.args[1:2]..., slots..., expr.args[3:end]...)
        else
            Expr(expr.head, addslots.(expr.args, (slots,))...)
        end
    else
        expr
    end
end


# based on Metathory's @rule macro
macro array_rule(args...)
    length(args) >= 1 || ArgumentError("@rule requires at least one argument")
    slots = args[1:(end-1)]
    expr = args[end]

    e = macroexpand(__module__, expr)
    e = Metatheory.Syntax.rmlines(e)
    op = operation(e)
    RuleType = Metatheory.Syntax.rule_sym_map(e)
    
    l, r = arguments(e)
    pvars = Symbol[]
    lhs = makepattern(l, pvars, slots, __module__)
    rhs = RuleType <: SymbolicRule ? esc(makepattern(r, [], slots, __module__)) : r

    if RuleType == DynamicRule
        rhs = Metatheory.Syntax.rewrite_rhs(r)
        rhs = Metatheory.Syntax.makeconsequent(rhs)
        params = Expr(:tuple, :_lhs_expr, :_subst, :_egraph, pvars...)
        rhs =  :($(esc(params)) -> $(esc(rhs)))
    end

    return quote
        $(__source__)
        ($RuleType)($(QuoteNode(expr)), $(esc(lhs)), $rhs)
    end
end

# based on Metatheory's @theory macro
macro array_theory(args...)
    length(args) >= 1 || ArgumentError("@rule requires at least one argument")
    slots = args[1:(end - 1)]
    expr = args[end]

    e = macroexpand(__module__, expr)
    e = Metatheory.Syntax.rmlines(e)
    # e = interp_dollar(e, __module__)

    if exprhead(e) == :block
        ee = Expr(:vect, map(x -> addslots(:(@array_rule($x)), slots), arguments(e))...)
        esc(ee)
    else
        error("theory is not in form begin a => b; ... end")
    end
end

# canonicalize broadcasts using classic term rewriting
const canonicalize_broadcasting = @array_theory A B op begin
    Base.materialize(A) --> A
    Base.broadcasted(op, A, B) --> Main.broadcast(op, A, B)
    Base.broadcasted(op, A) --> Main.broadcast(op, A)
    # TODO: generic in nr of arguments
end

# some mathematical properties of matrices
#
const addition_properties = @array_theory A B C begin
    # commutativity
    A + B == B + A

    # associativity
    broadcast(+, broadcast(+, A, B), C) == broadcast(+, A, broadcast(+, C, B))

    # lowering of addition with itself, useful?
    broadcast(+, C, C) == C .* 2

    # TODO: add dynamic identity / neutral element rule
end

const multiplication_properties = @array_theory A B C d begin
    # distributivity of multiplication
    A * (broadcast(+, B, C)) == broadcast(+, A*B, A*C)
    A * (B + C) == A*B + A*C

    # associativity
    (A*B)*C == A*(B*C)

    # adjoint
    adjoint(A*B) == adjoint(B)*adjoint(A)

    # TODO: product with a scalar; confirm it is indeed a scalar (dynamic rules?)
end

const adjoint_properties = @array_theory A B begin
    adjoint(adjoint(A)) == A
end

function gettype(X::EClass)
    hasdata(X, MetadataAnalysis) || return Any
    ty = getdata(X, MetadataAnalysis)
    if (ty == nothing)
        Any
    else
        ty
    end
end

function istype(X::EClass, type::Type)
    ty = CC.widenconst(gettype(X))
    return ty <: type
end

# TODO: why are we implementing these?
function EGraphs.make(an::Type{MetadataAnalysis}, g::EGraph, n::ENodeLiteral)
    return typeof(n.value)
end

# TODO: why are we implementing these?
function EGraphs.make(an::Type{MetadataAnalysis}, g::EGraph, n::ENodeTerm)
    return Any
end

function EGraphs.join(an::Type{MetadataAnalysis}, a, b)
    if (a == Any)
        return b
    else
        return a
    end
end

function cost_function(n::ENodeTerm, g::EGraph, an::Type{<:AbstractAnalysis})
    op = operation(n)
    if op isa Input
        op = op.val
    end

    args = arguments(n)

    if op == GlobalRef(Main, :GemmWithEpilogue)
        cost = 1
    elseif op == GlobalRef(Main, :Gemm)
        cost = 10
    elseif op == GlobalRef(Main, :matmul)
        cost = 1000
    elseif op == :tuple
        cost = 0
    elseif op == GlobalRef(Base, :copyto!) || op == GlobalRef(Base, :_to_linear_index)
        println("whoop")
        cost = 1
    else
        cost = 1000
    end

    for id in args
        eclass = g[id]
        # if the child e-class has not yet been analyzed, return +Inf
        !hasdata(eclass, an) && (cost += Inf; break)
        cost += last(getdata(eclass, an))
    end

    # interesting illustration:
    # TOOD: add debug levels?
    #println(n)
    #println(cost)

    return cost
end

cost_function(n::ENodeLiteral, g::EGraph, an::Type{<:AbstractAnalysis}) = 1

const theory = addition_properties ∪ multiplication_properties ∪ addition_properties

# TODO: should we lower dot & plus to map / reduce combos?
# NOTE: as long as tensor_expr satisfies TermInterface, this will work
function simplify(tensor_expr; extra_rules=Metatheory.AbstractRule[])
    @assert tensor_expr.head == :output

    # remove :output wrapper
    type = tensor_expr.type
    tensor_expr = tensor_expr.args[1]

    # TODO: fix this, typing info is lost :(
    tensor_expr = Postwalk(Chain(canonicalize_broadcasting))(tensor_expr)
    
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
