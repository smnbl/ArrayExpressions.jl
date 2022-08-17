using ArrayAbstractions: ArrayExpr, Input
const AA = ArrayAbstractions
using Metatheory

Gemm(A, B, C) = ()

# test inrules instruction matcher
#= broken (for now)
inst4 = Expr(:invoke, nothing, Input(GlobalRef(ArrayAbstractions, :+), typeof(ArrayAbstractions.:+)), :A, :B)
inst3 = Expr(:invoke, nothing, Input(GlobalRef(Metatheory, :*), typeof(ArrayAbstractions.:*)), :A, :B)
inst5 = Expr(:call, Input(GlobalRef(ArrayAbstractions, :*), typeof(ArrayAbstractions.:*)), :A, :B)

rules = @array_theory A B C begin
    A * B == B * A
end

@test AA.inrules(inst3, rules) == true
@test AA.inrules(inst4, rules) == false
@test AA.inrules(inst5, rules) == true
=#

# test array rules
let l = @array_rule Main.println(11) --> 10
    e = Expr(:call, Input(GlobalRef(Main, :println)), 11)
    @test l(e) === 10

    e = Expr(:call, GlobalRef(Main, :println), -1)
    @test l(e) === nothing
end

let l2 = @array_rule Base.Broadcast.broadcasted(~op, ~A, ~B) --> broadcast(~op, ~A, ~B)
    e = Expr(:call, Base.Broadcast.broadcasted, :op, :A, :B)
    @test l2(e) == Expr(:call, broadcast, :op, :A, :B)
end

let gemm = @array_rule (~A * ~B) + ~C --> Gemm(~A, ~B, ~C)
    e = Expr(:call, +, Expr(:call, *, :A, :B), :C)
    @test gemm(e) == Expr(:call, Main.Gemm, :A, :B, :C)
end

let gemm = @array_rule Base.broadcasted(+, ~A * ~B, ~C) --> Gemm(~A, ~B, ~C)
    e = ArrayExpr(:call, [Input(GlobalRef(Base, :broadcasted)), Input(GlobalRef(Main, :+)), ArrayExpr(:call, [Input(GlobalRef(Main, :*)), :A, :B]), :C])
    @test gemm(e) == ArrayExpr(:call, [Main.Gemm, :A, :B, :C])
end

let gemm = @array_rule Base.broadcast(+, ~A * ~B, ~C) --> Gemm(~A, ~B, ~C)
    e = ArrayExpr(:call, [Base.broadcast, Input(GlobalRef(Main, :+), Core.Const(+)), ArrayExpr(:call, [Input(GlobalRef(Main, :*), Core.Const(*)), :A, :B]), :C])
    @test gemm(e) == ArrayExpr(:call, [Main.Gemm, :A, :B, :C])
end

# test equivalences of ENodeTerms
# TODO: still not enough; compare EClasses?
let a = Metatheory.ENodeLiteral(Input(GlobalRef(Main, :+), Core.Const(+))), b = Metatheory.ENodeLiteral(GlobalRef(Base, :+))
    @test isequal(a, b)
end

# test input unwrapping
input = Input(GlobalRef(Main, :println), typeof(Main.println))
@test isequal(input, GlobalRef(Main, :println))
@test isequal(GlobalRef(Main, :println), input)

let l = @array_rule Main.println(11) --> 10
    e = ArrayExpr(:call, [Input(GlobalRef(Main, :println)), 11], Union{})
    @test l(e) == 10
end

let t = @array_theory A begin
    Main.println(A) --> A
end
    e = ArrayExpr(:call, [Input(GlobalRef(Main, :println)), 11], Union{})
    @test Metatheory.Chain(t)(e) == 11
end

# global ref resolver
@test eval(ArrayAbstractions.resolveref(:(Base.Broadcast.broadcast), Main)) == Base.Broadcast.broadcast

# multi-rewrite rules
let multi = @array_rule (print(~A), print(~B)) --> print((~A, ~B))
    e = ArrayExpr(:call, [tuple, ArrayExpr(:call, [print, :A]), ArrayExpr(:call, [print, :B])])

    rew = Metatheory.Rewriters.Postwalk(Metatheory.Rewriters.PassThrough(multi))

    @test rew(e) == ArrayExpr(:call, [print, ArrayExpr(:tuple, [:A, :B])])
end


matmul(a, b) = println("matmul")
split(a, b) = println("split")
concat(a, b) = println("concat")

let multi = @array_rule (matmul(~x, ~w1), matmul(~x, ~w2)) --> split(matmul(~x, concat(~w1, ~w2)))
    e = ArrayExpr(:call, [tuple, ArrayExpr(:call, [Input(GlobalRef(Main, :matmul)), :x, :w1]), ArrayExpr(:call, [Input(GlobalRef(Main, :matmul)), :x, :w2])])

    rew = Metatheory.Rewriters.Postwalk(Metatheory.Rewriters.PassThrough(multi))

    @test rew(e) == ArrayExpr(:call, [split, ArrayExpr(:call, [matmul, :x, ArrayExpr(:call, [concat, :w1, :w2])])])
end

# test fuzzy rule match
let l = @array_rule ~A * ~B --> Gemm(~A, ~B)
    e = ArrayExpr(:call, [Input(GlobalRef(ArrayAbstractions, :*)), :A, :B])
    @test l(e) == ArrayExpr(:call, [Gemm, :A, :B])
end

# test input comparisons & hashing
let x = Input(GlobalRef(Main, :+)), y = Input(GlobalRef(Base, :+))
    @test x == y
    @test hash(x) == hash(y)
end
