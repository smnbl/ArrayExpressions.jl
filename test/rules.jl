using ArrayAbstractions: ArrayExpr, Input
const AA = ArrayAbstractions
using Metatheory
using Metatheory: Chain

# test inrules instruction matcher
inst = Expr(:call, :test, :A, :B)
inst2 = Expr(:call, :test2, :A, :B)
inst3 = Expr(:call, :test, :A)
inst4 = Expr(:call, :test, :A, :B, :C)

rules = @theory A B C begin
    test(A, B) == blabla(A, B)
    test(A, B, C) == bla(A, B, C)
end

@test AA.inrules(inst, rules) == true
@test AA.inrules(inst2, rules) == false
@test AA.inrules(inst2, rules) == false
@test AA.inrules(inst4, rules) == true

# test array rules
let l = @array_rule Main.println(11) --> 10
    e = Expr(:call, GlobalRef(Main, :println), 11)
    @test l(e) === 10

    e = Expr(:call, GlobalRef(Main, :println), -1)
    @test l(e) === nothing
end

let l2 = @array_rule Base.Broadcast.broadcasted(~op, ~A, ~B) --> broadcast(~op, ~A, ~B)
    e = Expr(:call, GlobalRef(Base.Broadcast, :broadcasted), :op, :A, :B)
    @test l2(e) == Expr(:call, GlobalRef(Main, :broadcast), :op, :A, :B)
end

let gemm = @array_rule (~A * ~B) + ~C --> Gemm(~A, ~B, ~C)
    e = Expr(:call, GlobalRef(Main, :+), Expr(:call, GlobalRef(Main, :*), :A, :B), :C)
    @test gemm(e) == Expr(:call, GlobalRef(Main, :Gemm), :A, :B, :C)
end

# test input unwrapping
input = Input(GlobalRef(Main, :println))
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
    @test Chain(t)(e) == 11
end

# global ref resolver
@test eval(ArrayAbstractions.resolveref(:(Base.Broadcast.broadcast), Main)) == Base.Broadcast.broadcast

# multi-rewrite rules
let multi = @array_rule (print(~A), print(~B)) --> print((~A, ~B))
    e = :((print(:A), print(:B)))
    e = ArrayExpr(:tuple, [ArrayExpr(:call, [GlobalRef(Main, :print), :A]), ArrayExpr(:call, [GlobalRef(Main, :print), :B])])
    @test multi(e) == ArrayExpr(:call, [GlobalRef(Main, :print), ArrayExpr(:tuple, [:A, :B])])
end


let multi = @array_rule (matmul(~x, ~w1), matmul(~x, ~w2)) --> split(matmul(~x, concat(~w1, ~w2)))
    e = ArrayExpr(:tuple, [ArrayExpr(:call, [Input(GlobalRef(Main, :matmul)), :x, :w1]), ArrayExpr(:call, [Input(GlobalRef(Main, :matmul)), :x, :w2])])
    @test multi(e) == ArrayExpr(:call, [GlobalRef(Main, :split), ArrayExpr(:call, [GlobalRef(Main, :matmul), :x, ArrayExpr(:call, [GlobalRef(Main, :concat), :w1, :w2])])])
end