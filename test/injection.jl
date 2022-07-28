using Core.Compiler: inflate_ir, OldSSAValue
using ArrayAbstractions
const AA = ArrayAbstractions
using ArrayAbstractions: replace!
# test SSA IR injections

# create SSA IR

f(A, B, C) = A * B + C

# the SSA IR of this function should be:
#=
 1 ─ %1 = (_2 * _3)::Matrix{Int64}
 │   %2 = (%1 + _4)::Matrix{Int64}
 └──      return %2
=#

ci, _ = code_typed(f, [Matrix, Matrix, Matrix], optimize=false)[1]

let ci = copy(ci)
    ir = inflate_ir(ci)

    println("before:")
    println(ir)

    todo = Dict([OldSSAValue(2) => Pair(:(Gemm(_2, _3, _4)), Matrix)])
    visited = [1,2]

    ir = replace!(ir, visited, todo, nothing)

    # clean up
    ir = CC.compact!(ir)

    println("after")
    println(ir)

    @test AA.iscall(ir.stmts.inst[1], :Gemm)
    @test AA.iscall(ir.stmts.inst[2], GlobalRef(Base, :getfield))
end

let ci = copy(ci)
    ir = inflate_ir(ci)

    println("before:")
    println(ir)


    todo = Dict([OldSSAValue(1) => Pair(:(MUL(_2, _3)), Matrix)])
    visited = [1]

    ir = replace!(ir, visited, todo, nothing)

    # clean up
    ir = CC.compact!(ir)

    println(ir)

    @test AA.iscall(ir.stmts.inst[1], :MUL)
    @test AA.iscall(ir.stmts.inst[2], GlobalRef(Base, :getfield))
    @test AA.iscall(ir.stmts.inst[3], GlobalRef(Main, :+))
end
