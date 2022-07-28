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

    todo = Dict([OldSSAValue(2) => tuple(:(Gemm(_2, _3, _4)), Matrix, [OldSSAValue(2)])])
    visited = [1,2]

    ir = replace!(ir, visited, todo, nothing)

    # clean up
    ir = CC.compact!(ir)
    
    #=
     result: 
     1 ─ %1 = Gemm(_2, _3, _4)::Matrix
     │   %2 = Base.getfield(%1, 1)::Any
     └──      return %2
    =#

    println(ir)

    #@test AA.iscall(ir.stmts.inst[1], :Gemm)
    #@test AA.iscall(ir.stmts.inst[2], GlobalRef(Base, :getfield))
end

# partial replacement
let ci = copy(ci)
    ir = inflate_ir(ci)

    todo = Dict([OldSSAValue(1) => tuple(:(MUL(_2, _3)), Matrix, [OldSSAValue(1)])])
    visited = [1]

    ir = replace!(ir, visited, todo, nothing)

    # clean up
    ir = CC.compact!(ir)

    #=
     result:
     1 ─ %1 = MUL(_2, _3)::Matrix
     │   %2 = Base.getfield(%1, 1)::Any
     │   %3 = (%2 + _4)::Any
     └──      return %3
    =#

    println(ir)

    #@test AA.iscall(ir.stmts.inst[1], :MUL)
    #@test AA.iscall(ir.stmts.inst[2], GlobalRef(Base, :getfield))
    #@test AA.iscall(ir.stmts.inst[3], GlobalRef(Main, :+))
end


# multi-output
multi(A, B, C) = A * B + C, A * B + C
ci, _ = code_typed(multi, [Matrix, Matrix, Matrix], optimize=false)[1]

# the SSA IR of this function should be:
#=
 1 ─ %1 = (_2 * _3)::Any                 
 │   %2 = (%1 + _4)::Any                  
 │   %3 = (_2 * _3)::Any                   
 │   %4 = (%3 + _4)::Any                    
 │   %5 = Core.tuple(%2, %4)::Tuple{Any, Any}
 └──      return %5
=#

let ci = copy(ci)
    ir = inflate_ir(ci)

    println(ir)

    todo = Dict([OldSSAValue(2) => tuple(:(GEMM_MULTI(_2, _3, _4)), Matrix, [OldSSAValue(2), OldSSAValue(4)])])
    visited = [1,2,3,4]

    ir = replace!(ir, visited, todo, nothing)

    println(ir)

    # clean up
    ir = CC.compact!(ir)

    println(ir)

    #=
     result:
     1 ─ %1 = MUL(_2, _3)::Matrix
     │   %2 = Base.getfield(%1, 1)::Any
     │   %3 = (%2 + _4)::Any
     └──      return %3
    =#

    #@test AA.iscall(ir.stmts.inst[1], :GEMM_MULTI)
    #@test AA.iscall(ir.stmts.inst[2], GlobalRef(Base, :getfield))
    #@test AA.iscall(ir.stmts.inst[3], GlobalRef(Base, :getfield))
    #@test AA.iscall(ir.stmts.inst[4], GlobalRef(Core, :tuple))
end

# test latest_ref
multi(A, B, C) = A * B + C, A * B + C
ci, _ = code_typed(multi, [Matrix, Matrix, Matrix], optimize=false)[1]

# the SSA IR of this function should be:
#=
 1 ─ %1 = (_2 * _3)::Any                 
 │   %2 = (%1 + _4)::Any                  
 │   %3 = (_2 * _3)::Any                   
 │   %4 = (%3 + _4)::Any                    
 │   %5 = Core.tuple(%2, %4)::Tuple{Any, Any}
 └──      return %5
=#

let ci = copy(ci)
    ir = inflate_ir(ci)

end
