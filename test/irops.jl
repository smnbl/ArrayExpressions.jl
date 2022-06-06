# tests all kinds of ir operations
function f(A, B)
    C = A
    D = A * B
    return C + D
end

ci, _ = code_typed(f, (Matrix, Matrix))[1]
ir = CC.inflate_ir(ci)

pass = AA.ArrOptimPass(AbstractArray)

ir = pass(ir, Main)

# test lambdas
println(ir)


# test invoke compare

@test compare("matmul", "#matmul#6")
