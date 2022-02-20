using Metatheory

# Define Metatheory rules
const custom_kernel_rules = @theory A B C begin
    # GEMM
    (+)((*)(A, B), C) --> ArrayGemm(A, B, C)
    # TODO: add map(reduce()) -> mapreduce() rule
end

# TODO: should we lower dot & plus to map / reduce combos?
function simplify(ir::ArrayIR)
    # TODO: implement using equality saturation stuff
    return Metatheory.PassThrough(custom_kernel_rules[1])(ir.op_expr)
end
