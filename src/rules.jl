using Metatheory

ArrayGemm(A, B, C) = error("should still be linked with GemmKernels.jl")

# Define Metatheory rules
const custom_kernel_rules = @theory A B C begin
    # GEMM;
    # TODO how to merge with prefix? prologue?
    (+)((*)(A, B), C) --> ArrayGemm(A, B, C)

    # TODO: add map(reduce()) -> mapreduce() rule & other rules
end

# TODO: should we lower dot & plus to map / reduce combos?
function simplify(ir::ArrayIR)
    g = EGraph(ir.op_expr)
    
    # saturate graph
    report = saturate!(g, custom_kernel_rules)

    # TODO: replace with own cost function
    # astsize: cost function that favors smaller expressions in the equivalence classes
    ex = extract!(g, astsize)
    return ex
end

# TODO: typechecking using EGraph Analyses?
