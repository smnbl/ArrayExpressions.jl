using Core.Compiler: IRCode, SSAValue
using GPUArrays

const C = Core
const CC = C.Compiler

# functional representation of the array expressions as a Tree-like object
struct ArrayIR
    # expresses the operation & inputs
    op_expr::Expr
    # expresses results of type inference on this expression
    type_expr::Expr
    # TODO add tracking to original code location (for modifications?)
end
