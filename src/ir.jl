using Core.Compiler: IRCode, SSAValue, PhiNode
using GPUArrays

const C = Core
const CC = C.Compiler

const InputTypes = Any

# functional representation of the array expressions as a Tree-like object
struct ArrayIR
    # expresses the operation & inputs
    op_expr::Expr
    # expresses results of type inference on this expression
    type_expr::Expr
    # TODO add tracking to original code location (for modifications?)
    #
    # location of all the input locations aka the point at which the inputs are fed to the expression
    # this is used when replacing an expression by removing all the intermediate operations that get replaced by the optimised kernel
    # these can be the location of phi nodes or GlobalRefs to the used array variables
    input_locs::Set{InputTypes}
    ArrayIR(op::Expr, type::Expr) = new(op, type, Set())
end
