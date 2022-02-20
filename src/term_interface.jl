using TermInterface
# define interface between Metatheory.jl & ArrayAbstractions expression trees

# inspired by https://github.com/QuantumBFS/YaoEGraph.jl/blob/main/src/term_interface.jl
# which defines interface between Yoa & Metatheory
const T = TermInterface

T.istree(x::Type{ArrayIR}) = true

# should always be :call for now!!
T.exprhead(i::ArrayIR) = :call

# the operation is the first argument of :call
T.operation(i::ArrayIR) = i.op_expr.args[1]
T.arguments(i::ArrayIR) = i.op_expr.args[2:end]

# TODO how do we work with symtypes?
# T.symtype(x::Type{ArrayIR}) = eval(i.type_expr.head)

function T.similarterm(x::ArrayIR, head, args, symtype=nothing; metadata=nothing, exprhead=:call)
    op_expr = Expr(:call, head, args...)
    ArrayIR(x.op_expr, x.type_expr)
end
