using Core.Compiler
const CC = Core.Compiler

# resolve globalref to global variable (function, struct, variables, ...)
function resolve(gr::GlobalRef)
    try
        getproperty(gr.mod, gr.name)
    catch e
        e isa UndefVarError ? nothing : rethrow(e)
    end
end

resolve(s) = s

Base.nameof(v::CC.GlobalRef) = Base.nameof(resolve(v))

iscall(expr::ArrayExpr, op) = expr.head == :call && expr.args[1] == op
iscall(expr::Expr, op) = expr.head == :call && expr.args[1] == op
iscall(expr, op) = false
