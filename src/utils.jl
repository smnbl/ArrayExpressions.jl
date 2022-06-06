# resolve globalref to global variable (function, struct, variables, ...)
resolve(gr::GlobalRef) = getproperty(gr.mod, gr.name)
resolve(s) = s

Base.nameof(v::CC.GlobalRef) = Base.nameof(resolve(v))

iscall(expr::ArrayExpr, op) = expr.head == :call && expr.args[1] == op
iscall(expr::Expr, op) = expr.head == :call && expr.args[1] == op
iscall(expr, op) = false
