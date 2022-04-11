using Core.Compiler
const CC = Core.Compiler

# lower expression to :-> block
function codegen_expr!(arrexpr, nargs)
    args = Array{Any}(undef, nargs)

    # TODO: this will make mistakes when arguments are unused;
    for i in 1:nargs
        args[i] = Symbol("var_$(i)")
    end

    expr = Expr(:->, Expr(:tuple, args...), Expr(:block, _codegen_expr(arrexpr)))
end

function _codegen_expr(arrexpr)
    if arrexpr isa ArrayExpr
        for (ind, arg) in enumerate(arrexpr.args)
            arrexpr.args[ind] = _codegen_expr(arg)
        end

        if (arrexpr.head == :app)
            return Expr(:call, arrexpr.args...)
        elseif (arrexpr.head == :call && arrexpr.args[1] == :input)
            if arrexpr.args[2] isa Core.Argument
                return Symbol("var_$(arrexpr.args[2].n)")
            elseif arrexpr.args[2] isa GlobalRef
                return Symbol(arrexpr.args[2])
            else
                # literals
                return arrexpr.args[2]
            end
        end

        return convert(Expr, arrexpr)
    else
        return arrexpr
    end
end

# reconstructs SSA IR from arrexpr
# for use with the OC SSA IR interface: (https://github.com/JuliaLang/julia/pull/44197)
function codegen_ssa!(arrexpr)
    # TODO insert return node!
    # linearize
    stmts = reverse(linearize(arrexpr))

    # replace relative SSAValues with absolute ones
    for (stmt_loc, stmt) in enumerate(stmts)
        if stmt isa Expr
            for (arg_loc, arg) in enumerate(stmt.args)
                if arg isa SSAValue
                    stmt.args[arg_loc] = Core.SSAValue(stmt_loc - arg.id)
                elseif arg isa ArrayExpr # input annotations
                    stmt.args[arg_loc] = arg.args[2]
                end
            end
        end
    end

    return stmts
end

function linearize(arrexpr)
    if arrexpr isa ArrayExpr
        stmts = Any[]

        for (ind, arg) in enumerate(arrexpr.args)
            if arg isa ArrayExpr && !(arg.head == :call && arg.args[1] == :input)
                arrexpr.args[ind] = SSAValue(length(stmts) + 1)
                append!(stmts, linearize(arg))
            end
        end

        if arrexpr.head == :ϕ
            # PROBLEM
        elseif arrexpr.head == :->
            # TODO
        else
            return pushfirst!(stmts, Expr(arrexpr.head, arrexpr.args...))
        end
    else
        throw("please only call on ArrayExpr objects!")
    end
end
