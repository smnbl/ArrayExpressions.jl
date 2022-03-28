function codegen(arrexpr)
    if arrexpr isa ArrayExpr
        for (ind, arg) in enumerate(arrexpr.args)
            arrexpr.args[ind] = codegen(arg)
        end

        if (arrexpr.head == :app)
            return Expr(:call, arrexpr.args...)
        elseif (arrexpr.head == :input)
            return arrexpr.args[1]
        end

        return convert(Expr, arrexpr)
    else
        return arrexpr
    end
end
