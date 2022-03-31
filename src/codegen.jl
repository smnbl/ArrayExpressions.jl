# lower expression to :-> block
function codegen(arrexpr, inputs)
    #=
    args = Array{Any}(undef, length(inputs))

    # TODO: this will make mistakes when arguments are unused;
    # idea: get nr of arguments from mi
    for i in inputs
        if(i isa Core.Argument)
            args[i.n - 1] = Symbol("var_$(i.n)")
        end
    end
    =#
    args = []
    expr = Expr(:->, Expr(:tuple, args...), Expr(:block, _codegen(arrexpr)))
end

function _codegen(arrexpr)
    if arrexpr isa ArrayExpr
        for (ind, arg) in enumerate(arrexpr.args)
            arrexpr.args[ind] = _codegen(arg)
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
