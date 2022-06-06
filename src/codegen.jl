using Core.Compiler
const CC = Core.Compiler

# lower expression to :-> block
function codegen_expr!(arrexpr, nargs)
    block, input_map = _codegen_expr(arrexpr)

    args = [Symbol("var_$i") for i in 1:length(input_map)]

    Expr(:->, Expr(:tuple, args...), Expr(:block, block)), input_map
end

function _codegen_expr(arrexpr, input_map = Dict{Any, Int}())
    if arrexpr isa ArrayExpr
        for (ind, arg) in enumerate(arrexpr.args)
            arrexpr.args[ind], input_map = _codegen_expr(arg, input_map)
        end

        if (arrexpr.head == :app)
            return Expr(:call, arrexpr.args...), input_map
        end

        #=
        # wrap closure expressions inside opaque closures
        if (arrexpr.head == :->)
            return Expr(:opaque_closure, convert(Expr, arrexpr)), input_map
        end
        =#

        return convert(Expr, arrexpr), input_map
    elseif arrexpr isa Input
        if (isprimitivetype(typeof(arrexpr.val)) || arrexpr.val isa Core.GlobalRef) 
            return arrexpr.val, input_map
        else
            input = get!(input_map, arrexpr.val, length(input_map) + 1)
            return Symbol("var_$(input)"), input_map
        end
    else
        return arrexpr, input_map
    end
end

# reconstructs SSA IR from arrexpr
# for use with the OC SSA IR interface: (https://github.com/JuliaLang/julia/pull/44197)
#= OLD CODE
function codegen_ssa!(arrexpr)
    # TODO insert return node!
    # linearize
    stmts = reverse(linearize(arrexpr))

    # replace relative SSAValues with absolute ones
    for (stmt_loc, stmt) in enumerate(stmts)
        if stmt isa ArrayExpr
            for (arg_loc, arg) in enumerate(stmt.args)
                if arg isa TempSSAValue
                    stmt.args[arg_loc] = TempSSAValue(stmt_loc - arg.id)
                elseif arg isa ArrayExpr # input expressions
                    stmt.args[arg_loc] = arg.args[2]
                end
            end
        end
    end

    return stmts
end

struct TempSSAValue
    id::Int64
end

function linearize(arrexpr)
    if arrexpr isa Union{ArrayExpr, Input}
        stmts = Any[]

        for (ind, arg) in enumerate(arrexpr.args)
            if arg isa ArrayExpr
                arrexpr.args[ind] = TempSSAValue(length(stmts) + 1)
                append!(stmts, linearize(arg))
            elseif arg isa Input && arg.val isa SSAValue
                # remove input label from SSAValue arguments
                arrexpr.args[ind] == arg.val
            end
        end

        return pushfirst!(stmts, arrexpr)
    else
        throw("please only call on ArrayExpr objects!")
    end
end
=#
