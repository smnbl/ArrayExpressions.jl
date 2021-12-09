export optimize

# TODO switch to type-inferred analysis
# -> need type propagation

function optimize(@nospecialize(f::Function), tt::Tuple)
    # TODO what if code_typed returns multiple ci's?
    ci = Base.code_lowered(f, tt; debuginfo=:default)[1]
    @info "before"
    println(ci)
    @info "after"
    ci_after = transform_gemm(ci)
    println(ci_after)
    ci_after
end

# resolve globalref to global variable (function, struct, variables, ...)
resolve(gr::GlobalRef) = getproperty(gr.mod, gr.name)

# (A * B) + C -> (A * B + C)
function transform_gemm(ci)
    code_new = deepcopy(ci.code)
    for (idx, stmt) in enumerate(ci.code)
        stmt isa Expr || continue
        if stmt.head == :(=)
            stmt = stmt.args[2]
        end

        stmt isa Expr || continue

        stmt.head == :call || continue
        resolve(stmt.args[1]) == ArrayPlus || continue

        loc_dot = stmt.args[2].id

        lhs = ci.code[loc_dot]
        lhs isa Expr || continue
        lhs.head == :call || continue
        resolve(lhs.args[1]) == ArrayDot || continue

        print(true)
        (A = lhs.args[2]) isa GlobalRef || continue
        (B = lhs.args[3]) isa GlobalRef || continue
        (C = stmt.args[3]) isa GlobalRef || continue

        # TODO: make sure ArrayDot is only used as argument for ArrayMul
        # -> needs use-def info -> look at compiler passes
        code_new[loc_dot] = Expr(:call, :+, 0) # NOP ArrayDot
        code_new[idx] = Expr(:call, ArrayGemm, A, B, C) # ArrayDot -> ArrayMul
    end

    # TODO look at all stuff we have to update
    # use insertion types as in ir.jl passes to collect and perform insertions in 1 go?
    ci.code = code_new
    ci
end
