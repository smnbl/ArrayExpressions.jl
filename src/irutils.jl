# resolve globalref to global variable (function, struct, variables, ...)
resolve(gr::GlobalRef) = getproperty(gr.mod, gr.name)
resolve(s::CC.SSAValue) = s

Base.nameof(v::CC.GlobalRef) = Base.nameof(resolve(v))

