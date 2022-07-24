struct Intrinsic
    pattern::GlobalRef
    operation_pos::Int # position in inst.args that matches with the actual operation
    #argtypes::Vector{Type}
    outputs::Vector{Int}
end

struct IntrinsicInstance
    location::Int
    args_offset::Int
    intrinsic::Intrinsic
end

Base.isless(intr::IntrinsicInstance, x) = intr.location < x
Base.isless(x, intr::IntrinsicInstance) = intr.location < x
# TODO: other ordering?

function getargs(ir::IRCode, instance::IntrinsicInstance)
    args = []
    inst = ir.stmts.inst[instance.location]
    for i in 1:length(inst.args)
        if (i - instance.args_offset) ∉ instance.intrinsic.outputs
            push!(args, inst.args[i])
        else
            # wrap in output tag
            push!(args, Output(inst.args[i]))
        end
    end
    return args
end

# TODO: use GPUCompiler's isintrinsic?
