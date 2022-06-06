struct Intrinsic
    operation::GlobalRef
    outputs::Vector{Int}
    inputs::Vector{Int}
end

struct IntrinsicInstance
    location::Int
    intrinsic::Intrinsic
end

Base.isless(intr::IntrinsicInstance, x) = intr.location < x
Base.isless(x, intr::IntrinsicInstance) = intr.location < x
# TODO: other ordering?
