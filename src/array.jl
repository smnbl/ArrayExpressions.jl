# define a stub array type
# TODO
# mark IR as intrinsic -> how do we do this?
#
struct StubArray{S <: Tuple, T, N} <: AbstractArray{T, N}
end

@inline @generated function StubArray{S, T}(args...) where {S, T}
    dims = (S.parameters...,)
    N = length(dims)
    quote
        StubArray{S, T, $N}(args...)
    end
end

# array interface
Base.IndexStyle(::Type{<:StubArray}) = IndexCartesian()
Base.size(x::StubArray{S}) where {S} = (S.parameters...,)
Base.@propagate_inbounds Base.getindex(v::StubArray{S, T, N}, y::Int, x::Int) where {S, T, N} = 0

# map operations to IR
function Base.:+(A::StubArray, B::StubArray)
    ArrayPlus(A, B)
end

function Base.:*(A::StubArray, B::StubArray)
    ArrayDot(A, B)
end
