using Flux
using NNlib
using CUDA
using ArrayAbstractions

const AA = ArrayAbstractions

include("compile.jl")
include("gpu/gpu_rules.jl")

# checkif these values are typical
x = rand(Float32, 1000, 3, 1000)
w = rand(Float32, 10, 3, 10)
y = rand(Float32, 1000, 3, 1000)

# based on NNlibs conv_im2col implementation
function conv_im2col!(
                y::AbstractArray{T,3}, x::AbstractArray{T,3},
                w::AbstractArray{T,3}, cdims::DenseConvDims) where {T}

    col::AbstractArray{T,1} = similar(x, NNlib.im2col_dims(cdims))
    alpha::T=T(1)
    beta::T=T(0)
    NNlib.check_dims(size(x), size(w), size(y), cdims)

    #   COL   *    W    ->    Y
    # [M x K] * [K x N] -> [M x N]
    #
    #  M: output spatial resolution
    #  N: output channels
    #  K: size of input "patch" (kernel size and input channels combined)
    #
    # In english, we're grabbing each input patch and laying them out along
    # the M dimension in `col`, so that the GEMM call below multiplies each
    # kernel (which is kernel_h * kernel_w * channels_in elments long) is
    # dotproducted with that input patch, effectively computing a convolution
    # in a somewhat memory-wasteful but easily-computed way (since we already
    # have an extremely highly-optimized GEMM call available in BLAS).
    M = prod(output_size(cdims))
    N = channels_out(cdims)
    K = prod(kernel_size(cdims))*channels_in(cdims)

    for batch_idx in 1:size(x,5)
        # col_slice is a thread-local workspace
        col_slice = view(col, :, :, threadid())

        im2col!(col_slice, view(x, :, :, :, :, batch_idx), cdims)
        y_slice = view(y, col:col_end,row:row_end)

        col = (batch_idx - 1)*M*N % size(y, 1) + 1 
        col_end = (batch_idx)*M*N % size(y, 1) + 1
        row = (batch_idx - 1)*M*N ÷ size(y, 1) + 1
        row_end = (batch_idx)*M*N ÷ size(y, 1) + 1


        copyto!(y_slice, alpha * col_slice * w + beta*y_slice)
    end
    return y
end


function f(y, x, w)
    cdims = DenseConvDims(x, w)
    # First, your basic convolution with no parameters
    return conv_im2col!(y, x, w, cdims)
end

args = [y, x, w]
argtype = Tuple{typeof.(args)...}

rules = AA.canonicalize_broadcasting ∪ gemm_properties

# ci = compile_expression(f, argtype, [:x, :w], eltype=CuArray, extra_rules=rules)
# @generated f_opt(x, w) = ci
@eval f_opt(y, x, w) = $(compile_with_gpucompiler(f, argtype, [:y, :x, :w], extra_rules=rules))

f_opt(y, x, w)

println("testing layer_opt")
println(isapprox(Array(f(y, x, w)), Array(f_opt(y, x, w)), rtol=1.0, nans=true))

println("flux before:")
@time CUDA.@sync begin
    f(y, x, w)
end

println("flux after:")
@time CUDA.@sync begin
    f_opt(y, x, w)
end

println(":)")
