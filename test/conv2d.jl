using Flux
using NNlib
using CUDA
using ArrayAbstractions
using Metatheory

const AA = ArrayAbstractions

include("compile.jl")
include("gpu/gpu_rules.jl")

function copyto_repl(T, S, cs, ct)
    println("copyto_repl")
    # convert cs to linear index
    to_lin_T = ArrayExpr(:call, [GlobalRef(Base, :_to_linear_index), T, ct...])
    # convert ct to linear index
    to_lin_S = ArrayExpr(:call, [GlobalRef(Base, :_to_linear_index), S, cs...])
    return ArrayExpr(:call, [GlobalRef(Base, :copyto!), T, to_lin_T, S, to_lin_S, 1])
end

mem_rules = @array_theory T S s1 s2 s3 s4 c1 c2 c3 c4 c5 c6 c7  begin
    Base.setindex!(T, Base.getindex(S, s1, s2, s3, s4), c1, c2, c3, c4, c5, c6, c7) => copyto_repl(T, S, (s1, s2, s3, s4), (c1, c2, c3, c4, c5, c6, c7)) where (istype(T, CuArray) && istype(S, CuArray))
end

function im2col_dims(c::ConvDims)
    return (
        # Output size
        prod(output_size(c)),
        # Size of single dotproduct within convolution
        prod(kernel_size(c))*channels_in(c),
        # One workspace per thread
        Threads.nthreads(),
    )
end

CUDA.allowscalar(true)

using NNlib: spatial_dims, input_size, kernel_size, channels_in, padding, dilation, stride, output_size, calc_padding_regions, kernel_index

function im2col!(col::AbstractArray{T,2}, x::AbstractArray{T,4}, cdims::ConvDims) where {T}
    if spatial_dims(cdims) != 3
        throw(DimensionMismatch("im2col!() only accepts 3d convoluitional inputs"))
    end

    # Extract those nice, compile-time constant type parameters from `cdims`.
    width, height, depth = input_size(cdims)
    kernel_w, kernel_h, kernel_d = kernel_size(cdims)
    C_in = channels_in(cdims)
    pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi, pad_d_lo, pad_d_hi = padding(cdims)
    dil_w, dil_h, dil_d = dilation(cdims)
    stride_w, stride_h, stride_d = stride(cdims)
    out_width, out_height, out_depth = output_size(cdims)

    # Reshape col for easy access.
    col_reshaped = reshape(col, (
        # Output resolution
        out_width,
        out_height,
        out_depth,

        # By input patch size
        kernel_w,
        kernel_h,
        kernel_d,
        C_in,
    ))

    padded_regions, central_region = calc_padding_regions(cdims)

    # A helper function to project from output (w, h) to input (input_w, input_h)
    @inline project(idx, stride, pad) = (idx - 1)*stride - pad + 1


    # We begin by copying the central region of the image which requires no padding at all.
    # Eliminating the branches of the fully generalized version below gives us a nice
    # speedup on the majority of the data.
    @inbounds for c in 1:C_in
        # Unpack "central region"
        w_region, h_region, d_region = central_region

        for kd in 1:kernel_d,
            kh in 1:kernel_h,
            kw in 1:kernel_w,
            d in d_region,
            h in h_region,
            w in w_region

            input_kd = project(d, stride_d, pad_d_lo) + (kd - 1)*dil_d
            input_kh = project(h, stride_h, pad_h_lo) + (kh - 1)*dil_h
            input_kw = project(w, stride_w, pad_w_lo) + (kw - 1)*dil_w
            kidxs = kernel_index(kw, kh, kd, cdims)

            xval::T = x[input_kw, input_kh, input_kd, c]
            col_reshaped[w, h, d, kidxs..., c] = xval
        end
    end


    # For each "padded region", we run the fully general version
    @inbounds for (w_region, h_region, d_region) in padded_regions
        for c in 1:C_in,
            d in d_region,
            h in h_region,
            w in w_region,
            kd in 1:kernel_d,
            kh in 1:kernel_h,
            kw in 1:kernel_w

            input_kd = project(d, stride_d, pad_d_lo) + (kd - 1)*dil_d
            input_kh = project(h, stride_h, pad_h_lo) + (kh - 1)*dil_h
            input_kw = project(w, stride_w, pad_w_lo) + (kw - 1)*dil_w

            kidxs = kernel_index(kw, kh, kd, cdims)

            out_of_bounds = (
                input_kd <= 0 || input_kd > depth ||
                input_kh <= 0 || input_kh > height ||
                input_kw <= 0 || input_kw > width
            )
            if out_of_bounds
                col_reshaped[w, h, d, kidxs..., c] = T(0)
                continue
            end

            # Copy the data over
            xval::T = x[input_kw, input_kh, input_kd, c]
            col_reshaped[w, h, d, kidxs..., c] = xval
        end
    end
end

function im2col_faster!(col::AbstractArray{T,2}, x::AbstractArray{T,4}, cdims::ConvDims) where {T}
    if spatial_dims(cdims) != 3
        throw(DimensionMismatch("im2col!() only accepts 3d convoluitional inputs"))
    end

    # Extract those nice, compile-time constant type parameters from `cdims`.
    width, height, depth = input_size(cdims)
    kernel_w, kernel_h, kernel_d = kernel_size(cdims)
    C_in = channels_in(cdims)
    pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi, pad_d_lo, pad_d_hi = padding(cdims)
    dil_w, dil_h, dil_d = dilation(cdims)
    stride_w, stride_h, stride_d = stride(cdims)
    out_width, out_height, out_depth = output_size(cdims)

    # Reshape col for easy access.
    col_reshaped = reshape(col, (
        # Output resolution
        out_width,
        out_height,
        out_depth,

        # By input patch size
        kernel_w,
        kernel_h,
        kernel_d,
        C_in,
    ))

    padded_regions, central_region = calc_padding_regions(cdims)

    # A helper function to project from output (w, h) to input (input_w, input_h)
    @inline project(idx, stride, pad) = (idx - 1)*stride - pad + 1

    # We begin by copying the central region of the image which requires no padding at all.
    # Eliminating the branches of the fully generalized version below gives us a nice
    # speedup on the majority of the data.
    @inbounds for c in 1:C_in
        # Unpack "central region"
        w_region, h_region, d_region = central_region

        for kd in 1:kernel_d,
            kh in 1:kernel_h,
            kw in 1:kernel_w,
            d in d_region,
            h in h_region,
            w in w_region

            input_kd = project(d, stride_d, pad_d_lo) + (kd - 1)*dil_d
            input_kh = project(h, stride_h, pad_h_lo) + (kh - 1)*dil_h
            input_kw = project(w, stride_w, pad_w_lo) + (kw - 1)*dil_w
            kidxs = kernel_index(kw, kh, kd, cdims)

            #xval::T = x_cpu[input_kw, input_kh, input_kd, c]
            #col_reshaped_cpu[w, h, d, kidxs..., c] = xval

            #sub_map[w, h, d, kidxs..., c] = input_kw, input_kh, input_kd, c

            S = x
            D = col_reshaped
            lin_S = Base._to_linear_index(S, input_kw, input_kh, input_kd, c)
            lin_T = Base._to_linear_index(D, w, h, d, kidxs..., c)
            unsafe_copyto!(D, lin_T, S, lin_S, 1)
        end
    end

    # For each "padded region", we run the fully general version
    @inbounds for (w_region, h_region, d_region) in padded_regions
        for c in 1:C_in,
            d in d_region,
            h in h_region,
            w in w_region,
            kd in 1:kernel_d,
            kh in 1:kernel_h,
            kw in 1:kernel_w

            input_kd = project(d, stride_d, pad_d_lo) + (kd - 1)*dil_d
            input_kh = project(h, stride_h, pad_h_lo) + (kh - 1)*dil_h
            input_kw = project(w, stride_w, pad_w_lo) + (kw - 1)*dil_w

            kidxs = kernel_index(kw, kh, kd, cdims)

            out_of_bounds = (
                input_kd <= 0 || input_kd > depth ||
                input_kh <= 0 || input_kh > height ||
                input_kw <= 0 || input_kw > width
            )
            if out_of_bounds
                col_reshaped[w, h, d, kidxs..., c] = T(0)
                continue
            end

            # Copy the data over
            #xval::T = x_cpu[input_kw, input_kh, input_kd, c]
            #col_reshaped_cpu[w, h, d, kidxs..., c] = xval
            
            S = x
            D = col_reshaped
            lin_S = Base._to_linear_index(S, input_kw, input_kh, input_kd, c)
            lin_T = Base._to_linear_index(D, w, h, d, kidxs..., c)
            unsafe_copyto!(D, lin_T, S, lin_S, 1)
        end
    end
end

# relies on NNlibs im2col! & col2im! implementations
# -> speed up using GPU implementations?

# based on NNlibs conv_im2col implementation
function conv_im2col!(
                y::AbstractArray{T,5}, x::AbstractArray{T,5},
                w::AbstractArray{T,5}, cdims::DenseConvDims) where {T}

    # create column (3 dimensions)
    col::AbstractArray{T,3} = similar(x, NNlib.im2col_dims(cdims))
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
    M = prod(NNlib.output_size(cdims))
    N = NNlib.channels_out(cdims)
    K = prod(NNlib.kernel_size(cdims))*NNlib.channels_in(cdims)

    # process batch by batch
    for batch_idx in 1:size(x,5)
        # M x K
        #col_slice = view(col, :, :, 1)
        col_slice = similar(x, M, K)

        view_x = view(x, :, :, :, :, batch_idx);

        # converts 3d image 'x' into a matrix 'col'
        # patches of 'x' of size (kernel_w, kernel_h, kernel_d, C_in) will be extracted and laid out along the rows of col
        # TODO: make im2col parallel version?
        @time "im2col" im2col!(col_slice, view_x, cdims)

        # need to flatten w into 2D matrix
        # K x M
        w_flat = reshape(w, size(col_slice, 2), :)
        # flatten first 3 dims of y
        y_flat = reshape(y, M, N, size(y, 5))
        # y_m = M x N
        y_m = view(y_flat, :, :, batch_idx)

        @time "mul" copyto!(y_m, alpha * col_slice * w_flat + beta*y_m)
    end
    return y
end

function conv_im2col_faster!(
                y::AbstractArray{T,5}, x::AbstractArray{T,5},
                w::AbstractArray{T,5}, cdims::DenseConvDims) where {T}

    # create column (3 dimensions)
    col::AbstractArray{T,3} = similar(x, NNlib.im2col_dims(cdims))
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
    M = prod(NNlib.output_size(cdims))
    N = NNlib.channels_out(cdims)
    K = prod(NNlib.kernel_size(cdims))*NNlib.channels_in(cdims)

    # process batch by batch
    for batch_idx in 1:size(x,5)
        # M x K
        #col_slice = view(col, :, :, 1)
        col_slice = similar(x, M, K)

        view_x = view(x, :, :, :, :, batch_idx);

        # converts 3d image 'x' into a matrix 'col'
        # patches of 'x' of size (kernel_w, kernel_h, kernel_d, C_in) will be extracted and laid out along the rows of col
        # TODO: make im2col parallel version?
        @time "im2col" im2col_faster!(col_slice, CuArray(view_x), cdims)

        # need to flatten w into 2D matrix
        # K x M
        w_flat = reshape(w, size(col_slice, 2), :)
        # flatten first 3 dims of y
        y_flat = reshape(y, M, N, size(y, 5))
        # y_m = M x N
        y_m = view(y_flat, :, :, batch_idx)

        # todo add beta
        @time "mul" copyto!(y_m, alpha * col_slice * w_flat + beta*y_m)
    end
    return y
end

function bench_im2col(x, w)
    cdims = DenseConvDims(x, w)
    M = prod(NNlib.output_size(cdims))
    N = NNlib.channels_out(cdims)
    K = prod(NNlib.kernel_size(cdims))*NNlib.channels_in(cdims)

    view_x = CuArray(view(x, :, :, :, :, 1));
    col_slice_gpu = CuArray{eltype(x)}(undef, M, K)

    view_x_cpu = Array(view_x)
    col_slice_cpu = Array(col_slice_gpu)

    println("before (cpu)")
    im2col!(col_slice_cpu, view_x_cpu, cdims)
    @time im2col!(col_slice_cpu, view_x_cpu, cdims)

    println("gpu_faster")
    im2col_faster!(col_slice_gpu, view_x, cdims)
    @time im2col_faster!(col_slice_gpu, view_x, cdims)

    println("gpu_ before")
    @time im2col!(col_slice_gpu, view_x, cdims)
end

function bench_conv(y, x, w)
    cdims = DenseConvDims(x, w)
    x_cpu = Array(x)
    w_cpu = Array(w)
    y_cpu = Array(y)

    println("before (cpu)")
    conv_im2col!(y_cpu, x_cpu, w_cpu, cdims)
    @time conv_im2col!(y_cpu, x_cpu, w_cpu, cdims)

    println("gpu_faster")
    conv_im2col_faster!(y, x, w, cdims)
    @time conv_im2col_faster!(y, x, w, cdims)

    #println("gpu_before")
    #@time conv_im2col!(y, x, w, cdims)
end

@array_opt function f(y, x, w)
    cdims = DenseConvDims(x, w)
    # First, your basic convolution with no parameters
    return @inline conv_im2col!(y, x, w, cdims)
end

@array_opt function f_faster(y, x, w)
    cdims = DenseConvDims(x, w)
    # First, your basic convolution with no parameters
    return @inline conv_im2col_faster!(y, x, w, cdims)
end

function f_nnlib(y, x, w)
    cdims = DenseConvDims(x, w)
    # First, your basic convolution with no parameters
    return NNlib.conv_im2col!(y, x, w, cdims)
end

# test implementation against NNlib
# x: 5 dimensions: w, h, d (rgb), nr_input_channels?, nr_batches
# w: 5 dimensions: w, h, d,     , nr_input_channels, nr_output_channels (for each output channel different kernel)
# y: 5 dimensions: w, h, d,     , nr_output_channels, nr_batches 
# checkif these values are typical
batches = 1
x = CuArray(rand(Float32, 100, 100, 3, 2, batches))
w = CuArray(rand(Float32, 10, 10, 3, 2, 4))
cdims = DenseConvDims(x, w)
y = CuArray(rand(Float32, NNlib.output_size(cdims)..., 4, batches))


println("comparing im2col! performance")
bench_im2col(x, w)
bench_conv(y, x, w)

args = [y, x, w]
argtype = Tuple{typeof.(args)...}

# todo: add gemm rules
rules = AA.canonicalize_broadcasting ∪ mem_rules

# ci = compile_expression(f, argtype, [:x, :w], eltype=CuArray, extra_rules=rules)
# @generated f_opt(x, w) = ci
@eval f_opt(y, x, w) = $(compile_with_gpucompiler(f, argtype, [:y, :x, :w], extra_rules=rules))

println("compiling done")
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
