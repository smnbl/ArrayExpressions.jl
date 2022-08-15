using ArrayAbstractions
using ArrayAbstractions: App, Lambda, Intrinsic
using Metatheory
using CUDA
using GemmKernels

# for a fair comparison
@inline function (Base.:*)(A::CuMatrix{T}, B::CuMatrix{T}) where T
    return Gemm(A, B)
end

@inline function Gemm(A, B, C, alpha=1.0, beta=1.0)
    # return LinearAlgebra.mul!(C, A, B, 1.0, 1.0)
    return GemmWithEpilogue(A, B, C, identity, alpha, beta)
end

@inline function Gemm(A::CuMatrix{T}, B::CuMatrix{T}) where T
    # TODO: support mixed precision?
    C = CuArray{T}(undef, (size(A, 1), size(B, 2)))
    # C = 1.0 * A*B + 0.0 * C
    Gemm!(A, B, C, 1.0, 0.0)
    return C
end

@inline function Gemm!(A, B, C, alpha=1.0, beta=1.0)
    GemmWithEpilogue!(A, B, C, identity, alpha, beta)
end

# TODO:
# add prologue?
@inline function GemmWithEpilogue(A, B, C, transform, alpha=1.0, beta=1.0)
    D = similar(C)
    GemmInterface(A, B, C, D, transform, alpha=alpha, beta=beta)
    return D
end

@inline function GemmWithAdd(A, B, C, d, alpha=1.0, beta=1.0)
    transform = epi(+, d)
    D = similar(C)
    GemmInterface(A, B, C, D, transform, alpha=alpha, beta=beta)
    return D
end

@inline function GemmWithEpilogue!(A, B, C, transform, alpha=1.0, beta=1.0)
    GemmInterface(A, B, C, C, transform, alpha=alpha, beta=beta)
end

@inline function GemmBias(A, B, C, transform, bias)
    D = similar(C)
    GemmInterface(A, B, C, D, transform, alpha=alpha, beta=beta, bias=bias)
    return D
end

@inline function GemmInterface(A, B, C, D, transform; alpha=1.0, beta=1.0, bias=nothing)
    m = size(A, 1)
    k = size(A, 2)
    n = size(B, 2)

    if m != size(C, 1) || n != size(C, 2) || k != size(B, 1)
        throw(DimensionMismatch("Dimensions do not match, $(size(A)) x $(size(B)) = $(size(C))"))
    end

    println("$(size(A)) x $(size(B)) = $(size(C))")

    a_layout = GemmKernels.BLAS.global_layout(typeof(A), Val(false))
    b_layout = GemmKernels.BLAS.global_layout(typeof(B), Val(false))

    conf = GemmKernels.get_config(
            gemm_shape = (M = m, N = n, K = k),
            # TODO: gemmkernels interface changes here in latest version: .., eltype(C)}
            operator = Operator.SIMTOp,

            global_a_layout = a_layout,
            global_b_layout = b_layout,
            global_c_layout = GemmKernels.BLAS.global_layout(typeof(C), Val(false)),
            global_d_layout = GemmKernels.BLAS.global_layout(typeof(C), Val(false)),

            shared_a_layout = GemmKernels.BLAS.shared_layout_ab(typeof(A), Val(false)),
            shared_b_layout = GemmKernels.BLAS.shared_layout_ab(typeof(B), Val(false)),
            shared_c_layout = GemmKernels.BLAS.shared_layout_cd(typeof(C), Val(false)),
            shared_d_layout = GemmKernels.BLAS.shared_layout_cd(typeof(C), Val(false)),

            is_a_col_major = true,
            is_b_col_major = true
                                )
    GemmKernels.matmul(A, B, C, D, conf;
                       # TODO: prologues, bias stuff

                       transform_shared_to_regs_a = Transform.Elementwise(x -> x * alpha),
                       transform_shared_to_regs_c = Transform.Elementwise(x -> x * beta),

                       transform_regs_to_shared_d = Transform.Elementwise(transform),
                       epilogue = if (bias != nothing) GemmKernels.Epilogue.Bias(pointer(bias)) else GemmKernels.Epilogue.Default() end,
                       kernel = GemmKernels.BLAS.kernel(a_layout, b_layout)
                      )
end

struct EpiL{F}
    func::F
    c::Float32
end

(epi::EpiL)(el) = epi.func(el, epi.c)

struct EpiR{F}
    func::F
    c::Float32
end

(epi::EpiR)(el) = epi.func(epi.c, el)

function Gemm(A::EClass, B::EClass, C::EClass)
    return ArrayExpr(:call, [GlobalRef(Main, :Gemm), A, B, C], Union{})
end

function Gemm(A::EClass, B::EClass, C::EClass)
    return ArrayExpr(:call, [GlobalRef(Main, :Gemm), A, B, C, 1.0, 1.0], Union{})
end

function Gemm(A::EClass, B::EClass)
    # this will not be used tho, make it smarter?
    C = :(CuArray{T}(undef, (size(A, 1), size(B, 2))))
    return ArrayExpr(:call, [GlobalRef(Main, :Gemm), A, B, C, 1.0, 0.0], Union{})
end

function Gemm!(A::EClass, B::EClass, C::EClass)
    return ArrayExpr(:call, [GlobalRef(Main, :Gemm!), A, B, C], Union{})
end

function GemmWithEpilogue(A::EClass, B::EClass, C::EClass, epilogue)
    return ArrayExpr(:call, [GlobalRef(Main, :GemmWithEpilogue), A, B, C, epilogue], Union{})
end

function GemmWithAdd(A::EClass, B::EClass, C::EClass, d)
    return ArrayExpr(:call, [GlobalRef(Main, :GemmWithAdd), A, B, C, d], Union{})
end

function GemmWithEpilogue!(A::EClass, B::EClass, C::EClass, epilogue)
    return ArrayExpr(:call, [GlobalRef(Main, :GemmWithEpilogue!), A, B, C, epilogue], Union{})
end

function GemmKernelsBias(A::EClass, B::EClass, C::EClass, bias)
    return ArrayExpr(:call, [GlobalRef(Main, :GemmBias), A, B, C, epilogue])
end

# big rewrite rules with custom implementations
const gemm_properties = @array_theory A B C op d epi begin
    # TODO: make sure A, B, C is a matrix, and not a scalar!!
    # idea: make mul with scalar separate function? (this supports purely syntactical rewrites)
    # TODO: problem with dynamic rules like this is is that is does not work in the opposite direction

    A * B => Gemm(A, B) where (istype(A, CuMatrix) && istype(B, CuMatrix))
    A * B + C => Gemm(A, B, C) where (istype(A, CuMatrix) && istype(B, CuMatrix) && istype(C, CuMatrix))

    # merge operations in prologue / epilogue
    # TODO: how to merge with prefix? prologue?
    # TODO: make sure epilogue is scalar
    broadcast(op, Gemm(A, B, C), d) => GemmWithEpilogue(A, B, C, :(el -> op(el, $d))) where istype(d, Number)
    broadcast(op, d, Gemm(A, B, C)) => GemmWithEpilogue(A, B, C, :(el -> op($d, el))) where istype(d, Number)

    # TODO: add support for these type of rules
    # C = Gemm(A, B, C) => Gemm!(A, B, C)
    copyto!(C, Gemm(A, B, C)) == Gemm!(A, B, C)

    # fuse operations
    broadcast(op, Gemm(A, B, C)) => GemmWithEpilogue(A, B, C, op)
    broadcast(op, GemmWithEpilogue(A, B, C, epi), d) => GemmWithEpilogue(A, B, C, Lambda(:el, App(op, [App(epi, [:el]), d])))
    broadcast(op, d, GemmWithEpilogue(A, B, C, epi)) => GemmWithEpilogue(A, B, C, Lambda(:el, App(op, [d, App(epi, [:el])])))
    broadcast(op, GemmWithEpilogue(A, B, C, epi)) => GemmWithEpilogue(A, B, C, Lambda(:el, App(op, [App(epi, [:el])])))

    # TODO: add support for these type of rules
    # C = GemmWithEpilogue(A, B, C, epi) => GemmWithEpilogue!(A, B, C, epi)
    copyto!(C, GemmWithEpilogue(A, B, C, epi)) == GemmWithEpilogue!(A, B, C, epi)

    # TODO: add gemm alpha(=0) return rule (only if C is dead tho)
    # Gemm(A, B, C, 1.0, 0,0) == Gemm!(A, B, C); C
end

# TODO:
# what determines how deep an intrinsic goes?
const gpu_intrinsics = [Intrinsic(GlobalRef(Main, :*), 1, [1]),
                        Intrinsic(GlobalRef(Main, :+), 1, [1])]

# TODO: GemmKernels.matmul intrinsic

# @checked function cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B,
