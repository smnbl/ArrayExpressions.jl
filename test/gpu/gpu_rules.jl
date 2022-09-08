using ArrayAbstractions
using ArrayAbstractions: App, Lambda, Intrinsic
using Metatheory
using CUDA
using GemmKernels

# for a fair comparison
@inline function (Base.:*)(A::CuMatrix{T}, B::CuMatrix{T}) where T
    return Gemm(A, B)
end

@inline function Gemm(A::CuMatrix{T}, B::CuMatrix{T}, C::CuArray{T}, alpha::T=T(1.0), beta::T=T(1.0)) where T
    # return LinearAlgebra.mul!(C, A, B, 1.0, 1.0)
    return GemmWithTransform(A, B, C, identity, alpha, beta)
end

@inline function Gemm(A::CuMatrix{T}, B::CuMatrix{T}) where T
    # TODO: support mixed precision?
    C = CuArray{T}(undef, (size(A, 1), size(B, 2)))
    # C = 1.0 * A*B + 0.0 * C
    Gemm!(C, A, B, C, T(1.0), T(0.0))
end

@inline function Gemm!(D, A::CuMatrix{T}, B::CuMatrix{T}, C::CuArray{T}, alpha=T(1.0), beta=T(1.0)) where T
    GemmWithTransform!(D, A, B, C, identity, alpha, beta)
end

# TODO:
# add prologue?
@inline function GemmWithTransform(A::CuArray{T}, B::CuArray{T}, C::CuArray{T}, transform, alpha=T(1.0), beta=T(1.0)) where T
    D = similar(C, size(A, 1), size(B, 2))
    GemmInterface(A, B, C, D, transform, alpha=alpha, beta=beta)
end

@inline function GemmWithAdd(A, B, C, d, alpha=1.0, beta=1.0)
    transform = epi(+, d)
    D = similar(C, size(A, 1), size(B, 2))
    GemmInterface(A, B, C, D, transform, alpha=alpha, beta=beta)
end

@inline function GemmWithTransform!(D::CuMatrix{T}, A::CuMatrix{T}, B::CuMatrix{T}, C::CuArray{T}, transform, alpha=T(1.0), beta=T(1.0)) where T
    GemmInterface(A, B, C, D, transform, alpha=alpha, beta=beta)
end

@inline function GemmBias(A, B, C, transform, bias)
    D = similar(C, size(A, 1), size(B, 2))
    GemmInterface(A, B, C, D, transform, alpha=alpha, beta=beta, bias=bias)
end

@inline function GemmInterface(A::CuArray{T}, B::CuArray{T}, C::CuArray{T, d}, D::CuArray{T}, transform; alpha::T=T(1.0), beta::T=T(1.0), bias=nothing) where {T, d}
    m = size(A, 1)
    k = size(A, 2)
    n = size(B, 2)

    @assert m === size(D, 1)
    @assert n === size(D, 2)
    @assert m === size(C, 1)

    a_layout = GemmKernels.BLAS.global_layout(typeof(A), Val(false))
    b_layout = GemmKernels.BLAS.global_layout(typeof(B), Val(false))

    is_broadcast_c = d === 1

    conf = GemmKernels.get_config(
            gemm_shape = (M = m, N = n, K = k),
            # TODO: gemmkernels interface changes here in latest version: .., eltype(C)}
            operator = Operator.WMMAOp{16, 16, 16, eltype(C)},

            global_a_layout = a_layout,
            global_b_layout = b_layout,
            global_c_layout = GemmKernels.BLAS.global_layout(typeof(C), Val(false)),
            global_d_layout = GemmKernels.BLAS.global_layout(typeof(C), Val(false)),

            shared_a_layout = GemmKernels.BLAS.shared_layout_ab(typeof(A), Val(false)),
            shared_b_layout = GemmKernels.BLAS.shared_layout_ab(typeof(B), Val(false)),
            shared_c_layout = GemmKernels.BLAS.shared_layout_cd(typeof(D), Val(false)),
            shared_d_layout = GemmKernels.BLAS.shared_layout_cd(typeof(D), Val(false)),

            is_a_col_major = true,
            is_b_col_major = true,
            is_broadcast_c = is_broadcast_c,
                                )
    GemmKernels.matmul(A, B, C, D, conf;
                # TODO: prologues, bias stuff

                transform_shared_to_regs_a = Transform.Elementwise(x -> x * alpha),
                transform_shared_to_regs_c = Transform.Elementwise(x -> x * beta),

                transform_regs_to_shared_d = Transform.Elementwise(transform),
                epilogue = if (bias != nothing) GemmKernels.Epilogue.Bias(pointer(bias)) else GemmKernels.Epilogue.Default() end,
                kernel = GemmKernels.Kernel.matmul_singlestage
                )
    return D
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
    return ArrayExpr(:call, [Gemm, A, B, C])
end

function Gemm(A::EClass, B::EClass)
    # this will not be used tho, make it smarter?
    C = :(CuArray{T}(undef, (size(A, 1), size(B, 2))))
    return ArrayExpr(:call, [Gemm, A, B, C])
end

function Gemm!(D::EClass, A::EClass, B::EClass, C::EClass)
    return ArrayExpr(:call, [Gemm!, D, A, B, C], Union{})
end

function GemmWithTransform(A::EClass, B::EClass, C::EClass, epilogue)
    return ArrayExpr(:call, [GemmWithTransform, A, B, C, epilogue], CuMatrix)
end

function GemmWithAdd(A::EClass, B::EClass, C::EClass, d)
    return ArrayExpr(:call, [GemmWithAdd, A, B, C, d])
end

function GemmWithTransform!(D::EClass, A::EClass, B::EClass, C::EClass, epilogue)
    return ArrayExpr(:call, [GemmWithTransform!, D, A, B, C, epilogue], CuMatrix)
end

function GemmKernelsBias(A::EClass, B::EClass, C::EClass, bias)
    return ArrayExpr(:call, [GemmBias, A, B, C, epilogue])
end

# temporary fix
function conditional_gemm(op, A, B, C)
    if istype(op, typeof(+))
        return Gemm(A, B, C)
    end
    return nothing
end

# big rewrite rules with custom implementations
const gemm_properties = @array_theory A B C D op d epi begin
    # TODO: make sure A, B, C is a matrix, and not a scalar!!
    # idea: make mul with scalar separate function? (this supports purely syntactical rewrites)
    # TODO: problem with dynamic rules like this is is that is does not work in the opposite direction

    #A * B => Gemm(A, B) where (istype(A, CuMatrix) && istype(B, CuMatrix))
    A * B + C => Gemm(A, B, C) where (istype(A, CuMatrix) && istype(B, CuMatrix) && istype(C, CuArray))

    # temporary fix as matching with op doesn't seem to work (hashing issue?) -> look at EGraph.lookup
    broadcast(op, A * B, C) => Gemm(A, B, C) where (istype(op, typeof(+)) && istype(A, CuMatrix) && istype(B, CuMatrix) && istype(C, CuArray))

    # TODO: these ones are broken for now
    #broadcast(+, A * B, C) => Gemm(A, B, C) where (istype(A, CuMatrix) && istype(B, CuMatrix) && istype(C, CuMatrix))
    #broadcast(+, C, A * B) => Gemm(A, B, C) where (istype(A, CuMatrix) && istype(B, CuMatrix) && istype(C, CuMatrix))

    # merge operations in prologue / epilogue
    # TODO: how to merge with prefix? prologue?
    # TODO: make sure epilogue is scalar

    # TODO: fix this ugly Lamda/App stuff by fixing array_rules
    broadcast(op, Gemm(A, B, C), d) => GemmWithTransform(A, B, C, Lambda(:el, App(op, [:el, d]))) where istype(d, Number)
    broadcast(op, d, Gemm(A, B, C)) => GemmWithTransform(A, B, C, Lambda(:el, App(op, [d, :el]))) where istype(d, Number)

    # fuse operations
    broadcast(op, Gemm(A, B, C)) => GemmWithTransform(A, B, C, op)
    broadcast(op, GemmWithTransform(A, B, C, epi), d) => GemmWithTransform(A, B, C, Lambda(:el, App(op, [App(epi, [:el]), d]))) where istype(d, Number)
    broadcast(op, d, GemmWithTransform(A, B, C, epi)) => GemmWithTransform(A, B, C, Lambda(:el, App(op, [d, App(epi, [:el])]))) where istype(d, Number)
    broadcast(op, GemmWithTransform(A, B, C, epi)) => GemmWithTransform(A, B, C, Lambda(:el, App(op, [App(epi, [:el])])))


    # 3: copyto rules
    copyto!(D, Gemm(A, B, C)) => Gemm!(D, A, B, C) where (istype(D, CuMatrix))
    copyto!(D, GemmWithTransform(A, B, C, epi)) => GemmWithTransform!(D, A, B, C, epi) where (istype(D, CuMatrix))

    # TODO: add gemm alpha(=0) return rule (only if C is dead tho)
    # Gemm(A, B, C, 1.0, 0,0) == Gemm!(A, B, C); C
end

# TODO:
# what determines how deep an intrinsic goes?
const gpu_intrinsics = [Intrinsic(GlobalRef(Main, :*), 1, [1]),
                        Intrinsic(GlobalRef(Main, :+), 1, [1])]


operations = Dict{Any, Any}()
operations[Base.broadcasted] = +Inf
operations[Base.materialize] = +Inf
operations[GemmWithTransform] = 1
operations[GemmWithTransform!] = 1
operations[Gemm] = 10
operations[Gemm!] = 10
#operations[matmul] = 1000
operations[tuple] = 0
operations[:tuple] = 0

# force replacement of broadcast
operations[broadcast] = 2000

# closures
operations[:->] = 0
operations[:block] = 0

using Metatheory: AbstractAnalysis, operation, arguments, ENodeTerm, ENodeLiteral, EGraph

function cost_function(n::ENodeTerm, g::EGraph, an::Type{<:AbstractAnalysis})
    op = operation(n)
    if op isa Input
        if op.type isa Core.Const
            op = op.type.val
        else
            op = op.val
        end
    end


    cost = get(operations, op, 1000)

    for id in arguments(n)
        eclass = g[id]
        # if the child e-class has not yet been analyzed, return +Inf
        !hasdata(eclass, an) && (cost += Inf; break)
        cost += last(getdata(eclass, an))
    end

    # interesting illustration:
    # TOOD: add debug levels?

    return cost
end

cost_function(n::ENodeLiteral, g::EGraph, an::Type{<:AbstractAnalysis}) = 100


# TODO: GemmKernels.matmul intrinsic

# @checked function cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B,
