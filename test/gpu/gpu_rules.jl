using ArrayAbstractions
using ArrayAbstractions: App, Lambda, Intrinsic
using Metatheory

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
    (A*B) + C => Gemm(A, B, C) where (istype(A, CuMatrix) && istype(B, CuMatrix) && istype(C, CuMatrix))
    # (A*B) + C => ArrayExpr(:call, [:Gemm, A, B, C], Union{}) where (istype(A, Matrix) && istype(B, Matrix) && istype(C, Matrix))

    # merge operations in prologue / epilogue
    # TODO: how to merge with prefix? prologue?
    # TODO: make sure epilogue is scalar
    #broadcast(op, Gemm(A, B, C), d) => GemmWithEpilogue(A, B, C, ArrayExpr(:call, [:EpiL, op, d])) where istype(d, Number)
    #broadcast(op, d, Gemm(A, B, C)) => GemmWithEpilogue(A, B, C, ArrayExpr(:call, [:EpiR, op, d]))  where istype(d, Number)
    broadcast(op, Gemm(A, B, C), d) => GemmWithEpilogue(A, B, C, Lambda(:el, App(op, [:el, d]))) where istype(d, Number)
    broadcast(op, d, Gemm(A, B, C)) => GemmWithEpilogue(A, B, C, ArrayExpr(:call, [:EpiR, op, d]))  where istype(d, Number)

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

    # TODO: add gemm alpha rule
end

# TODO:
# what determines how deep an intrinsic goes?
const gpu_intrinsics = [Intrinsic(GlobalRef(Main, :*), 1, [1]),
                        Intrinsic(GlobalRef(Main, :+), 1, [1])]

# TODO: GemmKernels.matmul intrinsic

# @checked function cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B,
