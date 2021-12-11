# interface with the julia compiler

using Core.Compiler: MethodInstance

const C = Core
const CC = Core.Compiler

# JULIA IR
function emit_julia(f, @nospecialize(atype), sparams::C.SimpleVector)
    sig = CC.signature_type(f, atype)
    meth::CC.MethodMatch = Base._which(sig)

    mi::MethodInstance = CC.specialize_method(meth)
    return mi
end

# INFERENCE + CODEGEN
struct ArrayInterpreter <: CC.AbstractInterpreter
    # cache of inference results
    cache::Vector{CC.InferenceResult}

    # the world age we're working inside of
    world::UInt

    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams

    function ArrayInterpreter(
            world::UInt = CC.get_world_counter();
            inf_params = CC.InferenceParams(),
            opt_params = CC.OptimizationParams(

                            inlining = false,
                                              ),
        )

        @assert world <= CC.get_world_counter()

        return new(
                   Vector{CC.InferenceResult}(),
                   world,
                   inf_params,
                   opt_params,
                  )
    end
end

# Quickly and easily satisfy the AbstractInterpreter API contract
CC.InferenceParams(ai::ArrayInterpreter) = ai.inf_params
CC.OptimizationParams(ai::ArrayInterpreter) = ai.opt_params
CC.get_world_counter(ai::ArrayInterpreter) = ai.world
CC.get_inference_cache(ai::ArrayInterpreter) = ai.cache

# re-use the global cache, but limit the view using WorldView
# WorldView limits range of valid cache contents to cache entries that have world_age == world_counter
# TODO: built own cache to not infere with NativeInterpreter's global code_cache
CC.code_cache(ai::ArrayInterpreter) = CC.WorldView(CC.GLOBAL_CI_CACHE, CC.get_world_counter(ai))

# hints that a certain MethodInstance is in inference to prevent inferring the same thing multiple times?
CC.lock_mi_inference(ai::ArrayInterpreter, mi::MethodInstance) = (mi.inInference = true; nothing)
CC.unlock_mi_inference(ai::ArrayInterpreter, mi::MethodInstance) = (mi.inInference = false; nothing)

CC.may_optimize(::ArrayInterpreter) = true
CC.may_compress(::ArrayInterpreter) = true
CC.may_discard_trees(::ArrayInterpreter) = true
CC.verbose_stmt_info(::ArrayInterpreter) = false

# TODO: can be used to add analysis remarks during inference for the current line
function CC.add_remark!(ai::ArrayInterpreter, sv::CC.InferenceState, msg)
    @debug "Inference remark during array compilation of $(sv.linfo): $msg"
end

# resolve globalref to global variable (function, struct, variables, ...)
resolve(gr::GlobalRef) = getproperty(gr.mod, gr.name)

function CC.optimize(interp::ArrayInterpreter, opt::CC.OptimizationState, params::CC.OptimizationParams, @nospecialize(result))
    @info "optimizing..."
    nargs = Int(opt.nargs) - 1

    # TODO: interface changed in 1.8
    ir = CC.run_passes(opt.src, nargs, opt)

    ### perform optimization?
    # outline:
    #   1. replace (dot matrix matrix) -> (gemm matrix matrix)
    #   2. replace (plus (gemm matrix matrix) matrix) -> (gemm matrix matrix matrix)
    #
    inst_stream = ir.stmts
    for (idx, stmt) in enumerate(inst_stream.inst)
        stmt isa Expr || continue
        if stmt.head == :(=)
            stmt = stmt.args[2]
        end

        stmt isa Expr || continue

        # TODO: also support :call in case dynamic dispatched?
        stmt.head == :invoke || continue

        # in invoke, function ref is in second argument
        stmt.args[2] isa GlobalRef || continue
        resolve(stmt.args[2]) == ArrayPlus || continue

        loc_dot = stmt.args[3].id

        lhs = inst_stream.inst[loc_dot]
        lhs isa Expr || continue
        lhs.head == :invoke || continue

        # in invoke, function ref is in second argument
        lhs.args[2] isa GlobalRef || continue
        resolve(lhs.args[2]) == ArrayDot || continue

        println("replacing :)")
        println(lhs.args)

        # get GEMM arguments
        A = lhs.args[3]
        B = lhs.args[4]
        C = stmt.args[4]

        # TODO: make sure ArrayDot is only used as argument for ArrayMul
        # -> needs use-def info -> look at compiler passes
        CC.setindex!(ir, Expr(:call, :+, 0), CC.SSAValue(loc_dot))# NOP ArrayDot
        CC.setindex!(ir, Expr(:call, ArrayGemm, A, B, C), CC.SSAValue(idx)) # perform replacement
    end

    ###

    CC.finish(interp, opt, params, ir, result)
end

## LLVM IR Generation
#

## PUTTING IT ALL TOGETHER
function codegen(output::Symbol, f, atype, sparams::C.SimpleVector)
    @info "Emitting Julia"
    mi = emit_julia(f, atype, sparams)

    output == :julia && return Base.uncompressed_ir(mi.def)

    @info "Performing Inference"
    interp = ArrayInterpreter()
    src = CC.typeinf_ext(interp, mi)

    # get code instance & link inferred src
    ci = CC.getindex(CC.code_cache(interp), mi)
    ci.inferred = src

    output == :typed && return src

    # TODO:
    # 1. populate cache with code
    # 2. generate native IR
end
