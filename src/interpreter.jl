using Core.Compiler

using Core.Compiler: InferenceParams,
    OptimizationParams,
    OptimizationState,
    InferenceResult,
    CodeInfo

using Core.Compiler: WorldView
using Base: get_world_counter

const CC = Core.Compiler

export ArrOptimPass

struct ArrOptimPass
    element_type::Type
    extra_rules::Vector{Metatheory.AbstractRule}
    intrinsics::Vector{Intrinsic}
    mod::Module

    ArrOptimPass(eltype; mod=@__MODULE__, extra_rules=[], intrinsics=[]) = return new(eltype, extra_rules, intrinsics, mod)
end

"""
    ArrayInterpreter

Similar to NativeInterpreter, but with customized optimization passes
"""
struct ArrayInterpreter <: CC.AbstractInterpreter
    # Cache of inference results for this particular interpreter
    cache::Vector{InferenceResult}
    # The world age we're working inside of
    world::UInt

    # Parameters for inference and optimization
    inf_params::InferenceParams
    opt_params::OptimizationParams

    aro::ArrOptimPass
    ci_cache::Any

    function ArrayInterpreter(aro::ArrOptimPass, world::UInt = get_world_counter();
                              inline = true,
                              ci_cache = CodeCache()
                               )
        # Sometimes the caller is lazy and passes typemax(UInt).
        # we cap it to the current world age
        if world == typemax(UInt)
            world = get_world_counter()
        end

        # If they didn't pass typemax(UInt) but passed something more subtly
        # incorrect, fail out loudly.
        @assert world <= get_world_counter()

        optim_params = NamedTuple()

        if inline
            # def = 100
            # thersh = typemax(Int)
            optim_params = (optim_params..., inline_cost_threshold=1000000)
        end

        # for internal passes -> need for custom interpreter
        opt_params = CC.OptimizationParams(; optim_params...)
        inf_params = CC.InferenceParams()

        return new(
            # Initially empty cache
            Vector{InferenceResult}(),

            # world age counter
            world,

            # parameters for inference and optimization
            inf_params,
            opt_params,
            aro,
            ci_cache
        )
    end
end

# Quickly and easily satisfy the AbstractInterpreter API contract
CC.InferenceParams(ni::ArrayInterpreter) = ni.inf_params
CC.OptimizationParams(ni::ArrayInterpreter) = ni.opt_params
CC.get_world_counter(ni::ArrayInterpreter) = ni.world
CC.get_inference_cache(ni::ArrayInterpreter) = ni.cache
CC.code_cache(ni::ArrayInterpreter) = WorldView(ni.ci_cache, CC.get_world_counter(ni))

CC.lock_mi_inference(::ArrayInterpreter, mi::MethodInstance) = (mi.inInference = true; nothing)

CC.unlock_mi_inference(::ArrayInterpreter, mi::MethodInstance) = (mi.inInference = false; nothing)

# run the optimization work
function CC.optimize(interp::ArrayInterpreter, opt::OptimizationState,
                  params::OptimizationParams, caller::InferenceResult)
    ir = run_passes(opt.src, opt, caller, interp.aro)
    return CC.finish(interp, opt, params, ir, caller)
end

export @array_opt

macro array_opt(ex)
    esc(isa(ex, Expr) ? Base.pushmeta!(ex, :array_opt) : ex)
end

function run_passes(ci::CodeInfo, sv::OptimizationState, caller::InferenceResult, aro)
    ir = CC.convert_to_ircode(ci, sv)
    ir = CC.slot2reg(ir, ci, sv)
    ir = CC.compact!(ir)

    perform_array_opt = CC._any(@nospecialize(x) -> CC.isexpr(x, :meta) && x.args[1] === :array_opt, ir.meta)   # perform optimization
    #if (perform_array_opt)
    # optimize function body if decorated with @array_opt

    ir = custom_ssa_inlining_pass!(ir, ir.linetable, sv.inlining, ci.propagate_inbounds, aro)
    #ir = CC.ssa_inlining_pass!(ir, ir.linetable, sv.inlining, ci.propagate_inbounds)
    ir = CC.compact!(ir)

    if (perform_array_opt)
        ir = aro(ir, ci.parent.def.module)

        ir = CC.compact!(ir)
    end
    
    # verify_ir(ir)
    ir = CC.sroa_pass!(ir)
    ir = CC.adce_pass!(ir)
    ir = CC.type_lift_pass!(ir)
    ir = CC.compact!(ir)

    gref = GlobalRef(ci.parent.def.module, ci.parent.def.name)
    #=
    if JLOptions().debug_level == 2
        (verify_ir(ir); verify_linetable(ir.linetable))
    end
    =#
    return ir
end
