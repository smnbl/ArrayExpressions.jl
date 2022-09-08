using Core.Compiler
using Dates

using Core.Compiler: InferenceParams,
    OptimizationParams,
    OptimizationState,
    InferenceResult,
    InferenceState,
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
    cost_function::Function
    ArrOptimPass(eltype, cost_function; mod=@__MODULE__, extra_rules=[], intrinsics=[]) = new(eltype, extra_rules, intrinsics, mod, cost_function)
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
            # change threshold on function body (lower for Core.Compiler functions?) that is being optimized?
            optim_params = (optim_params..., inline_cost_threshold=100000)
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

# 1.7
CC.may_optimize(::ArrayInterpreter) = true
CC.may_compress(::ArrayInterpreter) = false
CC.may_discard_trees(::ArrayInterpreter) = true
CC.verbose_stmt_info(::ArrayInterpreter) = false

function CC.add_remark!(::ArrayInterpreter, sv::InferenceState, msg)
end

#= 1.8
# run the optimization work
function CC.optimize(interp::ArrayInterpreter, opt::OptimizationState,
                  params::OptimizationParams, caller::InferenceResult)
    ir = run_passes(opt.src, opt, caller, interp.aro)
    return CC.finish(interp, opt, params, ir, caller)
end
=#

# run the optimization work
function CC.optimize(interp::ArrayInterpreter, opt::OptimizationState, params::OptimizationParams, @nospecialize(result))
    nargs = Int(opt.nargs) - 1
    # timeit?
    ir = run_passes(opt.src, nargs, opt, interp.aro)
    # CC.finish(interp, opt, params, ir, result)
    finish(interp, opt, params, ir, result)
end

# compute inlining cost and sideeffects
# doesnt work :(
function finish(interp::ArrayInterpreter, opt::OptimizationState, params::OptimizationParams, ir::IRCode, @nospecialize(result))
    (; src, nargs, linfo) = opt

    aro = interp.aro

    ref = GlobalRef(src.parent.def.module, src.parent.def.name)
    #println(ref)

    if (src.parent.def.module === Core.Compiler)
        params = CC.OptimizationParams(; inline_cost_threshold=100)
    end

    # this will write .inlineable
    CC.finish(interp, opt::OptimizationState, params::OptimizationParams, ir::IRCode, @nospecialize(result))

    # overwrite inlineability
    if inrules(ir, ref, (aro.extra_rules)) || inintrinsics(ir, ref, gpu_intrinsics)
        src.inlineable = false
    end
end

export @array_opt

macro array_opt(ex)
    esc(isa(ex, Expr) ? Base.pushmeta!(ex, :array_opt) : ex)
end

function run_passes(ci::CodeInfo, nargs::Int, sv::OptimizationState, aro)
    preserve_coverage = CC.coverage_enabled(sv.mod)
    ir = CC.convert_to_ircode(ci, CC.copy_exprargs(ci.code), preserve_coverage, nargs, sv)
    ir = CC.slot2reg(ir, ci, nargs, sv)
    #@Base.show ("after_construct", ir)
    # TODO: Domsorting can produce an updated domtree - no need to recompute here
    ir = CC.compact!(ir)
    #@timeit "Inlining" ir = ssa_inlining_pass!(ir, ir.linetable, sv.inlining, ci.propagate_inbounds)
    #ir = custom_ssa_inlining_pass!(ir, ir.linetable, sv.inlining, ci.propagate_inbounds, aro)
    ir = CC.ssa_inlining_pass!(ir, ir.linetable, sv.inlining, ci.propagate_inbounds)
    #@timeit "verify 2" verify_ir(ir)
    ir = CC.compact!(ir)

    #@Base.show ("before_sroa", ir)
    ir = CC.getfield_elim_pass!(ir)
    #@Base.show ir.new_nodes
    #@Base.show ("after_sroa", ir)
    ir = CC.adce_pass!(ir)
    ir = CC.type_lift_pass!(ir)
    ir = CC.compact!(ir)

    perform_array_opt = CC._any(@nospecialize(x) -> CC.isexpr(x, :meta) && x.args[1] === :array_opt, ir.meta)   # perform optimization
    if (perform_array_opt)
        println("performing array opt pass")
        #before = Dates.now()
        time = @elapsed ir = aro(ir, ci.parent.def.module)
        #after = Dates.now() - before
        println("array opt pass done $time")
        # println("array opt pass done: took $after!")
    end
    ir = CC.compact!(ir)

    #@Base.show ir
    #=
    if JLOptions().debug_level == 2
        @timeit "verify 3" (verify_ir(ir); verify_linetable(ir.linetable))
    end
    =#
    return ir
end


#= 1.8
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
=#
