using ArrayAbstractions
using LLVM

using Core.Compiler
const CC = Core.Compiler

using Base: llvmcall

using IRTools

include("gpucompiler.jl")

using .LazyCodegen: call_delayed

function compile_with_gpucompiler(func, argtype, args, cost_function; eltype=AbstractArray, extra_rules=[], intrinsics=[])

    # don't run a pass in the loop
    #=
    job, _ = native_job_with_pass(func, (argtype), pass; kernel=false)
    mi, _ = GPUCompiler.emit_julia(job)
    println("emitting julia done...")

    ctx = JuliaContext()

    println("inferring julia code & emitting llvm code")
    ir, ir_meta = GPUCompiler.emit_llvm(job, mi; ctx, libraries=false)

    println("emitting llvm done")

    compiled = ir_meta[2]
    rettype = compiled[mi].ci.rettype
    println(rettype)

    fn = LLVM.name(ir_meta.entry)
    @assert !isempty(fn)

    # BROKEN :(
    quote
        # different on v1.8
        Base.@_inline_meta
        Base.llvmcall(($(string(ir)), $fn), $rettype, $argtype, $(args...))
    end
    =#
end

function compile_deferred(func, args, cost_function; eltype=AbstractArray, extra_rules=[], intrinsics=[])
    aro = ArrOptimPass(eltype, cost_function, extra_rules=extra_rules, intrinsics=intrinsics)
    return call_delayed(aro, func, args...)
end

@static if VERSION < v"1.8.0-DEV.267"
    function replace_code_newstyle!(ci, ir, n_argtypes)
        return Core.Compiler.replace_code_newstyle!(ci, ir, n_argtypes-1)
    end
else
    using Core.Compiler: replace_code_newstyle!
end

using MacroTools
using Core.Compiler: CodeInfo, SlotNumber, Slot

function slots!(ci::CodeInfo)
  ss = Dict{Slot,SlotNumber}()
  for i = 1:length(ci.code)
    function f(x)
      x isa Slot || return x
      haskey(ss, x) && return ss[x]
      push!(ci.slotnames, x.id)
      push!(ci.slotflags, 0x00)
      ss[x] = SlotNumber(length(ci.slotnames))
    end
    for i = 1:length(ci.code)
        ci.code[i] = let x = ci.code[i]
            x isa Core.ReturnNode ? ( isdefined(x, :val) ? Core.ReturnNode(f(x.val)) : nothing ) : # some unreachable statements seem to cause havoc?
            x isa Core.GotoIfNot ? Core.GotoIfNot(f(x.cond), x.dest) :
            f(x)
        end
    end
  end
  return ci
end

# this is necessary as this injection involves pre-inferred julia code
function update!(ci::CodeInfo, ir::Core.Compiler.IRCode)
    replace_code_newstyle!(ci, ir, length(ir.argtypes)-1)

    ci.inferred = false
    ci.ssavaluetypes = length(ci.code)

    # push args
    for arg in ir.argtypes
        push!(ci.slotnames, Symbol(""))
        push!(ci.slotflags, 0)
    end

    slots!(ci)
    fill!(ci.slotflags, 0)

    return ci
end

dummy() = return

# IDEA: stop inlining when coming across an 'intrinsic'
function compile_expression(func, argtype, args, cost_function; eltype=AbstractArray, extra_rules=[], intrinsics=[])
    pass = ArrOptimPass(eltype, cost_function, extra_rules=extra_rules, intrinsics=intrinsics)

    world_count = Base.get_world_counter()

    # for internal passes -> need for custom interpreter
    interpreter = ArrayInterpreter(pass)
    println("compiling function...")

    ci_dummy = code_lowered(dummy, Tuple{})[1]

    code_info, _ = Base.code_typed(func, argtype, interp = interpreter)[1]

    # get method instance
    meth = which(func, argtype)
    sig = Base.signature_type(func, argtype)::Type
    (ti, env) = ccall(:jl_type_intersection_with_env, Any,
                        (Any, Any), sig, meth.sig)::Core.SimpleVector
    meth = Base.func_for_method_checked(meth, ti, env)
    method_instance = ccall(:jl_specializations_get_linfo, Ref{Core.MethodInstance},
                    (Any, Any, Any, UInt), meth, ti, env, world_count)

    ir = CC.inflate_ir(code_info, method_instance)
  
    update!(ci_dummy, ir)

    println(ci_dummy)

    return ci_dummy

    #println("before: $code_info")
    #println("performing opt pass...")
    #pass(code_info, method_instance)
    #println(code_info)

    #println("done!")
end
