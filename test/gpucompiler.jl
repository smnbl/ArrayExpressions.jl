using ArrayAbstractions
const AA = ArrayAbstractions

using GPUCompiler
using GPUCompiler:
    cached_compilation,
    AbstractCompilerParams,
    GPUInterpreter

# the GPU runtime library
module TestRuntime
    # dummy methods
    signal_exception() = return
    malloc(sz) = ccall("extern malloc", llvmcall, Csize_t, (Csize_t,), sz)
    report_oom(sz) = return
    report_exception(ex) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return
end

struct TestCompilerParams <: AbstractCompilerParams end
GPUCompiler.runtime_module(::CompilerJob{<:Any,TestCompilerParams}) = TestRuntime

Base.@kwdef struct ArrayNativeCompilerTarget <: GPUCompiler.AbstractCompilerTarget
    cpu::String=(LLVM.version() < v"8") ? "" : unsafe_string(LLVM.API.LLVMGetHostCPUName())
    features::String=(LLVM.version() < v"8") ? "" : unsafe_string(LLVM.API.LLVMGetHostCPUFeatures())
    always_inline::Bool=false # will mark the job function as always inline
    jlruntime::Bool=true
    aro::ArrOptimPass = ArrOptimPass()
    cache::AA.CodeCache = AA.CodeCache()
end

GPUCompiler.llvm_triple(::ArrayNativeCompilerTarget) = Sys.MACHINE

function GPUCompiler.llvm_machine(target::ArrayNativeCompilerTarget)
    triple = GPUCompiler.llvm_triple(target)

    t = GPUCompiler.Target(triple=triple)

    tm = GPUCompiler.TargetMachine(t, triple, target.cpu, target.features)
    GPUCompiler.asm_verbosity!(tm, true)

    return tm
end

function GPUCompiler.process_entry!(job::CompilerJob{ArrayNativeCompilerTarget}, mod::LLVM.Module, entry::LLVM.Function)
    ctx = GPUCompiler.context(mod)
    if job.target.always_inline
        push!(function_attributes(entry), EnumAttribute("alwaysinline", 0; ctx))
    end
    invoke(GPUCompiler.process_entry!, Tuple{CompilerJob, LLVM.Module, LLVM.Function}, job, mod, entry)
end

## job
GPUCompiler.runtime_slug(job::CompilerJob{ArrayNativeCompilerTarget}) = "native_$(job.target.cpu)-$(hash(job.target.features))$(job.target.jlruntime ? "-jlrt" : "")"
GPUCompiler.uses_julia_runtime(job::CompilerJob{ArrayNativeCompilerTarget}) = job.target.jlruntime

# TODO: fix a way to insert the aro pass
function native_job_with_pass(@nospecialize(func), @nospecialize(types), aro = AA.ArrOptimPass(AbstractArray, (x...) -> 0); kernel::Bool=false, entry_abi=:specfunc, kwargs...)
    source = FunctionSpec(func, Base.to_tuple_type(types), kernel)
    target = ArrayNativeCompilerTarget(aro = aro, always_inline=true)
    params = TestCompilerParams()

    function end_pass(code_info::CC.CodeInfo, mi::CC.MethodInstance)
        ir = CC.inflate_ir(code_info, mi)
        ir = aro(ir, mi.def.module)

        CC.replace_code_newstyle!(code_info, ir, Int64(mi.def.nargs))

        return code_info
    end
    CompilerJob(target, source, params, entry_abi), kwargs
end

# use fresh code cache for each job
GPUCompiler.ci_cache(job::CompilerJob{ArrayNativeCompilerTarget}) = job.target.cache

GPUCompiler.get_interpreter(job::CompilerJob{ArrayNativeCompilerTarget}) =
    ArrayInterpreter(job.target.aro, job.source.world; ci_cache = GPUCompiler.ci_cache(job))

using GPUCompiler:
@timeit_debug, compile_method_instance, functions, @compiler_assert, safe_name, ModulePassManager, linkage!, internalize!, always_inliner!, can_throw, add!, add_lowering_passes!, run!, process_module!, process_entry!, mangle_call, lower_throw!

function optimize!(job::CompilerJob{ArrayNativeCompilerTarget}, mod::LLVM.Module)
    triple = llvm_triple(job.target)
    tm = llvm_machine(job.target)

    global current_job
    current_job = job

    @dispose pm=ModulePassManager() begin
        addTargetPasses!(pm, tm, triple)
        addOptimizationPasses!(pm)
        run!(pm, mod)
    end

    # NOTE: we need to use multiple distinct pass managers to force pass ordering;
    #       intrinsics should never get lowered before Julia has optimized them.
    # XXX: why doesn't the barrier noop pass work here?

    # lower intrinsics
    @dispose pm=ModulePassManager() begin
        addTargetPasses!(pm, tm, triple)

        if !uses_julia_runtime(job)
            add!(pm, FunctionPass("LowerGCFrame", lower_gc_frame!))
        end

        if job.source.kernel
            # GC lowering is the last pass that may introduce calls to the runtime library,
            # and thus additional uses of the kernel state intrinsic.
            # TODO: now that all kernel state-related passes are being run here, merge some?
            add!(pm, ModulePass("AddKernelState", add_kernel_state!))
            add!(pm, FunctionPass("LowerKernelState", lower_kernel_state!))
            add!(pm, ModulePass("CleanupKernelState", cleanup_kernel_state!))
        end

        if !uses_julia_runtime(job)
            # remove dead uses of ptls
            aggressive_dce!(pm)
            add!(pm, ModulePass("LowerPTLS", lower_ptls!))
        end

        if uses_julia_runtime(job)
            lower_exc_handlers!(pm)
        end
        # the Julia GC lowering pass also has some clean-up that is required

        # SIMON: this seems to error
        late_lower_gc_frame!(pm)
        if uses_julia_runtime(job)
            final_lower_gc!(pm)
        end

        remove_ni!(pm)
        remove_julia_addrspaces!(pm)

        if uses_julia_runtime(job)
            # We need these two passes and the instcombine below
            # after GC lowering to let LLVM do some constant propagation on the tags.
            # and remove some unnecessary write barrier checks.
            gvn!(pm)
            sccp!(pm)
            # Remove dead use of ptls
            dce!(pm)
            LLVM.Interop.lower_ptls!(pm, dump_native(job))
            instruction_combining!(pm)
            # Clean up write barrier and ptls lowering
            cfgsimplification!(pm)
        end

        # Julia's operand bundles confuse the inliner, so repeat here now they are gone.
        # FIXME: we should fix the inliner so that inlined code gets optimized early-on
        always_inliner!(pm)

        # some of Julia's optimization passes happen _after_ lowering intrinsics
        combine_mul_add!(pm)
        div_rem_pairs!(pm)

        run!(pm, mod)
    end

    # target-specific optimizations
    optimize_module!(job, mod)

    # we compile a module containing the entire call graph,
    # so perform some interprocedural optimizations.
    #
    # for some reason, these passes need to be distinct from the regular optimization chain,
    # or certain values (such as the constant arrays used to populare llvm.compiler.user ad
    # part of the LateLowerGCFrame pass) aren't collected properly.
    #
    # these might not always be safe, as Julia's IR metadata isn't designed for IPO.
    @dispose pm=ModulePassManager() begin
        addTargetPasses!(pm, tm, triple)

        # simplify function calls that don't use the returned value
        dead_arg_elimination!(pm)

        run!(pm, mod)
    end

    # compare to Clang by using the pass manager builder APIs:
    #LLVM.clopts("-print-after-all", "-filter-print-funcs=$(LLVM.name(entry))")
    #@dispose pm=ModulePassManager() begin
    #    addTargetPasses!(pm, tm, triple)
    #    PassManager@dispose pmb=Builder() begin
    #        optlevel!(pmb, 2)
    #        populate!(pm, pmb)
    #    end
    #    run!(pm, mod)
    #end

    return
end

# irgen with extra end pass
function GPUCompiler.irgen(job::CompilerJob{ArrayNativeCompilerTarget}, method_instance::Core.MethodInstance;
               ctx::Context)
    mod, compiled = @timeit_debug to "emission" compile_method_instance(job, method_instance; ctx)

    ci = compiled[method_instance].ci

    #=
    println("performing at end pass")

    src = if ci.inferred isa Vector{UInt8}
        ccall(:jl_uncompress_ir, Any, (Any, Ptr{Cvoid}, Any),
                method_instance.def, C_NULL, ci.inferred)
    else
        ci.inferred
    end

    job.target.aro(src, method_instance) 
    =#

    if job.entry_abi === :specfunc
        entry_fn = compiled[method_instance].specfunc
    else
        entry_fn = compiled[method_instance].func
    end

    # clean up incompatibilities
    @timeit_debug to "clean-up" begin
        for llvmf in functions(mod)
            # only occurs in debug builds
            delete!(function_attributes(llvmf), EnumAttribute("sspstrong", 0; ctx))

            if Sys.iswindows()
                personality!(llvmf, nothing)
            end

            # remove the non-specialized jfptr functions
            # TODO: Do we need to remove these?
            if job.entry_abi === :specfunc
                if startswith(LLVM.name(llvmf), "jfptr_")
                    unsafe_delete!(mod, llvmf)
                end
            end
        end

        # remove the exception-handling personality function
        if Sys.iswindows() && "__julia_personality" in functions(mod)
            llvmf = functions(mod)["__julia_personality"]
            @compiler_assert isempty(uses(llvmf)) job
            unsafe_delete!(mod, llvmf)
        end
    end

    # target-specific processing
    process_module!(job, mod)
    entry = functions(mod)[entry_fn]

    # sanitize function names
    # FIXME: Julia should do this, but apparently fails (see maleadt/LLVM.jl#201)
    for f in functions(mod)
        LLVM.isintrinsic(f) && continue
        llvmfn = LLVM.name(f)
        startswith(llvmfn, "julia.") && continue # Julia intrinsics
        startswith(llvmfn, "llvm.") && continue # unofficial LLVM intrinsics
        llvmfn′ = safe_name(llvmfn)
        if llvmfn != llvmfn′
            @assert !haskey(functions(mod), llvmfn′)
            LLVM.name!(f, llvmfn′)
        end
    end

    # rename and process the entry point
    if job.source.name !== nothing
        LLVM.name!(entry, safe_name(string("julia_", job.source.name)))
    end
    if job.source.kernel
        LLVM.name!(entry, mangle_call(entry, job.source.tt))
    end
    entry = process_entry!(job, mod, entry)
    if job.entry_abi === :specfunc
        func = compiled[method_instance].func
        specfunc = LLVM.name(entry)
    else
        func = LLVM.name(entry)
        specfunc = compiled[method_instance].specfunc
    end

    compiled[method_instance] =
        (; ci, func, specfunc)

    # minimal required optimization
    @timeit_debug to "rewrite" ModulePassManager() do pm
        global current_job
        current_job = job

        linkage!(entry, LLVM.API.LLVMExternalLinkage)

        # internalize all functions, but keep exported global variables
        exports = String[LLVM.name(entry)]
        for gvar in globals(mod)
            push!(exports, LLVM.name(gvar))
        end
        internalize!(pm, exports)

        # inline llvmcall bodies
        always_inliner!(pm)

        can_throw(job) || add!(pm, ModulePass("LowerThrow", lower_throw!))

        add_lowering_passes!(job, pm)

        run!(pm, mod)
    end

    return mod, compiled
end
module LazyCodegen
    using LLVM
    using LLVM.Interop
    using GPUCompiler

    import ..native_job_with_pass

    function absolute_symbol_materialization(name, ptr)
        address = LLVM.API.LLVMOrcJITTargetAddress(reinterpret(UInt, ptr))
        flags = LLVM.API.LLVMJITSymbolFlags(LLVM.API.LLVMJITSymbolGenericFlagsExported, 0)
        symbol = LLVM.API.LLVMJITEvaluatedSymbol(address, flags)
        gv = LLVM.API.LLVMJITCSymbolMapPair(name, symbol)

        return LLVM.absolute_symbols(Ref(gv))
    end

    function define_absolute_symbol(jd, name)
        ptr = LLVM.find_symbol(name)
        if ptr !== C_NULL
            LLVM.define(jd, absolute_symbol_materialization(name, ptr))
            return true
        end
        return false
    end

    struct CompilerInstance
        jit::LLVM.LLJIT
        lctm::LLVM.LazyCallThroughManager
        ism::LLVM.IndirectStubsManager
    end
    const jit = Ref{CompilerInstance}()

    function __init__()
        optlevel = LLVM.API.LLVMCodeGenLevelDefault
        tm = GPUCompiler.JITTargetMachine(optlevel=optlevel)
        LLVM.asm_verbosity!(tm, true)

        lljit = LLJIT(;tm)

        jd_main = JITDylib(lljit)

        prefix = LLVM.get_prefix(lljit)
        dg = LLVM.CreateDynamicLibrarySearchGeneratorForProcess(prefix)
        add!(jd_main, dg)
        if Sys.iswindows() && Int === Int64
            # TODO can we check isGNU?
            define_absolute_symbol(jd_main, mangle(lljit, "___chkstk_ms"))
        end

        es = ExecutionSession(lljit)

        lctm = LLVM.LocalLazyCallThroughManager(triple(lljit), es)
        ism = LLVM.LocalIndirectStubsManager(triple(lljit))

        jit[] = CompilerInstance(lljit, lctm, ism)
        atexit() do
            ci = jit[]
            dispose(ci.ism)
            dispose(ci.lctm)
            dispose(ci.jit)
        end
    end

    function get_trampoline(job)
        compiler = jit[]
        lljit = compiler.jit
        lctm  = compiler.lctm
        ism   = compiler.ism

        # We could also use one dylib per job
        jd = JITDylib(lljit)

        entry_sym = String(gensym(:entry))
        target_sym = String(gensym(:target))
        flags = LLVM.API.LLVMJITSymbolFlags(
            LLVM.API.LLVMJITSymbolGenericFlagsCallable |
            LLVM.API.LLVMJITSymbolGenericFlagsExported, 0)
        entry = LLVM.API.LLVMOrcCSymbolAliasMapPair(
            mangle(lljit, entry_sym),
            LLVM.API.LLVMOrcCSymbolAliasMapEntry(
                mangle(lljit, target_sym), flags))

        mu = LLVM.reexports(lctm, ism, jd, Ref(entry))
        LLVM.define(jd, mu)

        # 2. Lookup address of entry symbol
        addr = lookup(lljit, entry_sym)

        # 3. add MU that will call back into the compiler
        sym = LLVM.API.LLVMOrcCSymbolFlagsMapPair(mangle(lljit, target_sym), flags)

        function materialize(mr)
            JuliaContext() do ctx
                ir, meta = GPUCompiler.codegen(:llvm, job; validate=false, ctx)

                # Rename entry to match target_sym
                LLVM.name!(meta.entry, target_sym)

                # So 1. serialize the module
                buf = convert(MemoryBuffer, ir)

                # 2. deserialize and wrap by a ThreadSafeModule
                ThreadSafeContext() do ctx
                    mod = parse(LLVM.Module, buf; ctx=context(ctx))
                    tsm = ThreadSafeModule(mod; ctx)

                    il = LLVM.IRTransformLayer(lljit)
                    LLVM.emit(il, mr, tsm)
                end
            end

            return nothing
        end

        function discard(jd, sym)
        end

        mu = LLVM.CustomMaterializationUnit(entry_sym, Ref(sym), materialize, discard)
        LLVM.define(jd, mu)
        return addr
    end

    import GPUCompiler: deferred_codegen_jobs
    function deferred_codegen(aro, f, tt)
        job, _ = native_job_with_pass(f, tt, aro)

        addr = get_trampoline(job)
        trampoline = pointer(addr)
        id = Base.reinterpret(Int, trampoline)

        deferred_codegen_jobs[id] = job

        ptr = ccall("extern deferred_codegen", llvmcall, Ptr{Cvoid}, (Ptr{Cvoid},), trampoline)
        assume(ptr != C_NULL)
        return ptr
    end

    @generated function abi_call(f::Ptr{Cvoid}, rt::Type{RT}, tt::Type{T}, func::F, args::Vararg{Any, N}) where {T, RT, F, N}
        argtt    = tt.parameters[1]
        rettype  = rt.parameters[1]
        argtypes = DataType[argtt.parameters...]

        argexprs = Union{Expr, Symbol}[]
        ccall_types = DataType[]

        before = :()
        after = :(ret)


        # Note this follows: emit_call_specfun_other
        JuliaContext() do ts_ctx
            ctx = GPUCompiler.unwrap_context(ts_ctx)
            if !isghosttype(F) && !Core.Compiler.isconstType(F)
                isboxed = GPUCompiler.deserves_argbox(F)
                argexpr = :(func)
                if isboxed
                    push!(ccall_types, Any)
                else
                    et = convert(LLVMType, func; ctx)
                    if isa(et, LLVM.SequentialType) # et->isAggregateType
                        push!(ccall_types, Ptr{F})
                        argexpr = Expr(:call, GlobalRef(Base, :Ref), argexpr)
                    else
                        push!(ccall_types, F)
                    end
                end
                push!(argexprs, argexpr)
            end

            T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
            T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)

            for (source_i, source_typ) in enumerate(argtypes)
                if GPUCompiler.isghosttype(source_typ) || Core.Compiler.isconstType(source_typ)
                    continue
                end

                argexpr = :(args[$source_i])

                isboxed = GPUCompiler.deserves_argbox(source_typ)
                et = isboxed ? T_prjlvalue : convert(LLVMType, source_typ; ctx)

                if isboxed
                    push!(ccall_types, Any)
                elseif isa(et, LLVM.SequentialType) # et->isAggregateType
                    push!(ccall_types, Ptr{source_typ})
                    argexpr = Expr(:call, GlobalRef(Base, :Ref), argexpr)
                else
                    push!(ccall_types, source_typ)
                end
                push!(argexprs, argexpr)
            end

            if GPUCompiler.isghosttype(rettype) || Core.Compiler.isconstType(rettype)
                # Do nothing...
                # In theory we could set `rettype` to `T_void`, but ccall will do that for us
            # elseif jl_is_uniontype?
            elseif !GPUCompiler.deserves_retbox(rettype)
                rt = convert(LLVMType, rettype; ctx)
                if !isa(rt, LLVM.VoidType) && GPUCompiler.deserves_sret(rettype, rt)
                    before = :(sret = Ref{$rettype}())
                    pushfirst!(argexprs, :(sret))
                    pushfirst!(ccall_types, Ptr{rettype})
                    rettype = Nothing
                    after = :(sret[])
                end
            else
                # rt = T_prjlvalue
            end
        end

        quote
            $before
            ret = ccall(f, $rettype, ($(ccall_types...),), $(argexprs...))
            $after
        end
    end

    @inline function call_delayed(aro, f::F, args...) where F
        tt = Tuple{map(Core.Typeof, args)...}
        rt = Core.Compiler.return_type(f, tt)
        ptr = deferred_codegen(aro, f, tt)


        quote
            LazyCodegen.abi_call($ptr, $rt, $tt, $f, ($args)...)
        end
    end
end
