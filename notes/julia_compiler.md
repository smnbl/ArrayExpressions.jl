# Julia Compiler - An Overview

```
 Function
    ↓
 Method
    ↓
MethodInstance: uninferred code
    ↓
CodeInstance: inferred code
```

## conceptual code transformations that take place in the compiler

1. surface syntax AST (structured representation of code as it is written) constructed by `julia-parser.scm`
2. lowered form IR is constructed by `julia-syntax.scm`
3. optimizations (like inlining)
4. type inference
5. code generation

### more detail with functions called:

see: [julia init](https://docs.julialang.org/en/v1/devdocs/init/)
1. `Base._start` parses input with `parse()` to generate CodeInfo lowered source (MethodInstance) (ast.c)
    1. parse_all() -> parse() -> invokes julia-parser.scm
`jl_method_def` adds method def to the method table

2. start in `jl_toplevel_eval_flex`? (in `src/toplevel.c`
3. this function calls `jl_typeinfer` to generate inferred code from MethodInstance
    - performs type inference (via `typeinf_ext_toplevel`)

Julia code -lowering> CodeInfo
-> IRCode - optim passes -> IRCode 
-Type Inference-> IRCode 
-> ir_to_code_inf (which uses replace_code_newstyle! to update previous ci with new source)
-> CodeInfo

## Function -> generated code

see: [julia ast](https://docs.julialang.org/en/v1/devdocs/ast/)

- atypes: argument types ~ this stores the types of the arguments!
- sparams: static parameters (polymorphism / generics stuff) ~ this stores values of the parameters (types)
- CodeInfo: container for holding (lowerd / inferred) source code

## Function =atypes, sparams=> Method 

## Metho =specializations=> MethodInstance

- `backedges`: used for cache (contains CodeInstances) invalidation, reverse-list of cache dependencies, it tracks all the MethodInstances that have been inferred or optimized that contain a possible call to this `MethodInstance`

! MethodInstances hold uninferred code (within their `.uninferred` field), inference creates CodeInstances and populates the cache (for 'a toplevel thunk' tho ??)!

## MethodInstance =cache=> CodeInstance

- `inferred`: contains inferred source (CodeInfo?) or nothing to indicate `rettype` is inferred
- `ftpr`: jlcall entry point
- `min_world` / `max_world`: range of world ages this method instance is valid to be called, if `max_world` is the special token value `-1` -> value is not yet known

## GPUCompiler

hooks in at third step by defining new NativeInterpreter
4. perform type inference (`ci_cache_populate` called by `compile_method_instance`)
5. codegen (called from `compile_method_instance` called by `irgen`)
6. irgen (happens in `irgen` (irgen.jl) called by)

### optim.jl

contains LLVM optimization passes (in same spirit as Julia's optimization pipeline)

## Intrinsics & Builtin functions
Intrinsics & builtin functions are implemented in c/cpp.
They are mapped to Julia functions and behave like normal julia functions to the user.
Referenced by a `jl_f_...` symbol.

### Julia's inlining cost-model
see also: [The inlining algorithm](https://docs.julialang.org/en/v1/devdocs/inference/)

To calculate the associated cost of a function invocation (mapped to a code instance object), cost is ~ calculated as follows:
- every `:invoke` intrinsic (a call for which all the inputs and outputs were succesfully inferred) is mapped to a fixed cost of 20 cycles
- while an `:call` (for functions other than builtins/intrinsics!!) invocation still requires dynamic dispatch, thus the associated cost will be much higher: ~ 1000 cycles!!!

## tfuncs.jl
`tfunc` is an inference lookup table for Julia's intrinsic & builtin functions, see `base/compiler/tfuncs.jl`, it tells the compiler the return type of something which it will blindly trust!
Tfunc seems short for **transfer function**, they define the type transfer functions to use when type inferring intrinsic & builtin functions.

## invoke vs call
- invoke: static dispatch, preselected on MethodInstance
- call: still has to specialize the function to a MethodInstance

## _extype
My current theory is that extype stands for 'extended type' aka it stands for an extended lattice element, (which contains extra annotations, added to the Julia base type like Const(..)), it can be converted to a native Julia type using `widenconst`

