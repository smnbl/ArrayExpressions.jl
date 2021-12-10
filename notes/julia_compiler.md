# Julia Compiler - An Overview

Julia code -lowering> CodeInfo
-> IRCode - optim passes -> IRCode 
-Type Inference-> IRCode 
-> ir_to_code_inf (which uses replace_code_newstyle! to update previous ci with new source)
-> CodeInfo

# Function -> generated code
see: [julia ast](https://docs.julialang.org/en/v1/devdocs/ast/)

- atypes: argument types ~ this stores the types of the arguments!
- sparams: static parameters (polymorphism / generics stuff) ~ this stores values of the parameters (types)
- CodeInfo: container for holding (lowerd / inferred) source code

## Function =atypes, sparams=> Method 

## Metho =specializations=> MethodInstance
- `backedges`: used for cache (contains CodeInstances) invalidation, reverse-list of cache dependencies, it tracks all the MethodInstances that have been inferred or optimized that contain a possible call to this `MethodInstance`

## MethodInstance =cache=> CodeInstance
- `inferred`: contains inferred source (CodeInfo?) or nothing to indicate `rettype` is inferred
- `ftpr`: jlcall entry point
- `min_world` / `max_world`: range of world ages this method instance is valid to be called, if `max_world` is the special token value `-1` -> value is not yet known
