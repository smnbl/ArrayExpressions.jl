# Julia Compiler

Julia code -lowering> CodeInfo
-> IRCode - optim passes -> IRCode 
-Type Inference-> IRCode 
-> ir_to_code_inf (which uses replace_code_newstyle! to update previous ci with new source)
-> CodeInfo
