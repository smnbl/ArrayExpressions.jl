# How to implement ArrayAbstractions in Julia
## Metatheory interface upgrades
- add suport for type checks using type annotations to make them more intuitive (might require modifications to Metatheory.jl)
    -> look at ways to make Metatheory work with `symtype` from TermInterface to improve type decorations of rewrite rules

- look at the concat -> matmul -> split pattern (seems to be applied a lot in research as a fruitful optimization)
- look for a better way to fix matching non-top level function names (like Base.Broadcast.materialize); problem with classical matcher (-> report bug?) ?

- make lambda expressions work with Metatheory (does Metatheory not recurse in block arguments?; be careful with bounded variables)
- make replacement of operations possible? (requires changes to extraction mechanism) -> TODO: open up an issue about this

- make cost function that optimizes for replacements

## imperative -> functional control flow (see XLA.jl paper section 5.2)
        - replace phi nodes make up conditions for functional control flow
            - (in XLA.jl): computations inbetween get **outlined** in seperate functions
        - loop identification:
            - strongly connected regions of control flow get outlined as the loop body
            - put values that have uses outside the strongly strongly connected region as wel as loop caried phi's into the **iteration state**
                - a phi node corresponds to uses of loop values outside the loop?



