# How to implement ArrayAbstractions in Julia
## Metatheory interface upgrades
- add support for extra dynamic constraints

## imperative -> functional control flow (see XLA.jl paper section 5.2)
        - replace phi nodes make up conditions for functional control flow
            - (in XLA.jl): computations inbetween get **outlined** in seperate functions
        - loop identification:
            - strongly connected regions of control flow get outlined as the loop body
            - put values that have uses outside the strongly strongly connected region as wel as loop caried phi's into the **iteration state**
                - a phi node corresponds to uses of loop values outside the loop?



