# ArrayIR
## Algorith to extract IR from Julia's SSA IR
Flow backwards through the SSA using it's `use-def` chain.
This DAG of `use-def` relations (models data-flow dependencies (real data dependendies; RAW dependency) between the statements), will form the partial ordering that will limit the degree of parallelisation.
