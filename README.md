# ArrayExpressions

## Install & setup package
```
git clone https://github.com/smnbl/ArrayExpressions.jl
```

Setup customized GemmKernels
```
cd ArrayExpressions.jl
git clone --brach=variable_matrix_size https://github.com/smnbl/GemmKernels.jl.git test/gpu/GemmKernels.jl
```

Instantiate dependencies
```
julia --project="." -e "using Pkg; Pkg.instantiate()"
```

## Run benchmarks
### Convolutional network
```
julia --project="." test/conv_mnist.jl <img_size>
```

With <img_size> an integer encoding the input data size, e.g. '64', '128', '1024'

### Multilayer perceptron
```
julia --project="." test/mlp.jl <img_size>
```

With <img_size> an integer encoding the input data size, e.g. '64', '128', '1024'

### Processing the results
Invoke the results.jl script:
```
julia --project="." results.jl
```
