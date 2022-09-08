using Statistics
using JLD2

# process collected measurements

benchs = load("benchmarks.jld2")

for s in [1024]
    p(d) = round(1000*d, digits = 4)
    m = benchs["lenet before $s"]
    before = "$(p(mean(m))) \$\\pm\$$(p(std(m)))"
    a = benchs["lenet after $s"]
    after = "$(p(mean(a))) \$\\pm\$$(p(std(a)))"
    println("$(s)x$s & $before & $after")
end

#=
println("mlp results:")
println("values are in milliseconds")
for s in [64, 128, 512, 1024]
    p(d) = round(1000*d, digits = 4)
    m = benchs["mlp before $s"]
    before = "$(p(mean(m))) \$\\pm\$$(p(std(m)))"
    a = benchs["mlp after $s"]
    after = "$(p(mean(a))) \$\\pm\$$(p(std(a)))"
    println("$(s)x$s & $before & $after")
end

println("lenet results:")
println("values are in milliseconds")
for s in [64, 128, 512, 1024]
    p(d) = round(1000*d, digits = 4)
    m = benchs["lenet before $s"]
    before = "$(p(mean(m))) \$\\pm\$$(p(std(m)))"
    a = benchs["lenet after $s"]
    after = "$(p(mean(a))) \$\\pm\$$(p(std(a)))"
    println("$(s)x$s & $before & $after")
end
=#

