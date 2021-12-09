# wraps ex in closure
macro optimize(exs...)
    @macro 
end

macro optimize_call(exs...)
    code = ex[end]
    kwargs = ex[1:end-1]

    # A * B + C -> ArrayGemm(A, B, C)
end
