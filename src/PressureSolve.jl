module PressureSolve

function jacobi!(f, f_old, g, collision, dx, maxIterations)
    @assert ndims(f) == ndims(f_old) == ndims(g) == ndims(collision) == length(dx)
    
    dxn2 = @. (one(dx) / dx)^2
    c0 = 0.5 / sum(dxn2)
    c = c0 * dxn2

    d = ndims(f)
    n = length(f)

    for iter = 1:maxIterations
        copy!(f_old, f)

        for i = 1:n
            A = 0
            for (j, stride) in enumerate(strides(f))
                a1 = collision > 0 ? f_old[i-stride] : f_old[i]
                a2 = collision > 0 ? f_old[i+stride] : f_old[i]
                A += c[j] * (a1 + a2)
            end
            f[i] = A - c0 * g[i]
        end
    end
end

end # module