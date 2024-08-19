module PressureSolve

function jacobi!(f, f_old, g, collision, dx, maxIterations)
    @assert ndims(f) == ndims(f_old) == ndims(g) == ndims(collision) == length(dx)
    
    dxn2 = @. (one(dx) / dx)^2
    c0 = 0.5 / sum(dxn2)
    c = c0 * dxn2

    n = length(f)

    for iter = 1:maxIterations
        copy!(f_old, f)
        for i = 1:n
            if collision[i] > 0
                A = 0
                for (j, strid) in enumerate(strides(f))
                    a1 = collision[i-strid] > 0 ? f_old[i-strid] : f_old[i]
                    a2 = collision[i+strid] > 0 ? f_old[i+strid] : f_old[i]
                    A += c[j] * (a1 + a2)
                end
                f[i] = A - c0 * g[i]
            end
        end
    end
end

end # module