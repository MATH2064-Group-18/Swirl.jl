module Advection


function lerp(a, b, bias)
    return (1 - bias) * a + bias * b
end

function domainInterpolate(f, x, collision)
    if ndims(f) == 2
        return bilinearInterpolate(f, x, collision)
    end
    c = round.(Int, x)
    I = CartesianIndex(Tuple(c))
    return collision[I] > 0 ? f[I] : 0
end

function bilinearInterpolate(f, x, collision)
    c = round.(Int32, x)
    S = strides(f)
    if collision[1 + sum(@. S * (c - 1))] <= 0
        return 0
    end
    x0 = floor.(Int32, x)
    u = x - x0
    
    index0 = 1 + sum(@. S * (x0 - 1))

    b1 = lerp(f[index0], f[index0+S[1]], u[1])
    b2 = lerp(f[index0+S[2]], f[index0+S[1]+S[2]], u[1])

    return lerp(b1, b2, u[2])
end

function advectScalar!(f, vel, collision, dx, dt)
    f_old = similar(f)
    copy!(f_old, f)
    
    n = ndims(f)
    
    for I in CartesianIndices(f)
        if collision[I] > 0
            u = Vector{eltype(f)}(undef, n)
            indomain = true
            for j = 1:n
                #!!!
                u[j] = I[j] - vel[j, I] * dt / dx[j]
                if 0 > u[j] || u[j] > size(f, j)
                    indomain = false
                    break
                end 
            end

            if indomain
                f[I] = domainInterpolate(f_old, u, collision)
            end
        end
    end
end

end # module