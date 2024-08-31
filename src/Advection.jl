module Advection

using StaticArrays

export advectScalar!, advectVector!

"""
    lerp(a, b, bias)

Linearly interpolate between `a` and `b` based on `bias`.
"""
function lerp(a, b, bias)
    return (1 - bias) * a + bias * b
end

"""
    domainInterpolate(f, x, collision)

Interpolate values of `f` at pos `x` in index space.
"""
function domainInterpolate(f, x, collision)
    if ndims(f) == 2
        return bilinearInterpolate(f, x, collision)
    end
    return generalLinearInterpolate(f, x, collision)
end

function nearestInterpolate(f, x, collision)
    c = round.(Int, x)
    I = CartesianIndex(Tuple(c))
    return collision[I] > 0 ? f[I] : 0
end

function bilinearInterpolate(f, x, collision)
    c = round.(Int32, x)
    S = strides(collision)
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

function trilinearInterpolate(f, x, collision)
    c = round.(Int32, x)
    S = strides(collision)
    if collision[i + sum(@. S * (c - 1))] <= 0
        return 0
    end

    x0 = floor.(Int32, x)
    u = x - x0

    index0 = 1 + sum(@. S * (x0 - 1))

    A = @SMatrix [lerp(f[index0], f[index0+S[3]], u[3]) lerp(f[index0+S[2]], f[index0+S[2]+S[3]], u[3]) ;
        lerp(f[index0+S[1]], f[index0+S[1]+S[3]], u[3]) lerp(f[index0+S[1]+S[2]], f[index0+S[1]+S[2]+S[3]], u[3])
    ]

    return lerp(lerp(A[1,1], A[2,1], u[1]), lerp(A[1,2], A[2,2], u[1]), u[2])
end

@generated function generlLinearInterpolate(f::U1, x::U2, collision::Array{T, N}) where {N, T<:AbstractFloat, U1<:AbstractArray{T, N}, U2<:StaticVector{N, T}}
    quote
        c = round.(Int32, x)
        S = strides(collision)
        if collision[1 + sum(@. S * (c - 1))] <= 0
            return 0
        end

        x0 = floor.(Int32, x)
        u = x - x0

        index0 = 1 + sum(@. S * (x0 - 1))

        n = length(x)

        b = MVector{2<<(N-1), T}(undef)
    
        p = 1
        Base.Cartesian.@nloops $N i d0 -> (1:2) d -> j_d = ((i_d-1)*stride(collision, d)) begin
            index = index0 + Base.@ncall($N, +, j)
            b[p] = f[index]
            p+=1
        end

        for i in 1:N
            n = 2 << (N-i)
            for (j, k) in enumerate(1:2:n)
                b[j] = lerp(b[k], b[k+1], u[i])
            end
        end
            
        b[1]
    end
end

"""
    advectScalar(f, vel, collision, dx, dt)

Semi-Lagrangian advection of `f`.
"""
function advectScalar!(f::U1, vel::U2, collision::Array{T, N}, dx::Vector{T}, dt::T) where {T<:AbstractFloat, N, U1<:AbstractArray{T, N}, U2<:AbstractArray{T}}
    @assert size(collision) == size(f) == size(vel)[2:end]
    @assert ndims(f) == size(vel, 1) == length(dx)
    f_old = similar(f)
    copy!(f_old, f)
    
    n = ndims(f)
    
    Threads.@threads for I in CartesianIndices(f)
        if collision[I] > 0
            u = MVector{N, T}(undef)
            indomain = true
            for j = 1:n
                u[j] = I[j] - vel[j, I] * dt / dx[j]
                if 1 > u[j] || u[j] > size(f, j)
                    indomain = false
                    break
                end 
            end

            if indomain
                f[I] = domainInterpolate(f_old, u, collision)
            else
                f[I] = 0
            end
        end
    end
end

"""
    advectVector!(F, vel, collision, dx, dt)

Semi-Lagrangian advection of vector field `F`. Is same as advectScalar! but for vector fields.
"""
function advectVector!(F::U, vel::U, collision::Array{T, N}, dx::Vector{T}, dt::T) where {T<:AbstractFloat, N, NN, U<:AbstractArray{T, NN}}    
    n = size(F, 1)

    F_old = similar(F)
    copy!(F_old, F)
    
    Threads.@threads for I in CartesianIndices(size(F)[2:end])
        if collision[I] > 0
            u = MVector{N, T}(undef)
            indomain = true
            for j = 1:n
                u[j] = I[j] - vel[j, I] * dt / dx[j]
                if 1 > u[j] || u[j] > size(collision, j)
                    indomain = false
                    break
                end
            end

            if indomain
                for j = 1:n
                    F[j, I] = domainInterpolate(view(F_old, j, :, :), u, collision)
                end
            end
        end
    end
end

end # module