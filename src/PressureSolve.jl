module PressureSolve

using LinearAlgebra

struct PressureSolveInfo
    iterations
    solve_time
    residual_norm
end

function jacobi!(f, f_old, g, collision, dx, maxIterations; res_history=nothing)
    @assert size(f) == size(f_old) == size(g) == size(collision)
    @assert ndims(f) == ndims(f_old) == ndims(g) == ndims(collision) == length(dx)
    
    dxn2 = @. (one(dx) / dx)^2
    c0 = 0.5 / sum(dxn2)
    c = c0 * dxn2

    n = length(f)

    t1 = time()

    for iter = 1:maxIterations
        copy!(f_old, f)
        for i = 1:n
            @inbounds if collision[i] > 0
                A = 0
                for (j, strid) in enumerate(strides(f))
                    a1 = collision[i-strid] > 0 ? f_old[i-strid] : f_old[i]
                    a2 = collision[i+strid] > 0 ? f_old[i+strid] : f_old[i]
                    A += c[j] * (a1 + a2)
                end
                f[i] = A - c0 * g[i]
            end
        end

        if !isnothing(res_history)
            push!(res_history, residualNorm(f, g, collision, dxn2))
        end
    end

    t2 = time()
    
    return PressureSolveInfo(maxIterations, t2-t1, residualNorm(f, g, collision, dxn2))
end

function gaussSeidel!(f, g, collision, dx, maxIterations; res_history=nothing)
    @assert size(f) == size(g) == size(collision)
    @assert ndims(f) == ndims(g) == ndims(collision) == length(dx)
    
    dxn2 = @. (one(dx) / dx)^2
    c0 = 0.5 / sum(dxn2)
    c = c0 * dxn2

    n = length(f)

    t1 = time()

    for iter = 1:maxIterations
        for i = 1:n
            @inbounds if collision[i] > 0
                A = 0
                for (j, strid) in enumerate(strides(f))
                    a1 = collision[i-strid] > 0 ? f[i-strid] : f[i]
                    a2 = collision[i+strid] > 0 ? f[i+strid] : f[i]
                    A += c[j] * (a1 + a2)
                end
                f[i] = A - c0 * g[i]
            end
        end

        if !isnothing(res_history)
            push!(res_history, residualNorm(f, g, collision, dxn2))
        end
    end

    t2 = time()

    return PressureSolveInfo(maxIterations, t2-t1, residualNorm(f, g, collision, dxn2))
end

function conjugateGradient!(f, g, collision, dx, maxIterations, ϵ; res_history=nothing)
    @assert size(f) == size(g) == size(collision)
    @assert ndims(f) == ndims(g) == ndims(collision) == length(dx)

    dxn2 = @. (one(dx) / dx)^2

    n = length(f)
    
    v = similar(f)
    p = similar(f)
    r = zeros(eltype(f), size(f))

    t1 = time()

    for i in eachindex(f)
        if collision[i] > 0
            r[i] = g[i]
            for (j, s) in enumerate(strides(f))
                a1 = collision[i-s] > 0 ? f[i-s] : f[i]
                a2 = collision[i+s] > 0 ? f[i+s] : f[i]
                r[i] -= (a1 + a2 - 2 * f[i]) * dxn2[j]
            end
        end
    end

    copy!(p, r)
    
    res_sum = sum(abs2, r)
    tol = (ϵ * norm(g))^2
    iter = 0
    while iter < maxIterations && tol < res_sum
        for i in eachindex(collision)
            v[i] = 0
            @inbounds if collision[i] > 0
                for (j, s) in enumerate(strides(f))
                    a1 = collision[i-s] > 0 ? p[i-s] : p[i]
                    a2 = collision[i+s] > 0 ? p[i+s] : p[i]
                    v[i] += (a1 + a2 - 2 * p[i]) * dxn2[j]
                end
            end
        end
        
        α = res_sum / dot(p, v)
        
        # f = f + α * p
        axpy!(α, p, f)
        
        
        # r = r - α * v
        axpy!(-α, v, r)
        
        res_sum_old = res_sum
        res_sum = sum(abs2, r)
        
        β = res_sum / res_sum_old

        # p = r + β * p
        axpby!(1, r, β, p)
        
        iter += 1
        if !isnothing(res_history)
            push!(res_history, sqrt(res_sum))
        end
    end

    t2 = time()
    
    return PressureSolveInfo(iter,  t2 - t1, sqrt(res_sum))
end

function residualNorm(f, g, collision, dxn2)
    res_sum = 0
    for i in eachindex(f)
        if collision[i] > 0
            res = g[i]
            for (j, stride_) in enumerate(strides(f)) 
                a1 = collision[i-stride_] > 0 ? f[i-stride_] : f[i]
                a2 = collision[i+stride_] > 0 ? f[i+stride_] : f[i]
                res -= (a1 + a2 - 2 * f[i]) * dxn2[j]
            end
            res_sum += res*res
        end
    end
    return sqrt(res_sum)
end

end # module