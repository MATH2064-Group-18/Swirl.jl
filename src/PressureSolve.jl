module PressureSolve

using LinearAlgebra

export jacobi!, gaussSeidel!, conjugateGradient!, preconditionedConjugateGradient!, PressureSolveInfo, PressureSolveMethod

struct PressureSolveInfo
    iterations
    solve_time
    residual_norm # 2-norm
end

@enum PressureSolveMethod begin
    JacobiMethod
    GaussSeidelMethod
    ConjugateGradientMethod
    PreconditionedConjugateGradientMethod
end


abstract type PressureSolver{T<:AbstractFloat, N} end

mutable struct JacobiSolver{T<:AbstractFloat, N} <: PressureSolver{T, N}
    dx::Vector{T} #maybe make static vector
    maxIterations::Int
    f_old::Array{T, N}
    function JacobiSolver{T, N}(dx::Vector{T}, dims::Dims{N}, maxIterations::Int) where {T<:AbstractFloat, N}
        @assert length(dx) == length(dims)
        new(dx, maxIterations, Array{T, N}(undef, dims))
    end
end

mutable struct GaussSeidelSolver{T<:AbstractFloat, N} <: PressureSolver{T, N}
    dx::Vector{T}
    maxIterations::Int
    function GaussSeidelSolver{T, N}(dx::Vector{T}, dims::Dims{N}, maxIterations::Int) where {T<:AbstractFloat, N}
        @assert length(dx) == length(dims)
        new(dx, maxIterations)
    end
end

mutable struct ConjugateGradientSolver{T<:AbstractFloat, N} <: PressureSolver{T, N}
    dx::Vector{T}
    maxIterations::Int
    ϵ::T
    use_preconditioner::Bool
    L_diag_rcp::Array{T, N}
    p::Array{T, N}
    r::Array{T, N}
    v::Array{T, N}
    w::Array{T, N}
    z::Array{T, N}
    function ConjugateGradientSolver{T, N}(dx::Vector{T}, dims::Dims{N}, maxIterations::Int, ϵ::T, use_preconditioner::Bool) where {T <: AbstractFloat, N}
        @assert length(dx) == length(dims)
        empty_array = Array{T, N}(undef, dims .* 0)
        if use_preconditioner
            return new(dx, maxIterations, ϵ, use_preconditioner, Array{T, N}(undef, dims), Array{T, N}(undef, dims), Array{T, N}(undef, dims), Array{T, N}(undef, dims), Array{T, N}(undef, dims), Array{T, N}(undef, dims))
        end
        return new(dx, maxIterations, ϵ, use_preconditioner, similar(empty_array), Array{T, N}(undef, dims), Array{T, N}(undef, dims), Array{T, N}(undef, dims), similar(empty_array), similar(empty_array))
    end
end

"""
    jacobi!(f, f_old, g, collision, dx, maxIterations; res_history)

Solve Poisson equation ∇²f = g using Jacobi method.
"""
function jacobi!(f, f_old, g, collision, dx, maxIterations; res_history=nothing)
    @assert size(f) == size(f_old) == size(g) == size(collision)
    @assert ndims(f) == ndims(f_old) == ndims(g) == ndims(collision) == length(dx)
    
    dxn2 = @. (one(dx) / dx)^2
    c0 = convert(eltype(f), 0.5) / sum(dxn2)
    c = c0 * dxn2

    n = length(f)

    t1 = time()

    for iter = 1:maxIterations
        copy!(f_old, f)
        Threads.@threads for i = 1:n
            # Ain't got no time for bounds checks
            @inbounds if collision[i] > 0
                A = zero(eltype(f))
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

"""
    gaussSeidel!(f, g, collision, dx, maxIterations; res_history)

Solve Poisson equation ∇²f = g using Gauss-Seidel method.
"""
function gaussSeidel!(f, g, collision, dx, maxIterations; res_history=nothing)
    @assert size(f) == size(g) == size(collision)
    @assert ndims(f) == ndims(g) == ndims(collision) == length(dx)
    
    dxn2 = @. (one(dx) / dx)^2
    c0 = convert(eltype(f), 0.5) / sum(dxn2)
    c = c0 * dxn2

    n = length(f)

    t1 = time()

    for iter = 1:maxIterations
        for i = 1:n
            @inbounds if collision[i] > 0
                A = zero(eltype(f))
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

"""
    conjugateGradient!(f, g, collision, dx, maxIterations, ϵ; res_history)

Solve Poisson equation ∇²f = g using conjugate gradient method. 

"""
function conjugateGradient!(p, r, v, f, g, collision, dx, maxIterations, ϵ; res_history=nothing)
    @assert size(f) == size(g) == size(collision)
    @assert ndims(f) == ndims(g) == ndims(collision) == length(dx)

    dxn2 = @. (one(dx) / dx)^2

    n = length(f)
    

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
        Threads.@threads for i in eachindex(collision)
            @inbounds v[i] = 0
            @inbounds if collision[i] > 0
                for (j, s) in enumerate(strides(f))
                    a1 = collision[i-s] > 0 ? p[i-s] : p[i]
                    a2 = collision[i+s] > 0 ? p[i+s] : p[i]
                    v[i] += (a1 + a2 - 2 * p[i]) * dxn2[j]
                end
            end
        end
        
        α = res_sum / dot(p, v)
        
        @inbounds Threads.@threads for i in eachindex(f)
            f[i] += α * p[i]
            r[i] -= α * v[i]
        end
        
        
        res_sum_old = res_sum
        res_sum = dot(r, r)
        
        β = res_sum / res_sum_old

        # p = r + β * p
        @inbounds Threads.@threads for i in eachindex(f)
            p[i] = r[i] + β * p[i]
        end
        
        iter += 1
        if !isnothing(res_history)
            push!(res_history, sqrt(res_sum))
        end
    end

    t2 = time()
    
    return PressureSolveInfo(iter,  t2 - t1, sqrt(res_sum))
end

function conjugateGradient!(f, g, collision, dx, maxIterations, ϵ; res_history=nothing)
    v = similar(f)
    p = similar(f)
    r = zeros(eltype(f), size(f))
    conjugateGradient!(p, r, v, f, g, collision, dx, maxIterations, ϵ; res_history=res_history)
end

"""
    applyPreconditioner!(z, w, r, L_diag, collision, dxn2)

Apply incomplete Cholesky preconditioner.
"""
function applyPreconditioner!(z, w, r, L_diag_rcp, collision, dxn2)
    
    # solve Lw = r

    n = length(z)

    w[1] = collision[1] > 0 ? r[1] * L_diag_rcp[1] : 0

    # Forward substition
    @inbounds for i in eachindex(w)
        if collision[i] > 0
            w[i] = r[i]
            for (j, s) in enumerate(strides(w))
                if checkbounds(Bool, collision, i - s) && collision[i - s] > 0
                    w[i] += dxn2[j] * w[i - s] * L_diag_rcp[i-s]
                end
            end
            w[i] *= L_diag_rcp[i]
        else
            w[i] = 0
        end
    end

    # solve Lᵀz = w

    z[end] = collision[end] > 0 ? w[end] * L_diag_rcp[end] : 0 

    # Back substitution
    @inbounds for i in reverse(eachindex(z))
        if collision[i] > 0
            z[i] = w[i]
            for (j, s) in enumerate(strides(z))
                if (i + s) <= n && collision[i + s] > 0
                    z[i] += dxn2[j] * z[i + s] * L_diag_rcp[i]
                end
            end
            z[i] *= L_diag_rcp[i]
        else
            z[i] = 0
        end
    end
end

"""
    preconditionedConjugateGradient!(f, g, collision, dx, maxIterations, ϵ; res_history)

Solve Poisson equation ∇²f = g, using incomplete Cholesky preconditioned conjugate gradient method. 
"""
function preconditionedConjugateGradient!(L_diag_rcp, p, r, v, w, z, f, g, collision, dx, maxIterations, ϵ=0; res_history=nothing)
    @assert size(f) == size(g) == size(collision) == size(L_diag_rcp) == size(p) == size(r) == size(v) == size(w) == size(z)
    @assert ndims(f) == ndims(g) == ndims(collision) == length(dx)

    dxn2 = @. (one(dx) / dx)^2
    c0 = 2 * sum(dxn2)

    
    t1 = time()


    L_diag_rcp[1] = collision[1] <= 0 ? 0 : 1 / sqrt(c0)
    for i in eachindex(L_diag_rcp)
        if collision[i] > 0
            q = zero(eltype(f))
            A = c0
            for (j, s) in enumerate(strides(f))
                if checkbounds(Bool, L_diag_rcp, i - s) && collision[i-s] > 0
                    q += (-dxn2[j] * L_diag_rcp[i - s])^2
                else
                    A += -dxn2[j]
                end
                A += checkbounds(Bool, L_diag_rcp, i + s) && collision[i+s] > 0 ? 0 : -dxn2[j] 
            end
            L_diag_rcp[i] = 1 / sqrt(A - q)
        else
            L_diag_rcp[i] = 0
        end
    end





    for i in eachindex(f)
        if collision[i] > 0
            r[i] = -g[i]
            for (j, s) in enumerate(strides(f))
                a1 = collision[i-s] > 0 ? f[i-s] : f[i]
                a2 = collision[i+s] > 0 ? f[i+s] : f[i]
                r[i] -= -(a1 + a2 - 2 * f[i]) * dxn2[j]
            end
        end
    end


    applyPreconditioner!(z, w, r, L_diag_rcp, collision, dxn2)
    
    copy!(p, z)

    res_sum = sum(abs2, r)
    r_dot_z = dot(r, z)
    tol = (ϵ * norm(g))^2
    iter = 0
    while iter < maxIterations && tol < res_sum
        Threads.@threads for i in eachindex(collision)
            @inbounds v[i] = 0
            @inbounds if collision[i] > 0
                for (j, s) in enumerate(strides(f))
                    a1 = collision[i-s] > 0 ? p[i-s] : p[i]
                    a2 = collision[i+s] > 0 ? p[i+s] : p[i]
                    v[i] += -(a1 + a2 - 2 * p[i]) * dxn2[j]
                end
            end
        end
        
        α = r_dot_z / dot(p, v)
        
        # f = f + α * p
        f_update_task = Threads.@spawn axpy!(α, p, f)
        
        
        # r = r - α * v
        axpy!(-α, v, r)
        
        applyPreconditioner!(z, w, r, L_diag_rcp, collision, dxn2)
        res_sum = dot(r, r)
        r_dot_z_old = r_dot_z
        r_dot_z = dot(r, z)
        
        β = r_dot_z / r_dot_z_old

        wait(f_update_task)
        
        # p = z + β * p
        axpby!(1, z, β, p)
        
        iter += 1
        if !isnothing(res_history)
            push!(res_history, sqrt(res_sum))
        end
    end

    t2 = time()
    
    return PressureSolveInfo(iter,  t2 - t1, sqrt(res_sum))
end

"""
    residualNorm(f, g, collision, dxn2)

Compute 2-norm of residual.
"""
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

function preconditionedConjugateGradient!(f, g, collision, dx, maxIterations, ϵ=0; res_history=nothing)
    v = similar(f)
    p = similar(f)
    r = zeros(eltype(f), size(f))
    z = similar(r)

    w = similar(v)
    
    L_diag_rcp = similar(w)

    preconditionedConjugateGradient!(L_diag_rcp, p, r, v, w, z, f, g, collision, dx, maxIterations, ϵ; res_history=res_history)
end


function pressureSolve!(solver::GaussSeidelSolver{T, N}, f::Array{T, N}, g::Array{T, N}, collision::Array{T, N}, maxIterations; res_history=res_history) where {T<:AbstractFloat, N}
    return gaussSeidel!(f, g, collision, solver.dx, maxIterations; res_history=res_history)
end
function pressureSolve!(solver::GaussSeidelSolver{T, N}, f::Array{T, N}, g::Array{T, N}, collision::Array{T, N}; res_history=res_history) where {T<:AbstractFloat, N}
    return pressureSolve!(solver, f, g, collision, solver.maxIterations; res_history=res_history)
end

function pressureSolve!(solver::JacobiSolver{T, N}, f::Array{T, N}, g::Array{T, N}, collision::Array{T, N}, maxIterations; res_history=res_history) where {T<:AbstractFloat, N}
    return jacobi!(f, solver.f_old, g, collision, solver.dx, maxIterations; res_history=res_history)
end
function pressureSolve!(solver::JacobiSolver{T, N}, f::Array{T, N}, g::Array{T, N}, collision::Array{T, N}; res_history=res_history) where {T<:AbstractFloat, N}
    return pressureSolve!(solver, f, g, collision, solver.maxIterations; res_history=res_history)
end

function pressureSolve!(solver::ConjugateGradientSolver{T, N}, f::Array{T, N}, g::Array{T, N}, collision::Array{T, N}, maxIterations, ϵ; res_history=res_history) where {T<:AbstractFloat, N}
    if solver.use_preconditioner
        return preconditionedConjugateGradient!(solver.L_diag_rcp, solver.p, solver.r, solver.v, solver.w, solver.z, f, g, collision, solver.dx, maxIterations, ϵ; res_history=res_history)
    end
    return conjugateGradient!(solver.p, solver.r, solver.f, f, g, collision, solver.dx, maxIterations, ϵ; res_history=res_history)
end
function pressureSolve!(solver::ConjugateGradientSolver{T, N}, f::Array{T, N}, g::Array{T, N}, collision::Array{T, N}; res_history=res_history) where {T<:AbstractFloat, N}
    return pressureSolve!(solver, f, g, collision, solver.maxIterations, solver.ϵ; res_history=res_history)
end

end # module