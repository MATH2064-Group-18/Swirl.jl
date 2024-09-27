"""
    projectNonDivergent!(solver, vel, p, collision, dx; maxIterations=solver.maxIterations)

Removes divergence in `vel`.


Solves the pressure Poisson equation ∇²p = ∇⋅vel, 
and removing the divergence in `vel` by updating
it to vel = vel - ∇p.

Note: `p` is not the true pressure (we call it the pseudo-pressure), it is proportional
to it by the factor p_true = p_pseudo * Δt / ρ.
"""
function projectNonDivergent!(
    solver, 
    vel, 
    p, 
    collision, 
    dx;
    maxIterations = solver.maxIterations
)
    @assert size(p) == size(collision)
    n = length(p)
    d = ndims(p)

    v_div = zeros(eltype(p), size(p))
    
    s0 = size(vel, 1)
    for i = 1:n
        if collision[i] > 0
            for j = 1:d
                s = stride(p, j)
                
                b1 = collision[i - s] > 0 ? vel[j + s0*(i - s-1)] : 2 * vel[j + s0*(i - s-1)] - vel[j + s0 * (i-1)]
                b2 = collision[i + s] > 0 ? vel[j + s0*(i + s-1)] : 2 * vel[j + s0*( i + s -1)] - vel[j + s0 * (i-1)]

                v_div[i] += (b2 - b1) * 0.5 / dx[j]
            end
        end
    end

    pressureSolveInfo = PressureSolve.poissonSolve!(solver, p, v_div, collision; maxIterations=maxIterations)


    for i = 1:n
        if collision[i] > 0
            for j = 1:d
                strid = stride(p, j)
                a1 = collision[i - strid] > 0 ? p[i - strid] : p[i]
                a2 = collision[i + strid] > 0 ? p[i + strid] : p[i]

                vel[j + s0 * (i-1)] -= (a2 - a1) / (2 * dx[j])
            end
        end
    end

    return pressureSolveInfo
end

function projectNonDivergent!(solver, fluid; maxIterations=solver.maxIterations)
    projectNonDivergent!(solver, fluid.vel, fluid.p, fluid.collision, fluid.dx; maxIterations=maxIterations)
end

# for compatibility
function projectNonDivergent!(fluid; maxIterations=80)
    solver = PressureSolve.JacobiSolver{eltype(fluid.p), ndims(fluid.p)}(fluid.dx, size(fluid.p), maxIterations)
    projectNonDivergent!(solver, fluid)
end
