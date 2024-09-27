module FluidSolve

include("Fluids.jl")
include("Projection.jl")

using ..PressureSolve
using ..Advection

export timestepUpdate!, Fluid

"""
    timestepUpdate!(solver, fluid, dt)

Solve over timestep `dt`.
"""
function timestepUpdate!(solver, fluid, dt; maxIterations=solver.maxIterations)
    vel_old = similar(fluid.vel)
    copy!(vel_old, fluid.vel)
    Advection.advectVector!(fluid.vel, vel_old, fluid.collision, fluid.dx, dt)
    projectNonDivergent!(solver, fluid; maxIterations=maxIterations)
    Advection.advectScalar!(fluid.density, fluid.vel, fluid.collision, fluid.dx, dt)
end

# for compatibility.
function timestepUpdate!(fluid, dt; maxIterations=80)
    solver = PressureSolve.JacobiSolver{eltype(fluid.p), ndims(fluid.p)}(fluid.dx, size(fluid.p), maxIterations)
    timestepUpdate!(solver, fluid, dt)
end


end # module