module FluidSolve

include("Fluids.jl")
include("PressureSolve.jl")
include("Advection.jl")
include("Projection.jl")

export timestepUpdate!, Fluid

"""
    timestepUpdate!(fluid, dt)

Solve over timestep `dt`.
"""
function timestepUpdate!(fluid::Fluid, dt; solveMethod::PressureSolve.PressureSolveMethod=PressureSolve.JacobiMethod, maxIterations=80, ϵ=0.4)
    vel_old = similar(fluid.vel)
    copy!(vel_old, fluid.vel)
    Advection.advectVector!(fluid.vel, vel_old, fluid.collision, fluid.dx, dt)
    projectNonDivergent!(fluid, solveMethod=solveMethod; maxIterations=maxIterations, ϵ=ϵ)
    Advection.advectScalar!(fluid.density, fluid.vel, fluid.collision, fluid.dx, dt)
end


end # module