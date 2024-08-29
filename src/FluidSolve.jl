module FluidSolve

include("Fluids.jl")
include("PressureSolve.jl")
include("Advection.jl")
include("Projection.jl")

export timestepUpdate!, Fluid


function timestepUpdate!(fluid::Fluid, dt)
    vel_old = similar(fluid.vel)
    copy!(vel_old, fluid.vel)
    Advection.advectVector!(fluid.vel, vel_old, fluid.collision, fluid.dx, dt)
    projectNonDivergent!(fluid)
    Advection.advectScalar!(fluid.density, fluid.vel, fluid.collision, fluid.dx, dt)
end


end # module