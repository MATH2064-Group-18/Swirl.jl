module Swirl

include("Advection.jl")
include("PressureSolve.jl")
include("FluidSolve.jl")

using .FluidSolve

greet() = print("Hello World!")

end # module Swirl
