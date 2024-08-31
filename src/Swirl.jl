module Swirl

include("Advection.jl")
include("PressureSolve.jl")
include("FluidSolve.jl")

using .FluidSolve
using .PressureSolve
using .Advection

greet() = print("Hello World!")

end # module Swirl
