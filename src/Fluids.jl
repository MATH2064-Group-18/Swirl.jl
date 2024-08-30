
#lonely struct

"""
Stores fluid stuff.

⚠ Warning: Currently the domain must be closed
make sure that the input for collision is solid
on array bounds otherwise will segmentation fault.
"""
mutable struct Fluid{T<:AbstractFloat, N, NN}
    dx::Vector{T}
    vel::Array{T, NN}
    collision::Array{T, N}
    p::Array{T, N}
    density::Array{T, N}
end