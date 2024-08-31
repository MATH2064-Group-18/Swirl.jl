
#lonely struct

"""
Stores fluid stuff.

⚠ Warning: Currently the domain must be closed
make sure that the input for collision is solid
on array bounds otherwise will segmentation fault.

Note: `p` is not the true pressure (we call it the pseudo-pressure), it is proportional
to it by the factor p_true = p_pseudo * Δt / ρ. And density is not the mass, but the visual
density.
"""
mutable struct Fluid{T<:AbstractFloat, N, NN}
    dx::Vector{T}
    vel::Array{T, NN}
    collision::Array{T, N}
    p::Array{T, N}
    density::Array{T, N}
end