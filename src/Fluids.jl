
#lonely struct

mutable struct Fluid{T<:AbstractFloat, N, NN}
    dx::Vector{T}
    vel::Array{T, NN}
    collision::Array{T, N}
    p::Array{T, N}
    density::Array{T, N}
end