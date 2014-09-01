type SPLDA{T<:FloatingPoint}
    D::Matrix{T}
    V::Matrix{T}
    μ::Vector{T}
#    SPLDA(D::Matrix{T}, V::Matrix{T}) = new(copy(D), copy(V))
end

copy(m::SPLDA) = SPLDA(Base.copy(m.D), Base.copy(m.V), Base.copy(m.μ))
