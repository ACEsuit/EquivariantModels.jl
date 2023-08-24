using EquivariantModels
using Test

@testset "EquivariantModels.jl" begin
    @testset "Equivariance" begin include("test_equivariance.jl") end
end
