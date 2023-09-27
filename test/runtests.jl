using EquivariantModels
using Test

@testset "EquivariantModels.jl" begin
    @testset "CategoricalBasis" begin include("test_categorial.jl") end
    @testset "Equivariance" begin include("test_equivariance.jl"); include("test_equiv_with_cate.jl"); end
end
