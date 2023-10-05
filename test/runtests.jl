using EquivariantModels
using Test

@testset "EquivariantModels.jl" begin
    @testset "CategoricalBasis" begin include("test_categorial.jl") end
    @testset "Equivariance" begin 
        include("test_equivariance.jl") 
        include("test_equiv_with_cate.jl")
        include("test_rSH_equivariance.jl")
    end
    @testset "Linear_Dependence" begin 
        include("test_linear_dependence.jl")
    end
end
