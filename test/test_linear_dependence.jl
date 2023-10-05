using EquivariantModels, StaticArrays, Test, Polynomials4ML, LinearAlgebra
using ACEbase.Testing: print_tf
using Rotations, WignerD, BlockDiagonals
using EquivariantModels: Radial_basis, xx2AA, degord2spec
using Polynomials4ML:lux
using RepLieGroups

include("wigner.jl")

@info("Testing the chain that generates a single B basis")
rcut = 5.5
totdeg = 6
ν = 3
Lmax = 2
fcut(rcut::Float64,pin::Int=2,pout::Int=2) = r -> (r < rcut ? abs( (r/rcut)^pin - 1)^pout : 0)
ftrans(r0::Float64=.0,p::Int=2) = r -> ( (1+r0)/(1+r) )^p
radial = EquivariantModels.simple_radial_basis(MonoBasis(totdeg-1),r->sqrt(r)*fcut(rcut,2,2)(r),r->1/sqrt(r)*ftrans(1.0,2)(r))

L = 0
Aspec, AAspec =  degord2spec(radial; totaldegree = totdeg, 
                                order = ν, 
                                Lmax = L, islong = false)
# luxchain, ps, st = xx2AA(AAspec, radial)
luxchain, ps, st = equivariant_model(AAspec, radial, L; islong = false)
F(X) = luxchain(X, ps, st)[1]

X = [ @SVector(rand(3)) for i in 1:10 ]
F(X)

T = L == 0 ? ComplexF64 : SVector{2L+1,ComplexF64}
A = zeros(T,length(F(X)),3length(F(X)))
for i = 1:3length(F(X))
    local x = [ @SVector(rand(3)) for i in 1:10 ]
    A[:,i] = F(x)
end
B = 
try 
    A*A'
catch
    B = zeros(ComplexF64,length(F(X)),length(F(X)))
    for i = 1:length(F(X))
        for j = 1:length(F(X))
            B[i,j] = sum(RepLieGroups.O3.coco_dot(A[i,t], A[j,t]) for t = 1:3length(F(X)))
        end
    end
end
rank(B)