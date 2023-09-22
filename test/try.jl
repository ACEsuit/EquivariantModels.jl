using EquivariantModels
using StaticArrays
using Test
using ACEbase.Testing: print_tf
using Rotations, WignerD, BlockDiagonals
using LinearAlgebra
using ComponentArrays

L = 2
totdeg = 6
ν = 2
luxchain, ps, st = luxchain_constructor(totdeg,ν,L;islong = false)

cps = ComponentArray(ps)

D = wignerD(L, 0, 0, π / 2)
F(X) = luxchain(X, ps, st)[1]
X = [ @SVector(rand(3)) for i in 1:10 ]
θ = rand() * 2pi
Q = RotXYZ(0, 0, θ)
size(F(X))
size(D)



