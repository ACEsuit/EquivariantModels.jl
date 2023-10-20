using Polynomials4ML, StaticArrays, EquivariantModels, Test, Rotations, LinearAlgebra
using ACEbase.Testing: print_tf
using EquivariantModels: getspec1idx, _invmap, dropnames, SList, val2i, xx2AA, degord2spec, simple_radial_basis
using Polynomials4ML: lux
using DecoratedParticles

include("wigner.jl")

L = 4
totdeg = 4
ord = 2
radial = simple_radial_basis(legendre_basis(totdeg))
# radial = Radial_basis(legendre_basis(totdeg) |> lux)
Aspec, AAspec = degord2spec(radial; totaldegree = totdeg, 
                                  order = ord, 
                                  Lmax = 0, )
cats = [:O,:C]
cats_ext = [(:O,:C),(:C,:O),(:O,:O),(:C,:C)] |> unique
AAspec_tmp = []
for i = 1:length(AAspec)
   push!(AAspec_tmp, [ (spec..., s = cats_ext[1]) for spec in AAspec[i] ])
   push!(AAspec_tmp, [ (spec..., s = cats_ext[2]) for spec in AAspec[i] ])
end
pos = findall(x -> length(x)>1, AAspec)
_AAspec_tmp = [ [(AAspec[i][1]..., s = cats_ext[1]), (AAspec[i][2]..., s = cats_ext[2])] for i in pos ]
_AAspec_tmp2 = [ [(AAspec[i][1]..., s = cats_ext[2]), (AAspec[i][2]..., s = cats_ext[1])] for i in pos ]
append!(AAspec_tmp,_AAspec_tmp)
append!(AAspec_tmp,_AAspec_tmp2)

luxchain, ps, st = equivariant_model(AAspec_tmp, radial, L; categories=cats_ext)
F(X) = luxchain(X, ps, st)[1]
species = [ rand(cats) for i = 1:10 ]
Species = [ (species[1], species[i]) for i = 1:10 ]

@info("Testing the equivariance of chains that contain categorical basis")
for ntest = 1:10
   local X, θ1, θ2, θ3, Q, QX
   X = [ @SVector(rand(3)) for i in 1:10 ]
   XX = [State(rr = X[i], Zi = Species[i][1], Zj = Species[i][2]) for i = 1:length(X)]
   θ1 = rand() * 2pi
   θ2 = rand() * 2pi
   θ3 = rand() * 2pi
   Q = RotXYZ(θ1, θ2, θ3)
   QXX = [State(rr = Q * X[i], Zi = Species[i][1], Zj = Species[i][2]) for i = 1:length(X)]
   # QXX = [QX, Species]
   
   print_tf(@test F(XX)[1] ≈ F(QXX)[1])

   for l = 2:L
      D = wigner_D(l-1,Matrix(Q))'
      # D = wignerD(l-1, 0, 0, θ)
      print_tf(@test Ref(D') .* F(XX)[l] ≈ F(QXX)[l])
   end
end

println()
