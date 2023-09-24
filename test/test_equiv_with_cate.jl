using Polynomials4ML, StaticArrays, EquivariantModels, Test, Rotations, LinearAlgebra
using ACEbase.Testing: print_tf
using EquivariantModels: getspec1idx, _invmap, dropnames, SList, val2i, xx2AA, degord2spec

include("wigner.jl")

L = 4

Aspec, AAspec = degord2spec(; totaldegree = 4, 
                                  order = 2, 
                                  Lmax = 0, )
cats = [:O,:C]

ext(x,cats) = [ (x[i]..., s = cats) for i = 1:length(x)]
AAspec_tmp = [ ext.(AAspec,cats[1])..., ext.(AAspec,cats[2])... ] |> sort

luxchain, ps, st = equivariant_model(AAspec_tmp, L; categories=cats)
F(X) = luxchain(X, ps, st)[1]

@info("Testing the equivariance of chains that contain categorical basis")
for ntest = 1:10
   local X, θ1, θ2, θ3, Q, QX
   X = [ @SVector(rand(3)) for i in 1:10 ]
   θ1 = rand() * 2pi
   θ2 = rand() * 2pi
   θ3 = rand() * 2pi
   Q = RotXYZ(θ1, θ2, θ3)
   QX = [SVector{3}(x) for x in Ref(Q) .* X]
   
   print_tf(@test F(X)[1] ≈ F(QX)[1])

   for l = 2:L
      D = wigner_D(l-1,Matrix(Q))'
      # D = wignerD(l-1, 0, 0, θ)
      print_tf(@test norm.(Ref(D') .* F(X)[l] - F(QX)[l]) |> norm <1e-12)
   end
end

println()
