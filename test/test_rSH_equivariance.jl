using EquivariantModels, StaticArrays, Test, Polynomials4ML, LinearAlgebra
using ACEbase.Testing: print_tf
using Rotations, WignerD, BlockDiagonals
# using EquivariantModels: Radial_basis
# using Polynomials4ML:lux

include("wigner.jl")

@info("Testing the chain that generates a single B basis")
totdeg = 6
ν = 2
Lmax = 0
basis = legendre_basis(totdeg)
radial = EquivariantModels.simple_radial_basis(basis)

luxchain, ps, st = equivariant_model(totdeg, ν, radial, 0;islong = false, rSH = true)
F(X) = luxchain(X, ps, st)[1]

for L = 0:Lmax
   local F, luxchain, ps, st, F2, luxchain2, ps2, st2
   luxchain, ps, st = equivariant_model(totdeg, ν, radial, L;islong = false, rSH = true)
   F(X) = luxchain(X, ps, st)[1]
   
   luxchain2, ps2, st2 = equivariant_model(EquivariantModels.degord2spec(radial;totaldegree=totdeg,order=ν,Lmax=L,islong = true,rSH = true)[2][1:end-1],radial,L;islong = false,rSH = true)
   F2(X) = luxchain(X, ps2, st2)[1]
   
   @info("Tesing L = $L O(3) equivariance")
   for _ = 1:30
      local X, θ1, θ2, θ3, Q, QX
      X = [ @SVector(rand(3)) for i in 1:10 ]
      θ1 = rand() * 2pi
      θ2 = rand() * 2pi
      θ3 = rand() * 2pi
      Q = RotXYZ(θ1, θ2, θ3)
      # Q = rand_rot()
      # QX = [SVector{3}(x) for x in Ref(Q) .* X]
      QX = [ State(rr = Q * X[i]) for i in 1:length(X) ]
      X = [ State(rr = X[i]) for i in 1:length(X) ]
      D = wigner_D(L,Matrix(Q))'
      # D = wignerD(L, θ, θ, θ)

      print_tf(@test F(X) ≈ F(QX))
      
   end
   println()
   
   @info("Tesing consistency between the two ways of input - in particular the ``closure'' of specifications")
   for _ = 1:30
      local X
      X = [ @SVector(rand(3)) for i in 1:10 ]
      X = [ State(rr = X[i]) for i in 1:length(X) ]
      print_tf(@test F(X) ≈ F2(X))
   end
   println()
   
end

@info("Testing the chain that generates all B bases")
totdeg = 6
ν = 2
L = Lmax
basis = legendre_basis(totdeg)
radial = EquivariantModels.simple_radial_basis(basis)
luxchain, ps, st = equivariant_model(totdeg,ν,radial,L;islong = true)
F(X) = luxchain(X, ps, st)[1]
luxchain2, ps2, st2 = equivariant_model(EquivariantModels.degord2spec(radial;totaldegree=totdeg,order=ν,Lmax=L,islong = true)[2][1:end-1],radial,L;islong = true,rSH = true)
F2(X) = luxchain(X, ps2, st2)[1]

for ntest = 1:10
   local X, θ1, θ2, θ3, Q, QX
   X = [ @SVector(rand(3)) for i in 1:10 ]
   θ1 = rand() * 2pi
   θ2 = rand() * 2pi
   θ3 = rand() * 2pi
   Q = RotXYZ(θ1, θ2, θ3)
   # QX = [SVector{3}(x) for x in Ref(Q) .* X]
   QX = [ State(rr = Q * X[i]) for i in 1:length(X) ]
   X = [ State(rr = X[i]) for i in 1:length(X) ]
   
   print_tf(@test F(X)[1] ≈ F(QX)[1])
end
println()

@info("Consistency check")
totdeg = 6
ν = 2
L = Lmax
basis = legendre_basis(totdeg)
radial = EquivariantModels.simple_radial_basis(basis)
luxchain, ps, st = equivariant_model(totdeg,ν,radial,L;islong = true, rSH = true);
F(X) = luxchain(X, ps, st)[1]

for l = 0:Lmax
   @info("Consistency check for L = $l")
   local FF, luxchain, ps, st
   luxchain, ps, st = equivariant_model(totdeg,ν,radial,l;islong = false, rSH = true)
   FF(X) = luxchain(X, ps, st)[1]
   
   for ntest = 1:20
      X = [ @SVector(rand(3)) for i in 1:10 ]
      X = [ State(rr = X[i]) for i in 1:length(X) ]
      print_tf(@test F(X)[l+1] == FF(X))
   end
   println()
end

@info("Tesing consistency between the two ways of input - in particular the ``closure'' of specifications")
for _ = 1:10
   local X
   X = [ @SVector(rand(3)) for i in 1:10 ]
   X = [ State(rr = X[i]) for i in 1:length(X) ]
   print_tf(@test length(F(X)) == length(F2(X)) && all([F(X)[i] ≈ F2(X)[i] for i = 1:length(F(X))]))
end
println()

@info("Tesing the last way of input - given n_list and l_list")

for L = 0:Lmax
   local F, luxchain, ps, st, nn, ll
   
   nn = rand(0:2,4)
   ll = rand(0:2,4)
   while iseven(L) != iseven(sum(ll))
      ll = rand(0:2,4)
   end
   luxchain, ps, st = equivariant_model(nn,ll,radial,L;islong = false, rSH = true)
   F(X) = luxchain(X, ps, st)[1]
   
   @info("Tesing L = $L O(3) equivariance")
   for _ = 1:30
      local X, θ, Q, QX
      X = [ @SVector(rand(3)) for i in 1:10 ]
      θ = rand() * 2pi
      Q = RotXYZ(0, 0, θ)
      # Q = rand_rot()
      # QX = [SVector{3}(x) for x in Ref(Q) .* X]
      QX = [ State(rr = Q * X[i]) for i in 1:length(X) ]
      X = [ State(rr = X[i]) for i in 1:length(X) ]
      D = wignerD(L, 0, 0, θ)
      if length(F(X)) == 0 
         continue
      end
      print_tf(@test F(X) ≈ F(QX))
      
   end
   println()
end

