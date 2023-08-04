using EquivariantModels
using StaticArrays
using Test
using ACEbase.Testing: print_tf
using Rotations, WignerD
using LinearAlgebra

@info("Testing single vector case")
totdeg = 5
ν = 3

for L = 0:4
   local F, luxchain, ps, st
   luxchain, ps, st = luxchain_constructor(totdeg,ν,L;islong = false)
   F(X) = luxchain(X, ps, st)[1]
   
   @info("Tesing L = $L O(3) equivariance")
   for _ = 1:30
      local X, θ, Q, QX
      X = [ @SVector(rand(3)) for i in 1:10 ]
      θ = rand() * 2pi
      Q = RotXYZ(0, 0, θ)
      # Q = rand_rot()
      QX = [SVector{3}(x) for x in Ref(Q) .* X]
      D = wignerD(L, 0, 0, θ)
      if L == 0
         print_tf(@test F(X) ≈ F(QX))
      else
         print_tf(@test F(X) ≈ Ref(D) .* F(QX))
      end
   end
   println()
end

# SYYVector case
totdeg = 5
ν = 1
L = 4
luxchain, ps, st = luxchain_constructor(totdeg,ν,L;islong = true)
local F
F(X) = luxchain(X, ps, st)[1]
@info("Tesing L = $L O(3) full equivariance")

for ntest = 1:20
   local X, θ, Q, QX
   X = [ @SVector(rand(3)) for i in 1:10 ]
   θ = rand() * 2pi
   Q = RotXYZ(0, 0, θ)
   QX = [SVector{3}(x) for x in Ref(Q) .* X]

   for l = 0:L
      D = wignerD(l, 0, 0, θ)
      print_tf(@test Ref(D) .* [F(QX)[i][Val(l)] for i = 1:length(F(X))] ≈ [F(X)[i][Val(l)] for i = 1:length(F(X))])
   end
end
println()

## seperated blocks
@info("Testing the chain that generates several blocks from a long vector")
totdeg = 5
ν = 1
L = 4
luxchain, ps, st = luxchain_constructor_multioutput(totdeg,ν,L)
F(X) = luxchain(X, ps, st)[1]

# A small comparison - long vector does give us some redundent basis...

@info("Equivariance test")
l1l2set = [(l1,l2) for l1 = 0:L for l2 = 0:L-l1]
for ntest = 1:2
   local X, θ, Q, QX
   X = [ @SVector(rand(3)) for i in 1:10 ]
   θ = rand() * 2pi
   Q = RotXYZ(0, 0, θ)
   QX = [SVector{3}(x) for x in Ref(Q) .* X]

   for (i,(l1,l2)) in enumerate(l1l2set)
      D1 = wignerD(l1, 0, 0, θ)
      D2 = wignerD(l2, 0, 0, θ)
      if F(X)[i] |> length ≠ 0
         print_tf(@test norm(Ref(D1') .* F(X)[i] .* Ref(D2) - F(QX)[i]) < 1e-8) 
      end
   end
end
println()
