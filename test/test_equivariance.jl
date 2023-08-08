using EquivariantModels
using StaticArrays
using Test
using ACEbase.Testing: print_tf
using Rotations, WignerD, BlockDiagonals
using LinearAlgebra

@info("Testing single vector case")
totdeg = 6
ν = 2

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

@info("Testing SYYVector case")
totdeg = 6
ν = 2
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
   D = BlockDiagonal([ wignerD(l, 0, 0, θ) for l = 0:L] )
   
   print_tf(@test Ref(D) .* F(QX) ≈ F(X))
end
println()

## equivariant blocks
@info("Testing the chain that generates several equivariant blocks from a long vector")
totdeg = 6
ν = 2
L = 4
luxchain, ps, st = equivariant_luxchain_constructor(totdeg,ν,L)
F(X) = luxchain(X, ps, st)[1]

# A small comparison - long vector does give us some redundent basis...

@info("Equivariance test")
l1l2set = [(l1,l2) for l1 = 0:L for l2 = 0:L-l1]
for ntest = 1:10
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

## A second way - construct B^0, B^1, ..., B^L first
@info("Testing the chain that generates all the B bases")
totdeg = 6
ν = 2
L = 4
luxchain, ps, st = luxchain_constructor_multioutput(totdeg,ν,L)
F(X) = luxchain(X, ps, st)[1]

for ntest = 1:10
   local X, θ, Q, QX
   X = [ @SVector(rand(3)) for i in 1:10 ]
   θ = rand() * 2pi
   Q = RotXYZ(0, 0, θ)
   QX = [SVector{3}(x) for x in Ref(Q) .* X]
   
   print_tf(@test F(X)[1] ≈ F(QX)[1])

   for l = 2:L
      D = wignerD(l-1, 0, 0, θ)
      print_tf(@test norm.(Ref(D') .* F(X)[l] - F(QX)[l]) |> norm <1e-8)
   end
end
println()

@info("Consistency check")
totdeg = 6
ν = 2
L = 4
luxchain, ps, st = luxchain_constructor_multioutput(totdeg,ν,L);
F(X) = luxchain(X, ps, st)[1]

for l = 0:4
   @info("Consistency check for L = $l")
   local FF, luxchain, ps, st
   luxchain, ps, st = luxchain_constructor(totdeg,ν,l;islong = false)
   FF(X) = luxchain(X, ps, st)[1]
   
   for ntest = 1:20
      X = [ @SVector(rand(3)) for i in 1:10 ]
      print_tf(@test F(X)[l+1] == FF(X))
   end
   println()
end


# ##
# using EquivariantModels:cgmatrix
# 
# totdeg = 6
# ν = 2
# L = 3
# luxchain, ps, st = luxchain_constructor_multioutput(totdeg,ν,L);
# F(X) = luxchain(X, ps, st)[1]
# X = [ @SVector(rand(3)) for i in 1:10 ]
# θ = rand() * 2pi
# Q = RotXYZ(0, 0, θ)
# D = wignerD(1, 0, 0, θ)
# D2 = wignerD(2, 0, 0, θ)
# QX = [SVector{3}(x) for x in Ref(Q) .* X]
# vals = F(X)
# Qvals = F(QX)
# 
# for ntest = 1:20
#    ii = [ rand(1:length(vals[i])) for i = 1:length(vals) ]
# 
#    vec1 = [ 0; vals[2][ii[2]]; zeros(5); vals[4][ii[4]] ]
#    B1 = reshape(cgmatrix(1,2) * vec1,3,5)
#    vec2 = [ 0; Qvals[2][ii[2]]; zeros(5); Qvals[4][ii[4]] ]
#    B2 = reshape(cgmatrix(1,2) * vec2,3,5)
#    @assert D' * B1 * D2 - B2 |> norm <1e-12
# end
# 
# cgmatrix(1,2)
# 
