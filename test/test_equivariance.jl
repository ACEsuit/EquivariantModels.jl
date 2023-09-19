using EquivariantModels
using StaticArrays
using Test
using ACEbase.Testing: print_tf
using Rotations, WignerD, BlockDiagonals
using LinearAlgebra

include("wigner.jl")

@info("Testing the chain that generates a single B basis")
totdeg = 6
ν = 2
Lmax = 4

for L = 0:Lmax
   local F, luxchain, ps, st, F2, luxchain2, ps2, st2
   luxchain, ps, st = equivariant_model(totdeg,ν,L;islong = false)
   F(X) = luxchain(X, ps, st)[1]
   
   luxchain2, ps2, st2 = equivariant_model(EquivariantModels.degord2spec_nlm(totdeg,ν,L; islong = false)[1:end-1],L;islong = false)
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
      QX = [SVector{3}(x) for x in Ref(Q) .* X]
      D = wigner_D(L,Matrix(Q))'
      # D = wignerD(L, θ, θ, θ)
      if L == 0
         print_tf(@test F(X) ≈ F(QX))
      else
         print_tf(@test F(X) ≈ Ref(D) .* F(QX))
      end
   end
   println()
   
   @info("Tesing consistency between the two ways of input - in particular the ``closure'' of specifications")
   for _ = 1:30
      local X
      X = [ @SVector(rand(3)) for i in 1:10 ]
      print_tf(@test F(X) ≈ F2(X))
   end
   println()
   
end

@info("Testing the chain that generates all B bases")
totdeg = 6
ν = 2
L = Lmax
luxchain, ps, st = equivariant_model(totdeg,ν,L;islong = true)
F(X) = luxchain(X, ps, st)[1]
luxchain2, ps2, st2 = equivariant_model(EquivariantModels.degord2spec_nlm(totdeg,ν,L; islong = true)[1:end-1],L;islong = true)
F2(X) = luxchain(X, ps2, st2)[1]

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

@info("Consistency check")
totdeg = 6
ν = 2
L = Lmax
luxchain, ps, st = equivariant_model(totdeg,ν,L;islong = true);
F(X) = luxchain(X, ps, st)[1]

for l = 0:Lmax
   @info("Consistency check for L = $l")
   local FF, luxchain, ps, st
   luxchain, ps, st = equivariant_model(totdeg,ν,l;islong = false)
   FF(X) = luxchain(X, ps, st)[1]
   
   for ntest = 1:20
      X = [ @SVector(rand(3)) for i in 1:10 ]
      print_tf(@test F(X)[l+1] == FF(X))
   end
   println()
end

@info("Tesing consistency between the two ways of input - in particular the ``closure'' of specifications")
for _ = 1:10
   local X
   X = [ @SVector(rand(3)) for i in 1:10 ]
   print_tf(@test length(F(X)) == length(F2(X)) && all([F(X)[i] ≈ F2(X)[i] for i = 1:length(F(X))]))
end
println()

@info("Tesing the last way of input - given n_list and l_list")

for L = 0:Lmax
   local F, luxchain, ps, st
   
   nn = rand(0:2,4)
   ll = rand(0:2,4)
   while iseven(L) != iseven(sum(ll))
      ll = rand(0:2,4)
   end
   luxchain, ps, st = equivariant_model(nn,ll,L;islong = false)
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
      if length(F(X)) == 0 
         continue
      end
      if L == 0
         print_tf(@test F(X) ≈ F(QX))
      else
         print_tf(@test F(X) ≈ Ref(D) .* F(QX))
      end
   end
   println()
end

# We may want to get rid of it or use it in the future - to be discussed
@info("Testing SYYVector case")
totdeg = 6
ν = 2
L = Lmax
luxchain, ps, st = equivariant_SYY_model(totdeg,ν,L);
F(X) = luxchain(X, ps, st)[1]
luxchain2, ps2, st2 = equivariant_SYY_model(EquivariantModels.degord2spec_nlm(totdeg,ν,L; islong = true)[1:end-1],L)
F2(X) = luxchain(X, ps2, st2)[1]

@info("Tesing L = $L O(3) full equivariance")

for ntest = 1:20
   local X, θ1, θ2, θ3, Q, QX
   X = [ @SVector(rand(3)) for i in 1:10 ]
   θ1 = rand() * 2pi
   θ2 = rand() * 2pi
   θ3 = rand() * 2pi
   Q = RotXYZ(θ1, θ2, θ3)
   QX = [SVector{3}(x) for x in Ref(Q) .* X]
   D = BlockDiagonal([ wigner_D(l,Matrix(Q))' for l = 0:L] )
   
   print_tf(@test Ref(D) .* F(QX) ≈ F(X))
end
println()

@info("Tesing consistency between the two ways of input - in particular the ``closure'' of specifications")
for _ = 1:10
   local X
   X = [ @SVector(rand(3)) for i in 1:10 ]
   print_tf(@test length(F(X)) == length(F2(X)) && all([F(X)[i] ≈ F2(X)[i] for i = 1:length(F(X))]))
end
println()

@info("Tesing the last way of input - given n_list and l_list")

nn = rand(0:2,4)
ll = rand(0:2,4)
while iseven(Lmax) != iseven(sum(ll))
   ll = rand(0:2,4)
end

luxchain, ps, st = equivariant_SYY_model(nn,ll,L)
F(X) = luxchain(X, ps, st)[1]

@info("Tesing L = $L O(3) full equivariance")

for ntest = 1:20
   local X, θ1, θ2, θ3, Q, QX
   X = [ @SVector(rand(3)) for i in 1:10 ]
   θ1 = rand() * 2pi
   θ2 = rand() * 2pi
   θ3 = rand() * 2pi
   Q = RotXYZ(θ1, θ2, θ3)
   QX = [SVector{3}(x) for x in Ref(Q) .* X]
   D = BlockDiagonal([ wigner_D(l,Matrix(Q))' for l = 0:L] )
   
   print_tf(@test Ref(D) .* F(QX) ≈ F(X))
end
println()

## TODO: This should eventually go into ACEhamiltonians.jl
# equivariant blocks

@info("Testing the chain that generates several equivariant blocks from a long vector")
totdeg = 6
ν = 2
L = Lmax
luxchain, ps, st = equivariant_luxchain_constructor(totdeg,ν,L)
F(X) = luxchain(X, ps, st)[1]

# A small comparison - long vector does give us some redundent basis...

@info("Equivariance test")
l1l2set = [(l1,l2) for l1 = 0:L for l2 = 0:L-l1]
for ntest = 1:10
   local X, θ1, θ2, θ3, Q, QX
   X = [ @SVector(rand(3)) for i in 1:10 ]
   θ1 = rand() * 2pi
   θ2 = rand() * 2pi
   θ3 = rand() * 2pi
   Q = RotXYZ(θ1, θ2, θ3)
   QX = [SVector{3}(x) for x in Ref(Q) .* X]

   for (i,(l1,l2)) in enumerate(l1l2set)
      D1 = wigner_D(l1,Matrix(Q))'
      D2 = wigner_D(l2,Matrix(Q))'
      # D1 = wignerD(l1, 0, 0, θ)
      # D2 = wignerD(l2, 0, 0, θ)
      if F(X)[i] |> length ≠ 0
         print_tf(@test norm(Ref(D1') .* F(X)[i] .* Ref(D2) - F(QX)[i]) < 1e-12) 
      end
   end
end
println()

@info("Testing the equivariance of the second way of constructing equivariant bases")
totdeg = 6
ν = 1
L = Lmax
luxchain, ps, st = EquivariantModels.equivariant_luxchain_constructor_new(totdeg,ν,L);
F(X) = luxchain(X, ps, st)[1]

for ntest = 1:10
   local X, θ1, θ2, θ3, Q, QX
   X = [ @SVector(rand(3)) for i in 1:10 ]
   θ1 = rand() * 2pi
   θ2 = rand() * 2pi
   θ3 = rand() * 2pi
   Q = RotXYZ(θ1, θ2, θ3)
   QX = [SVector{3}(x) for x in Ref(Q) .* X]

   for i = 1:length(F(X))
      l1,l2 = Int.(size(F(X)[i][1]).-1)./2
      D1 = wigner_D(Int(l1),Matrix(Q))'
      D2 = wigner_D(Int(l2),Matrix(Q))'
      # D1 = wignerD(l1, 0, 0, θ)
      # D2 = wignerD(l2, 0, 0, θ)
      print_tf(@test Ref(D1') .* F(X)[i] .* Ref(D2) - F(QX)[i] |> norm < 1e-12)
   end
end
println()
