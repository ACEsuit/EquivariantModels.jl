using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Polynomials4ML
using Polynomials4ML: LinearLayer, RYlmBasis, lux 
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA, simple_radial_basis, ConstLinearLayer
using LuxCore
using SparseArrays
using Test
using ACEbase.Testing: print_tf, fdtest
using Zygote

rng = Random.MersenneTwister()

n, m, k = 40, 200, 80
sparsity = 0.02
T = ComplexF64
vec_len = 5

##
cases = ["AbstractSparseMatrixCSC{ <: Number} operation", "AbstractSparseMatrixCSC{ <: SVector} operation",
          "Testing general fallback to genericmatmul (<: Number)", "Testing general fallback to genericmatmul (<: SVector)",]
Cs = [sprand(T, n, m, sparsity), sprand(SVector{vec_len, T}, n, m, sparsity), 
      randn(T, n, m), randn(SVector{vec_len, T}, n, m)]
uTs = [T, SVector{5, T}, T, SVector{5, T}]

for (case, C, uT) in zip(cases, Cs, uTs)
   @info(case)
   l = ConstLinearLayer(C)
   ps, st = LuxCore.setup(rng, l)

   @info("Test vector input")

   @info("Testing evaluation")
   for ntest = 1:30
      local X
      X = randn(T, m)
      print_tf(@test l(X, ps, st)[1] ≈ l.op * X)
   end

   println()

   @info("Testing pullback")
   for ntest = 1:30
      local val, x, u, bu
      x = randn(T, m)
      bu = randn(T, m)
      _BB(t) = x + t * bu
      val, _ = l(x, ps, st)
      u = randn(uT, size(val))
      F(t) = dot(u, l(_BB(t), ps, st)[1])
      dF(t) = begin
         val, pb = Zygote.pullback(LuxCore.apply, l, _BB(t), ps, st)
         ∂BB = pb((u, st))[2]
         return dot(∂BB, bu)
      end
      print_tf(@test fdtest(F, dF, 0.0; verbose=false))
   end

   println()

   @info("Test matrix input")

   @info("Testing evaluation")
   for ntest = 1:30
      local X
      X = randn(T, m, k)
      print_tf(@test l(X, ps, st)[1] ≈ l.op * X)
   end

   println()

   @info("Testing pullback")
   for ntest = 1:30
      local val, x, u, bu
      x = randn(T, m, k)
      bu = randn(T, m, k)
      _BB(t) = x + t * bu
      val, _ = l(x, ps, st)
      u = randn(uT, size(val))
      F(t) = dot(u, l(_BB(t), ps, st)[1])
      dF(t) = begin
         val, pb = Zygote.pullback(LuxCore.apply, l, _BB(t), ps, st)
         ∂BB = pb((u, st))[2]
         return dot(∂BB, bu)
      end
      print_tf(@test fdtest(F, dF, 0.0; verbose=false))
   end

   println()
end