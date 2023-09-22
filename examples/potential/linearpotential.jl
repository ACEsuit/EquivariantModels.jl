using EquivariantModels, Lux, StaticArrays, Random 
using Polynomials4ML: LinearLayer
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA
rng = Random.MersenneTwister()

maxL = 0

Aspec, AAspec = degord2spec(; totaldegree = 6, 
                                  order = 3, 
                                  Lmax = maxL, )

chain_xx2AA, ps1, st1 = xx2AA(AAspec, maxL)

chain_AA2B, ps2, st2 = equivariant_model(AAspec, maxL)

# A = randn(length(Aspec))
# 
# chain_A2B(A, ps, st)

X = [ @SVector(randn(3)) for i in 1:10 ]
chain_xx2AA(X, ps1, st1)
B, st = chain_AA2B(X, ps2, st2)

# model = Lux.Chain(basis = chain_AA2B, 
#                   get1 = WrappedFunction(t -> real.(t[1])), 
#                   dot = LinearLayer(length(B[1]), 1))
model = append_layer(chain_AA2B, WrappedFunction(t -> real.(t[1])); l_name = :get1)
model = append_layer(model, LinearLayer(length(B[1]), 1); l_name = :dot)

ps, st = Lux.setup(rng, model)

##

module Pot 
   import JuLIP 
   import JuLIP: cutoff
   import ACEbase: evaluate!, evaluate_d!

   struct LuxCalc <: JuLIP.SitePotential 
      luxmodel
      ps
      st 
      rcut::Float64
   end

   cutoff(calc::LuxCalc) = calc.rcut

   function evaluate!(tmp, calc::LuxCalc, Rs, Zs, z0)
      E, st = calc.luxmodel(Rs, calc.ps, calc.st)
      return E[1]
   end

   function evaluate_d!(dEs, tmpd, calc::LuxCalc, Rs, Zs, z0)
      # TODO ... 
      return dEs 
   end
end 

##

using JuLIP 
at = bulk(:W, cubic=true, pbc=true) * 2
calc = Pot.LuxCalc(model, ps, st, 5.5)
JuLIP.energy(calc, at)
JuLIP.forces(calc, at)
