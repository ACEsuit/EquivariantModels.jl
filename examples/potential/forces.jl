using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote 
using Polynomials4ML: LinearLayer
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA
rng = Random.MersenneTwister()

##

maxL = 0
Aspec, AAspec = degord2spec(; totaldegree = 6, 
                                  order = 3, 
                                  Lmax = maxL, )

l_basis, ps_basis, st_basis = equivariant_model(AAspec, maxL)

X = [ @SVector(randn(3)) for i in 1:10 ]
B = l_basis(X, ps_basis, st_basis)[1][1]

model = append_layers(l_basis, 
                      get1 = WrappedFunction(t -> real.(t[1])), 
                      dot = LinearLayer(length(B), 1), 
                      get2 = WrappedFunction(t -> t[1]), )
ps, st = Lux.setup(rng, model)
model(X, ps, st)

# testing derivative (forces)
Zygote.gradient(X -> model(X, ps, st)[1], X)

##

module Pot 
   import JuLIP, Zygote 
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
      g = Zygote.gradient(X -> calc.luxmodel(X, calc.ps, calc.st)[1], Rs)[1]
      @assert length(g) == length(dEs) == length(Rs)
      dEs[:] .= g 
      return dEs 
   end
end 

##

using JuLIP 
ps.dot.W[:] .= 0 

at = bulk(:W, cubic=true, pbc=true) * 2
calc = Pot.LuxCalc(model, ps, st, 5.5)
JuLIP.energy(calc, at)
JuLIP.forces(calc, at)
JuLIP.virial(calc, at)
