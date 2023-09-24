using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote 
using Polynomials4ML: LinearLayer, RYlmBasis, lux 
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA
rng = Random.MersenneTwister()

##

rcut = 5.5 
maxL = 0
Aspec, AAspec = degord2spec(; totaldegree = 6, 
                              order = 3, 
                              Lmax = maxL, )

l_basis, ps_basis, st_basis = equivariant_model(AAspec, maxL)
X = [ @SVector(randn(3)) for i in 1:10 ]
B = l_basis(X, ps_basis, st_basis)[1][1]

# now build another model with a better transform 
L = maximum(b.l for b in Aspec) 
len_BB = length(B) 
get1 = WrappedFunction(t -> t[1])
embed = Parallel(nothing; 
       Rn = Chain(trans = WrappedFunction(xx -> [1/(1+norm(x)) for x in xx]), 
                   poly = l_basis.layers.embed.layers.Rn, ), 
      Ylm = Chain(Ylm = lux(RYlmBasis(L)),  ) )

model = Chain( 
         embed = embed, 
         A = l_basis.layers.A, 
         AA = l_basis.layers.AA, 
         AA_sort = l_basis.layers.AA_sort, 
         BB = l_basis.layers.BB, 
         get1 = WrappedFunction(t -> t[1]), 
         dot = LinearLayer(len_BB, 1), 
         get2 = WrappedFunction(t -> t[1]), )
ps, st = Lux.setup(rng, model)
out, st = model(X, ps, st)

# testing derivative (forces)
g = Zygote.gradient(X -> model(X, ps, st)[1], X)[1] 

##

module Pot 
   import JuLIP, Zygote 
   import JuLIP: cutoff, Atoms 
   import ACEbase: evaluate!, evaluate_d!

   import ChainRulesCore
   import ChainRulesCore: rrule, ignore_derivatives

   import Optimisers: destructure

   struct LuxCalc <: JuLIP.SitePotential 
      luxmodel
      ps
      st 
      rcut::Float64
      restructure
   end

   function LuxCalc(luxmodel, ps, st, rcut) 
      pvec, rest = destructure(ps)
      return LuxCalc(luxmodel, ps, st, rcut, rest)
   end

   cutoff(calc::LuxCalc) = calc.rcut

   function evaluate!(tmp, calc::LuxCalc, Rs, Zs, z0)
      E, st = calc.luxmodel(Rs, calc.ps, calc.st)
      return E[1]
   end

   function evaluate_d!(dEs, tmpd, calc::LuxCalc, Rs, Zs, z0)
      g = Zygote.gradient(X -> calc.luxmodel(X, calc.ps, calc.st)[1], Rs)[1]
      @assert length(g) == length(Rs) <= length(dEs)
      dEs[1:length(g)] .= g 
      return dEs 
   end

   # ----- parameter estimation stuff 

   
   function lux_energy(at::Atoms, calc::LuxCalc, ps::NamedTuple, st::NamedTuple)
      nlist = ignore_derivatives() do 
         JuLIP.neighbourlist(at, calc.rcut)
      end
      return sum( i -> begin
            Js, Rs, Zs = ignore_derivatives() do 
               JuLIP.Potentials.neigsz(nlist, at, i)
            end
            Ei, st = calc.luxmodel(Rs, ps, st)
            Ei[1] 
         end, 
         1:length(at)
         )
   end

   # function rrule(::typeof(lux_energy), at::Atoms, calc::LuxCalc, ps::NamedTuple, st::NamedTuple) 
   #    E = lux_energy(at, calc, ps, st)
   #    function pb(Δ)
   #       nlist = JuLIP.neighbourlist(at, calc.rcut)
   #       @show Δ 
   #       error("stop")
   #       function pb_inner(i)
   #          Js, Rs, Zs = JuLIP.Potentials.neigsz(nlist, at, i)
   #          gi = ReverseDiff.gradient()
   #       end
   #       for i = 1:length(at) 
   #          Ei, st = calc.luxmodel(Rs, calc.ps, calc.st)
   #          E += Ei[1]
   #       end 
   #    end
   # end

end 

##

using JuLIP
JuLIP.usethreads!(false) 
ps.dot.W[:] .= 0.01 * randn(length(ps.dot.W)) 

at = rattle!(bulk(:W, cubic=true, pbc=true) * 2, 0.1)
calc = Pot.LuxCalc(model, ps, st, rcut)
JuLIP.energy(calc, at)
JuLIP.forces(calc, at)
JuLIP.virial(calc, at)
Pot.lux_energy(at, calc, ps, st)

@time JuLIP.energy(calc, at)
@time Pot.lux_energy(at, calc, ps, st)
@time JuLIP.forces(calc, at)

##

using Optimisers, ReverseDiff

p_vec, _rest = destructure(ps)
f(_pvec) = Pot.lux_energy(at, calc, _rest(_pvec), st)

f(p_vec)
g = Zygote.gradient(f, p_vec)[1]

@time f(p_vec)
@time Zygote.gradient(f, p_vec)[1]


# This fails for now
# gr = ReverseDiff.gradient(f, p_vec)[1]