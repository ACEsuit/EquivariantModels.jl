using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote 
using Polynomials4ML: LinearLayer, RYlmBasis, lux 
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA
using JuLIP
using Combinatorics: permutations

rng = Random.MersenneTwister()

include("staticprod.jl")
##

# == configs and form model ==
rcut = 5.5 
maxL = 0
L = 4
Aspec, AAspec = degord2spec(; totaldegree = 4, 
                                  order = 2, 
                                  Lmax = 0, )
cats = AtomicNumber.([:W, :W])

new_spec = []
ori_AAspec = deepcopy(AAspec)
new_AAspec = []

for bb in ori_AAspec
   newbb = []
   for t in bb
      push!(newbb, (t..., s = cats))
   end
   push!(new_AAspec, newbb)
end

cat_perm2 = collect(SVector{2}.(permutations(cats, 2)))

luxchain, ps, st = equivariant_model(new_AAspec, 0; categories = cat_perm2, islong = false)

##

# == init example data == 

at = rattle!(bulk(:W, cubic=true, pbc=true) * 2, 0.1)
nlist = JuLIP.neighbourlist(at, rcut)
_, Rs, Zs = JuLIP.Potentials.neigsz(nlist, at, 1)
# centere atom
z0  = at.Z[1]

# serialization, I want the input data structure to lux as simple as possible
get_Z0S(zz0, ZZS) = [SVector{2}(zz0, zzs) for zzs in ZZS]
Z0S = get_Z0S(z0, Zs)

# input of luxmodel
X = (Rs, Z0S)


# == lux chain eval and grad
out, st = luxchain(X, ps, st)
B = out

model = append_layers(luxchain, get1 =  WrappedFunction(t -> real.(t)), dot = LinearLayer(length(B), 1), get2 = WrappedFunction(t -> t[1]))

ps, st = Lux.setup(MersenneTwister(1234), model)

model(X, ps, st)

# testing derivative (forces)
g = Zygote.gradient(X -> model(X, ps, st)[1], X)[1][1]



##

module Pot
   using StaticArrays: SVector

   import JuLIP, Zygote 
   import JuLIP: cutoff, Atoms 
   import ACEbase: evaluate!, evaluate_d!

   import ChainRulesCore
   import ChainRulesCore: rrule, ignore_derivatives

   import Optimisers: destructure

   get_Z0S(zz0, ZZS) = [SVector{2}(zz0, zzs) for zzs in ZZS]

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
      Z0S = get_Z0S(z0, Zs)
      X = (Rs, Z0S)
      E, st = calc.luxmodel(X, calc.ps, calc.st)
      return E[1]
   end

   function evaluate_d!(dEs, tmpd, calc::LuxCalc, Rs, Zs, z0)
      Z0S = get_Z0S(z0, Zs)
      X = (Rs, Z0S)
      g = Zygote.gradient(X -> calc.luxmodel(X, calc.ps, calc.st)[1], X)[1][1]
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
            Z0S = get_Z0S(at.Z[1], Zs)
            X = (Rs, Z0S)
            Ei, st = calc.luxmodel(X, ps, st)
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
