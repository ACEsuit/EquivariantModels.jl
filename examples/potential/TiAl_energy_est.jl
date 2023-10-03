using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote 
using Polynomials4ML: LinearLayer, RYlmBasis, lux 
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA
using JuLIP, Combinatorics

include("staticprod.jl")

# data
train = read_extxyz("TiAl_tutorial.xyz")

spec = [:Ti, :Al]

rng = Random.MersenneTwister()

rcut = 5.5 
maxL = 0
Aspec, AAspec = degord2spec(; totaldegree = 6, 
                                  order = 3, 
                                  Lmax = maxL, )
cats = AtomicNumber.(spec)
ipairs = collect(Combinatorics.permutations(1:length(cats), 2))
allcats = collect(SVector{2}.(Combinatorics.permutations(cats, 2)))

for (i, cat) in enumerate(cats) 
   push!(ipairs, [i, i]) 
   push!(allcats, SVector{2}([cat, cat])) 
end

new_spec = []
ori_AAspec = deepcopy(AAspec)
new_AAspec = []

for bb in ori_AAspec
   newbb = []
   for (t, ip) in zip(bb, ipairs)
      push!(newbb, (t..., s = cats[ip]))
   end
   push!(new_AAspec, newbb)
end

at = train[end]
nlist = JuLIP.neighbourlist(at, rcut)

luxchain, ps, st = equivariant_model(new_AAspec, maxL; categories=allcats, islong=false)
nlist = JuLIP.neighbourlist(at, rcut)
_, Rs, Zs = JuLIP.Potentials.neigsz(nlist, at, 1)
# centere atom
z0  = at.Z[1]
# serialization, I want the input data structure to lux as simple as possible
get_Z0S(zz0, ZZS) = [SVector{2}(zz0, zzs) for zzs in ZZS]
Z0S = get_Z0S(z0, Zs)
# input of luxmodel
X = (Rs, Z0S)
out, st = luxchain(X, ps, st)
B = out
model = append_layers(luxchain, get1 =  WrappedFunction(t -> real.(t)), dot = LinearLayer(length(B), 1), get2 = WrappedFunction(t -> t[1]))
ps, st = Lux.setup(MersenneTwister(1234), model)

E = 0
let st = st, E = E
   for i = 1:length(at)
      _, Rs, Zs = JuLIP.Potentials.neigsz(nlist, at, i)
      Z0S = get_Z0S(at.Z[i], Zs)
      X = (Rs, Z0S)
      Ei, st = model(X, ps, st)
      E += Ei[1]
   end
end

module Pot 
   import JuLIP, Zygote, StaticArrays
   import JuLIP: cutoff, Atoms 
   import ACEbase: evaluate!, evaluate_d!
   import StaticArrays: SVector, SMatrix

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
            Z0S = get_Z0S(at.Z[i], Zs)
            X = (Rs, Z0S)
            Ei, st = calc.luxmodel(X, ps, st)
            Ei[1] 
         end, 
         1:length(at)
         )
   end

end 


ps.dot.W[:] .= 0.01 * randn(length(ps.dot.W)) 
calc = Pot.LuxCalc(model, ps, st, rcut)
JuLIP.energy(calc, at)
Pot.lux_energy(at, calc, ps, st)

@time JuLIP.energy(calc, at)
@time Pot.lux_energy(at, calc, ps, st)

using Optimisers, ReverseDiff

p_vec, _rest = destructure(ps)

# energy loss function 
function E_loss(train, calc, p_vec)
   ps = _rest(p_vec)
   st = calc.st
   Eerr = 0
   for at in train
      Nat = length(at)
      Eref = at.data["energy"].data
      E = Pot.lux_energy(at, calc, ps, st)
      Eerr += ( (Eref - E) / Nat)^2
   end
   return Eerr 
end

p0 = zero.(p_vec)
E_loss(train, calc, p0)
ReverseDiff.gradient(p -> E_loss(train, calc, p), p0)
# Zygote.gradient(p -> E_loss(train, calc, p), p_vec)[1]

using Optim
obj_f = x -> E_loss(train, calc, x)
obj_g! = (g, x) -> copyto!(g, ReverseDiff.gradient(p -> E_loss(train, calc, p), x))
# obj_g! = (g, x) -> copyto!(g, Zygote.gradient(p -> E_loss(train, calc, p), x)[1])

res = optimize(obj_f, obj_g!, p0,
              Optim.BFGS(),
              Optim.Options(x_tol = 1e-15, f_tol = 1e-15, g_tol = 1e-6, show_trace = true))

Eerrmin = Optim.minimum(res)
pargmin = Optim.minimizer(res)



