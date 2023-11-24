using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote 
using Polynomials4ML: LinearLayer, RYlmBasis, lux, legendre_basis
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA, simple_radial_basis
using JuLIP, Combinatorics
using ASE
using Random

rng = Random.MersenneTwister()

function gen_dat()
    #eam = JuLIP.Potentials.EAM("/zfs/users/jerryho528/jerryho528/julia_ws/EquivariantModels.jl/examples/potential/w_eam4.fs")
    at = rattle!(bulk(:W, cubic=true) * 2, 0.1)
    set_data!(at, "energy", 10.0)
    set_data!(at, "forces", [@SVector randn(3) for _  = 1:length(at)])
    set_data!(at, "virial", @SMatrix randn(3,3))
    return at
end

## 

# === Model/SitePotential construction ===
rcut = 5.5 
maxL = 0
totdeg = 8
ord = 2

fcut(rcut::Float64,pin::Int=2,pout::Int=2) = r -> (r < rcut ? abs( (r/rcut)^pin - 1)^pout : 0)
ftrans(r0::Float64=2.0,p::Int=2) = r -> ( (1+r0)/(1+r) )^p
radial = simple_radial_basis(legendre_basis(totdeg))

Aspec, AAspec = degord2spec(radial; totaldegree = totdeg, 
                              order = ord, 
                              Lmax = maxL, )

l_basis, ps_basis, st_basis = equivariant_model(AAspec, radial, maxL; islong = false)
X = [ @SVector(randn(3)) for i in 1:10 ]
B = l_basis(X, ps_basis, st_basis)[1]

# now extend the above BB basis to a model
len_BB = length(B) 

model = append_layer(l_basis, WrappedFunction(t -> real(t)); l_name=:real)
model = append_layer(model, LinearLayer(len_BB, 1); l_name=:dot)
model = append_layer(model, WrappedFunction(t -> t[1]); l_name=:get1)
         
ps, st = Lux.setup(rng, model)
out, st = model(X, ps, st)

## 

module Pot 
   import JuLIP, Zygote, StaticArrays
   import JuLIP: cutoff, Atoms 
   import ACEbase: evaluate!, evaluate_d!
   import StaticArrays: SVector, SMatrix
   import ReverseDiff
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
            @show length(Rs)
            Ei, st = calc.luxmodel(Rs, ps, st)
            Ei[1] 
         end, 
         1:length(at)
         )
   end


   function lux_efv(at::Atoms, calc::LuxCalc, ps::NamedTuple, st::NamedTuple)
      nlist = ignore_derivatives() do 
         JuLIP.neighbourlist(at, calc.rcut)
      end
      T = promote_type(eltype(at.X[1]), eltype(ps.dot.W))
      E = 0.0 
      F = zeros(SVector{3, T}, length(at))
      V = zero(SMatrix{3, 3, T}) 
      for i = 1:length(at) 
         Js, Rs, Zs = ignore_derivatives() do 
            JuLIP.Potentials.neigsz(nlist, at, i)
         end
         comp = Zygote.withgradient(_X -> calc.luxmodel(_X, ps, st)[1], Rs)
         Ei = comp.val 
         _∇Ei = comp.grad[1]
         # ∇Ei = ReverseDiff.value.(_∇Ei)
         ∇Ei = _∇Ei
         # energy 
         E += Ei 

         # Forces 
         for j = 1:length(Rs) 
            F[Js[j]] -= ∇Ei[j] 
            F[i] += ∇Ei[j] 
         end

         # Virial 
         if length(Rs) > 0 
            V -= sum(∇Eij * Rij' for (∇Eij, Rij) in zip(∇Ei, Rs))
         end
      end
      
      return E, F, V 
   end
end

## === benchmarks ===
model(X, ps, st)
# define potential
calc = Pot.LuxCalc(model, ps, st, rcut)


using Optimisers
train = [gen_dat() for _ = 1:20]
at = train[1]

p_vec, _rest = destructure(ps)

Eref = JuLIP.energy(calc, at)
Fref = JuLIP.forces(calc, at)
nlist = JuLIP.neighbourlist(at, calc.rcut)


using BenchmarkTools
@info("evaluate energy")
@profview let calc = calc, at = at
   for _ = 1:10
        JuLIP.energy(calc, at)
   end
end

@btime JuLIP.energy($calc, $at)


@info("evaluate force")
@profview let calc = calc, at = at
   for _ = 1:10
        JuLIP.forces(calc, at)
   end
end

@btime JuLIP.forces($calc, $at)

function loss(train, calc, p_vec)
   ps = _rest(p_vec)
   st = calc.st
   err = 0
   for at in train
      Nat = length(at)
      Eref = at.data["energy"].data
      Fref = at.data["forces"].data
      Vref = at.data["virial"].data
      E, F, V = Pot.lux_efv(at, calc, ps, st)
      err += ( (Eref-E) / Nat)^2 + sum( f -> sum(abs2, f), (Fref .- F) ) / Nat / 30  + 
         sum(abs2, (Vref.-V) ) / 30
   end
   return err
end

@info("evaluate loss")
@btime loss($train, $calc, $p_vec)

@info("evaluate double pb")

using ReverseDiff
@profview let calc = calc, at = at
   for _ = 1:10
      ReverseDiff.gradient(p -> loss(train, calc, p), p_vec)
   end
end

@btime ReverseDiff.gradient(p -> $loss($train, $calc, p), $p_vec)