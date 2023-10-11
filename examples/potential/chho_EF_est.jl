using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote, Polynomials4ML
using Polynomials4ML: LinearLayer, RYlmBasis, lux 
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA, simple_radial_basis
using JuLIP, Combinatorics, Test
using ACEbase.Testing: println_slim, print_tf, fdtest
using Optimisers: destructure
using Printf

L = 0

include("staticprod.jl")

rng = Random.MersenneTwister()



# === overiding useful function as usual ===
import ChainRulesCore: ProjectTo
using ChainRulesCore
using SparseArrays

function (project::ProjectTo{SparseMatrixCSC})(dx::AbstractArray)
   dy = if axes(dx) == project.axes
       dx
   else
       if size(dx) != (length(project.axes[1]), length(project.axes[2]))
           throw(_projection_mismatch(project.axes, size(dx)))
       end
       reshape(dx, project.axes)
   end
   T = promote_type(ChainRulesCore.project_type(project.element), eltype(dx))
   nzval = Vector{T}(undef, length(project.rowval))
   k = 0
   for col in project.axes[2]
       for i in project.nzranges[col]
           row = project.rowval[i]
           val = dy[row, col]
           nzval[k += 1] = project.element(val)
       end
   end
   m, n = map(length, project.axes)
   return SparseMatrixCSC(m, n, project.colptr, project.rowval, nzval)
end


##

rcut = 5.5 
maxL = 0
totdeg = 6
ord = 3

fcut(rcut::Float64,pin::Int=2,pout::Int=2) = r -> (r < rcut ? abs( (r/rcut)^pin - 1)^pout : 0)
ftrans(r0::Float64=.0,p::Int=2) = r -> ( (1+r0)/(1+r) )^p
radial = simple_radial_basis(legendre_basis(totdeg),fcut(rcut),ftrans())

Aspec, AAspec = degord2spec(radial; totaldegree = totdeg, 
                              order = ord, 
                              Lmax = maxL, )
cats = AtomicNumber.([:W, :Cu, ])

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

luxchain, ps, st = equivariant_model(new_AAspec, radial, L; categories=allcats, islong = false)

at = rattle!(bulk(:W, cubic=true, pbc=true) * 2, 0.1)
iCu = [3, 5, 8, 12]; 
at.Z[iCu] .= cats[2]; 
nlist = JuLIP.neighbourlist(at, rcut)
_, Rs, Zs = JuLIP.Potentials.neigsz(nlist, at, 1)
# centere atom
z0  = at.Z[1]

# serialization, I want the input data structure to lux as simple as possible
get_Z0S(zz0, ZZS) = [SVector{2}(zz0, zzs) for zzs in ZZS]
Z0S = get_Z0S(z0, Zs)

# input of luxmodel
X = [Rs, Z0S]
out, st = luxchain(X, ps, st)


# == lux chain eval and grad
B = out

model = append_layers(luxchain, 
         get1 =  WrappedFunction(t -> real.(t)), 
         dot = LinearLayer(length(B), 1), 
         get2 = WrappedFunction(t -> t[1]))

ps, st = Lux.setup(MersenneTwister(1234), model)

model(X, ps, st)

# testing derivative (forces)
g = Zygote.gradient(_Rs -> model([_Rs, Z0S], ps, st)[1], Rs)[1]
grad_model(Rs, ps, st) = 
      Zygote.gradient(_Rs -> model([_Rs, Z0S], ps, st)[1], Rs)[1]

## check derivatives  

Us = randn(SVector{3, Float64}, length(g))
F1 = t -> model([Rs + t * Us, Z0S], ps, st)[1]
dF1 = t -> dot(Us, grad_model(Rs + t * Us, ps, st))
fdtest(F1, dF1, 0.0)



## === define toy loss ===


p_vec, _rest = destructure(ps)

function loss(Rs, p)
   ps = _rest(p)
   g = grad_model(Rs, ps, st)
   return dot(g, g)
end

## === testing reverse over reverse with toy loss ===

using ReverseDiff
g1 = ReverseDiff.gradient(_p -> loss(Rs, _p), p_vec)

##
using ACEbase
ACEbase.Testing.fdtest( 
         _p -> loss(Rs, _p), 
         _p -> ReverseDiff.gradient(__p -> loss(Rs, __p), _p), 
         p_vec )


# === actual lux potential === 

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

   get_Z0S(zz0, ZZS) = [SVector{2}(zz0, zzs) for zzs in ZZS]

   function LuxCalc(luxmodel, ps, st, rcut) 
      pvec, rest = destructure(ps)
      return LuxCalc(luxmodel, ps, st, rcut, rest)
   end

   cutoff(calc::LuxCalc) = calc.rcut

   function evaluate!(tmp, calc::LuxCalc, Rs, Zs, z0)
      Z0S = get_Z0S(z0, Zs)
      E, st = calc.luxmodel([Rs, Z0S], calc.ps, calc.st)
      return E[1]
   end

   function evaluate_d!(dEs, tmpd, calc::LuxCalc, Rs, Zs, z0)
      Z0S = get_Z0S(z0, Zs)
      g = Zygote.gradient(X -> calc.luxmodel([X, Z0S], calc.ps, calc.st)[1], Rs)[1]
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
            Ei, st = calc.luxmodel([Rs, Z0S], ps, st)
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
         Z0S = get_Z0S(at.Z[i], Zs)
         comp = Zygote.withgradient(_X -> calc.luxmodel([_X, Z0S], ps, st)[1], Rs)
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

ps.dot.W[:] .= 0.01 * randn(length(ps.dot.W)) 
calc = Pot.LuxCalc(model, ps, st, rcut)
at = rattle!(bulk(:W, cubic=true, pbc=true) * 2, 0.1)
iCu = [3, 5, 8, 12]; 
at.Z[iCu] .= cats[2]; 
E, F, V = Pot.lux_efv(at, calc, ps, st)

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
      err += ( (Eref-E) / Nat)^2 + sum( f -> sum(abs2, f), (Fref .- F) ) / Nat #  + 
         # sum(abs2, (Vref.-V) )
   end
   return err
end


# generate training data



ACEbase.Testing.fdtest( 
         _p -> loss(train, calc, _p), 
         _p -> ReverseDiff.gradient(__p -> loss(train, calc, __p), _p), 
         p_vec)

