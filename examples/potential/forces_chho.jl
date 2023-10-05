using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote, Polynomials4ML
using Polynomials4ML: LinearLayer, RYlmBasis, lux 
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA, simple_radial_basis
rng = Random.MersenneTwister()

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

# testing derivative (forces)
g = Zygote.gradient(X -> model(X, ps, st)[1], X)[1] 

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
      E = 0.0 
      F = zeros(SVector{3, Float64}, length(at))
      V = zero(SMatrix{3, 3, Float64}) 
      for i = 1:length(at) 
         Js, Rs, Zs = ignore_derivatives() do 
            JuLIP.Potentials.neigsz(nlist, at, i)
         end
         comp = Zygote.withgradient(_X -> calc.luxmodel(_X, ps, st)[1], Rs)
         Ei = comp.val 
         _∇Ei = comp.grad[1]
         ∇Ei = ReverseDiff.value.(_∇Ei)
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

#    site_virial(dV::AbstractVector{JVec{T1}}, R::AbstractVector{JVec{T2}}
#    ) where {T1, T2} =  (
# length(R) > 0 ? (- sum( dVi * Ri' for (dVi, Ri) in zip(dV, R) ))
#            : zero(JMat{fltype_intersect(T1, T2)})
# )   
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
ps.dot.W[:] .= 1e-2 * randn(length(ps.dot.W)) 

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
gz = Zygote.gradient(f, p_vec)[1]

@time f(p_vec)
@time Zygote.gradient(f, p_vec)[1]

# We can use either Zygote or ReverseDiff for gradients. 
gr = ReverseDiff.gradient(f, p_vec)
@show gr ≈ gz

@info("Interestingly ReverseDiff is much faster here, almost optimal")
@time f(p_vec)
@time Zygote.gradient(f, p_vec)[1]
@time ReverseDiff.gradient(f, p_vec)

##

@info("Compute Energies, Forces and Virials at the same time")
E, F, V = Pot.lux_efv(at, calc, ps, st)
@show E ≈ JuLIP.energy(calc, at)
@show F ≈ JuLIP.forces(calc, at)
@show V ≈ JuLIP.virial(calc, at)

## 

# make up a baby loss function type thing.
function loss(at, calc, p_vec)
   ps = _rest(p_vec)
   st = calc.st
   E, F, V = Pot.lux_efv(at, calc, ps, st)
   Nat = length(at)
   return (E / Nat)^2 + 
          sum( f -> sum(abs2, f), F ) / Nat + 
          sum(abs2, V)
end

loss(at, calc, p_vec)

# ====
using Polynomials4ML
import ChainRulesCore: ProjectTo
using ChainRulesCore
using SparseArrays
function Polynomials4ML._pullback_evaluate(∂A, basis::Polynomials4ML.PooledSparseProduct{NB}, BB::Polynomials4ML.TupMat) where {NB}
   nX = size(BB[1], 1)
   TA = promote_type(eltype.(BB)..., eltype(∂A))
   # @show TA
   ∂BB = ntuple(i -> zeros(TA, size(BB[i])...), NB)
   Polynomials4ML._pullback_evaluate!(∂BB, ∂A, basis, BB)
   return ∂BB
end

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
ReverseDiff.gradient(p -> loss(at, calc, p), p_vec)
