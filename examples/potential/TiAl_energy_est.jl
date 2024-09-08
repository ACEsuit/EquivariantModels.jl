using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote 
using Polynomials4ML: LinearLayer, RYlmBasis, lux, legendre_basis 
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA, simple_radial_basis
using JuLIP, Combinatorics, ACEbase

include("staticprod.jl")

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

# data
# TiAl
# train = read_extxyz("TiAl_tutorial.xyz")
# NiAl
using PyCall, ASE
asekim = pyimport("ase.calculators.kim.kim")
eam = asekim.KIM("EAM_Dynamo_MishinMehlPapaconstantopoulos_2002_NiAl__MO_109933561507_005");

eam = ASECalculator(eam)
io = pyimport("ase.io")
at0 = ASE.Atoms(ASE.ASEAtoms(io.read("mp-1487_AlNi.cif")))
function gen_dat()
   at_ = deepcopy(at0) * 2
   rattle!(at_, 0.1)
   set_data!(at_, "energy", energy(eam, at_))
   set_data!(at_, "forces", forces(eam, at_))
   set_data!(at_, "virial", virial(eam, at_))
   return at_
end
# Random.seed!(0)
train = [gen_dat() for _ = 1:10];

spec = [:Ni, :Al]
# spec = [:Ti, :Al]

rng = Random.MersenneTwister()

rcut = 5.5 
maxL = 0
totdeg = 5
ord = 2

fcut(rcut::Float64,pin::Int=2,pout::Int=2) = r -> (r < rcut ? abs( (r/rcut)^pin - 1)^pout : 0)
ftrans(r0::Float64=2.0,p::Int=2) = r -> ( (1+r0)/(1+r) )^p
radial = simple_radial_basis(legendre_basis(totdeg),fcut(rcut),ftrans())



Aspec, AAspec = degord2spec(radial; totaldegree = totdeg, 
                              order = ord, 
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

luxchain, ps, st = equivariant_model(new_AAspec, radial, maxL; categories=allcats, islong=false)
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

B = out
model = append_layers(luxchain, get1 =  WrappedFunction(t -> real.(t)), dot = LinearLayer(length(B), 1), get2 = WrappedFunction(t -> t[1]))
ps, st = Lux.setup(MersenneTwister(1234), model)

E = 0
let st = st, E = E
   for i = 1:length(at)
      _, Rs, Zs = JuLIP.Potentials.neigsz(nlist, at, i)
      Z0S = get_Z0S(at.Z[i], Zs)
      X = [Rs, Z0S]
      Ei, st = model(X, ps, st)
      E += Ei[1]
   end
end


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
      err += ( (Eref-E) / Nat)^2 + sum( f -> sum(abs2, f), (Fref .- F) ) / Nat / 100  #  + 
         # sum(abs2, (Vref.-V) )
   end
   return err
end

# ACEbase.Testing.fdtest( 
#          _p -> loss(train, calc, _p), 
#          _p -> ReverseDiff.gradient(__p -> loss(train, calc, __p), _p), 
#          p_vec)

p0 = zero.(p_vec)
E_loss(train, calc, p0)
ReverseDiff.gradient(p -> loss(train, calc, p), p0)
# Zygote.gradient(p -> E_loss(train, calc, p), p_vec)[1]

using Optim
obj_f = x -> loss(train, calc, x)
obj_g! = (g, x) -> copyto!(g, ReverseDiff.gradient(p -> loss(train, calc, p), x))
# obj_g! = (g, x) -> copyto!(g, Zygote.gradient(p -> E_loss(train, calc, p), x)[1])

res = optimize(obj_f, obj_g!, p0,
              Optim.BFGS(),
            #   Optimisers.AdamW(),
              Optim.Options(g_tol = 1e-6, show_trace = true))

Eerrmin = Optim.minimum(res)
RMSE = sqrt(Eerrmin / length(train))
pargmin = Optim.minimizer(res)

ace = Pot.LuxCalc(model, pargmin, st, rcut)
Eref = []
Eace = []
for tr in train
    exact = tr.data["energy"].data
    estim = Pot.lux_energy(tr, ace, _rest(pargmin), st) 
    push!(Eref, exact)
    push!(Eace, estim)
end

test = [gen_dat() for _ = 1:300];
Eref_te = []
Eace_te = []
for te in test
    exact = te.data["energy"].data
    estim = Pot.lux_energy(te, ace, _rest(pargmin), st) 
    push!(Eref_te, exact)
    push!(Eace_te, estim)
end

MIN = Eref_te |> minimum
MAX = Eref_te |> maximum
using PyPlot
figure()
scatter(Eref, Eace, c="red", alpha=0.4)
scatter(Eref_te, Eace_te, c="blue", alpha=0.4)
plot(MIN:0.01:MAX, MIN:0.01:MAX, lw=2, c="k", ls="--")
PyPlot.legend(["Train", "Test"], fontsize=14, loc=2);
xlabel("Reference energy")
ylabel("ACE energy")
axis("square")
xlim([MIN-0.05, MAX+0.05])
ylim([MIN-0.05, MAX+0.05])
PyPlot.savefig("NiAl_energy_fitting.png")

