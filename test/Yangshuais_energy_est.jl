using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote 
using Polynomials4ML: LinearLayer, RYlmBasis, lux, legendre_basis
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA, simple_radial_basis
rng = Random.MersenneTwister()

# dataset
using ASE, JuLIP
function gen_dat()
   eam = JuLIP.Potentials.EAM("test/w_eam4.fs")
   at = rattle!(bulk(:W, cubic=true) * 2, 0.1)
   set_data!(at, "energy", energy(eam, at))
   return at
end
Random.seed!(0)
train = [gen_dat() for _ = 1:100];

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

module Pot 
   import JuLIP, Zygote, StaticArrays
   import JuLIP: cutoff, Atoms 
   import ACEbase: evaluate!, evaluate_d!
   import StaticArrays: SVector, SMatrix

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
         ∇Ei = comp.grad[1] 
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
ReverseDiff.gradient(p -> E_loss(train, calc, p), p_vec)
Zygote.gradient(p -> E_loss(train, calc, p), p_vec)[1]

using Optim
obj_f = x -> E_loss(train, calc, x)
obj_g! = (g, x) -> copyto!(g, ReverseDiff.gradient(p -> E_loss(train, calc, p), x))
# obj_g! = (g, x) -> copyto!(g, Zygote.gradient(p -> E_loss(train, calc, p), x)[1])

using LineSearches: BackTracking
using LineSearches
# solver = Optim.ConjugateGradient()#linesearch = BackTracking(order=2, maxstep=Inf))
# solver = Optim.GradientDescent(linesearch = BackTracking(order=2, maxstep=Inf) )
# solver = Optim.BFGS()
solver = Optim.LBFGS() #alphaguess = LineSearches.InitialHagerZhang(),
               # linesearch = BackTracking(order=2, maxstep=Inf) )

res = optimize(obj_f, obj_g!, p0, solver,
               Optim.Options(f_tol = 1e-10, g_tol = 1e-6, show_trace = true))

Eerrmin = Optim.minimum(res)
RMSE = sqrt(Eerrmin / length(train))
pargmin = Optim.minimizer(res)
p1 = pargmin

train = [gen_dat() for _ = 1:500];
res_new = optimize(obj_f, obj_g!, p1, solver,
               Optim.Options(f_tol = 1e-10, g_tol = 1e-5, show_trace = true))

Eerrmin_new = Optim.minimum(res_new)
RMSE_new = sqrt(Eerrmin_new / length(train))
pargmin_new = Optim.minimizer(res_new)
# plot the fitting result

ace = Pot.LuxCalc(model, pargmin, st, rcut)
Eref = []
Eace = []
for tr in train
    exact = tr.data["energy"].data
    estim = Pot.lux_energy(tr, ace, _rest(pargmin_new), st) 
    push!(Eref, exact)
    push!(Eace, estim)
end

test = [gen_dat() for _ = 1:300];
Eref_te = []
Eace_te = []
for te in test
    exact = te.data["energy"].data
    estim = Pot.lux_energy(te, ace, _rest(pargmin_new), st) 
    push!(Eref_te, exact)
    push!(Eace_te, estim)
end

using PyPlot
figure()
scatter(Eref, Eace, c="red", alpha=0.4)
scatter(Eref_te, Eace_te, c="blue", alpha=0.4)
plot(-142.3:0.01:-141.5, -142.3:0.01:-141.5, lw=2, c="k", ls="--")
PyPlot.legend(["Train", "Test"], fontsize=14, loc=2);
xlabel("Reference energy")
ylabel("ACE energy")
axis("square")
xlim([-142.3, -141.5])
ylim([-142.3, -141.5])
PyPlot.savefig("W_energy_fitting.png")