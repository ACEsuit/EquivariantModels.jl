using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote 
using Polynomials4ML: LinearLayer, RYlmBasis, lux 
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA
rng = Random.MersenneTwister()

# dataset
using ASE, JuLIP
function gen_dat()
   eam = JuLIP.Potentials.EAM("w_eam4.fs")
   at = rattle!(bulk(:W, cubic=true) * 3, 0.1)
   set_data!(at, "energy", energy(eam, at))
   return at
end
Random.seed!(0)
train = [gen_dat() for _ = 1:1000];

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
         # AA_sort = l_basis.layers.AA_sort, 
         BB = l_basis.layers.BB, 
         get1 = WrappedFunction(t -> t[1]), 
         dot = LinearLayer(len_BB, 1), 
         get2 = WrappedFunction(t -> t[1]), )
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
ReverseDiff.gradient(p -> E_loss(train, calc, p), p0)
Zygote.gradient(p -> E_loss(train, calc, p), p_vec)[1]

using Optim
obj_f = x -> E_loss(train, calc, x)
# obj_g! = (g, x) -> copyto!(g, ReverseDiff.gradient(p -> E_loss(train, calc, p), x))
obj_g! = (g, x) -> copyto!(g, Zygote.gradient(p -> E_loss(train, calc, p), x)[1])

res = optimize(obj_f, obj_g!, p0,
                Optim.GradientDescent(),
              Optim.Options(x_tol = 1e-15, f_tol = 1e-15, g_tol = 1e-6, show_trace = true))

            #   Optim.BFGS()

Eerrmin = Optim.minimum(res)
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

using PyPlot
figure()
scatter(Eref, Eace, c="red", alpha=0.4)
plot(-480:0.1:-478, -480:0.1:-478, lw=2, c="k", ls="--")
xlabel("Reference energy")
ylabel("ACE energy")
axis("square")
xlim([-480, -478])
ylim([-480, -478])
PyPlot.savefig("W_energy_fitting.png")