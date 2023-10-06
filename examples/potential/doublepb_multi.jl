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
cats = AtomicNumber.([:W, :Cu, :Ni, :Fe, :Al])

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
iCu = [5, 12]; iNi = [3, 8]; iAl = [10]; iFe = [6];
at.Z[iCu] .= cats[2]; at.Z[iNi] .= cats[3]; at.Z[iAl] .= cats[4]; at.Z[iFe] .= cats[5];
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

model = append_layers(luxchain, get1 =  WrappedFunction(t -> real.(t)), dot = LinearLayer(length(B), 1), get2 = WrappedFunction(t -> t[1]))

ps, st = Lux.setup(MersenneTwister(1234), model)

model(X, ps, st)

# testing derivative (forces)
g = Zygote.gradient(X -> model(X, ps, st)[1], X)[1][1]
grad_model(X, ps, st) = Zygote.gradient(_X -> model(_X, ps, st)[1], X)[1]

F(Rs) = model([Rs, Z0S], ps, st)[1]
dF(Rs) = Zygote.gradient(rs -> model([rs, Z0S], ps, st)[1], Rs)[1]

# === define toy loss ===
function loss(X, p)
   ps = _rest(p)
   g = grad_model(X, ps, st)[1]
   return sum(norm.(g))
end

p_vec, _rest = destructure(ps)

# === override useful functions, will be fixed from P4ML end ===
using Polynomials4ML

function Polynomials4ML._pullback_evaluate(∂A, basis::Polynomials4ML.PooledSparseProduct{NB}, BB::Polynomials4ML.TupMat) where {NB}
   nX = size(BB[1], 1)
   TA = promote_type(eltype.(BB)..., eltype(∂A))
   # @show TA
   ∂BB = ntuple(i -> zeros(TA, size(BB[i])...), NB)
   Polynomials4ML._pullback_evaluate!(∂BB, ∂A, basis, BB)
   return ∂BB
end

# === reverse over reverse ===
using ReverseDiff
g1 = ReverseDiff.gradient(_p -> loss(X, _p), p_vec)

Zygote.gradient(_p -> loss(X, _p), p_vec)

using ACEbase
ACEbase.Testing.fdtest( 
         _p -> loss(X, _p), 
         _p -> Zygote.gradient(__p -> loss(X, __p), _p), 
         p_vec )
##