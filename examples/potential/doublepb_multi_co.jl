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

## === testing reverse over reverse ===

using ReverseDiff
g1 = ReverseDiff.gradient(_p -> loss(Rs, _p), p_vec)
@show g1

# Zygote.gradient(_p -> loss(X, _p), p_vec)

##
using ACEbase
ACEbase.Testing.fdtest( 
         _p -> loss(Rs, _p), 
         _p -> ReverseDiff.gradient(__p -> loss(Rs, __p), _p), 
         p_vec )
##