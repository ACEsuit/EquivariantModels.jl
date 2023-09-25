using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote 
using Polynomials4ML: LinearLayer, RYlmBasis, lux 
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA
using JuLIP, Combinatorics

rng = Random.MersenneTwister()

rcut = 5.5 
maxL = 0
L = 0
Aspec, AAspec = degord2spec(; totaldegree = 6, 
                                  order = 3, 
                                  Lmax = 0, )
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

luxchain, ps, st = equivariant_model(new_AAspec, L; categories=allcats)

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
X = (Rs, Z0S)

out, st = luxchain(X, ps, st)
