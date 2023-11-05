using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote 
using Polynomials4ML: LinearLayer, RYlmBasis, lux, legendre_basis
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA, simple_radial_basis
using JuLIP, Combinatorics, BenchmarkTools

rng = Random.MersenneTwister()

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
X = [ @SVector(randn(3)) for i in 1:56 ]
B = l_basis(X, ps_basis, st_basis)[1]

# now extend the above BB basis to a model
len_BB = length(B) 

model = append_layer(l_basis, WrappedFunction(t -> real(t)); l_name=:real)
model = append_layer(model, LinearLayer(len_BB, 1); l_name=:dot)
model = append_layer(model, WrappedFunction(t -> t[1]); l_name=:get1)
         
ps, st = Lux.setup(rng, model)


##

## === benchmarks ===
@info("evaluate")

model(X, ps, st)
Zygote.gradient(X -> model(X, ps, st)[1], X)[1]
Zygote.gradient(p -> model(X, p, st)[1], ps)[1]

@btime $model($X, $ps, $st) # 123.894 μs (23 allocations: 88.47 KiB)

@profview let model = model, X = X, ps = ps, st = st
   for _ = 1:15000
      model(X, ps, st)
   end
end


##

@btime Zygote.gradient(X -> $model(X, $ps, $st)[1], $X)[1] # 459.596 μs (622 allocations: 1.10 MiB)

@info("gradient w.r.t. X")
@profview let model = model, X = X, ps = ps, st = st
   for _ = 1:5000
      Zygote.gradient(X -> model(X, ps, st)[1], X)[1]
   end
end


##

@info("gradient w.r.t. parameter")
@profview let model = model, X = X, ps = ps, st = st
   for _ = 1:5000
      Zygote.gradient(p -> model(X, p, st)[1], ps)[1]
   end
end

@btime Zygote.gradient(p -> $model($X, p, $st)[1], $ps)[1] # 447.336 μs (624 allocations: 1.11 MiB)

##

using Optimisers, ReverseDiff

gsq = let model = model, st = st, ps = ps, X0 = X 
   p_vec, rest = destructure(ps) 
   normsq = x -> sum(abs2, x)
   p -> sum( normsq, 
            Zygote.gradient(X -> model(X, rest(p), st)[1], X0)[1] )
end

p_vec, _ = destructure(ps)
gsq(p_vec)

@btime ReverseDiff.gradient($gsq, $p_vec)
# 175 us for grad 
# 25 ms for grad-grad 
# ca factor 100. 

@profview let gsq = gsq, p_vec = p_vec
   for _ = 1:30
      ReverseDiff.gradient(gsq, p_vec)
   end
end
