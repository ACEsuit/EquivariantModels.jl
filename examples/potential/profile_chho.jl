using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote 
using Polynomials4ML: LinearLayer, RYlmBasis, lux, legendre_basis
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA, simple_radial_basis
using JuLIP, Combinatorics, BenchmarkTools, DecoratedParticles

rng = Random.MersenneTwister()

## 

# === Model/SitePotential construction ===
rcut = 5.5 
maxL = 0
totdeg = 8
ord = 2

fcut(rcut::Float64,pin::Int=2,pout::Int=2) = r -> (r < rcut ? abs( (r/rcut)^pin - 1)^pout : 0)
ftrans(r0::Float64=2.0,p::Int=2) = r -> ( (1+r0)/(1+r) )^p
radial = simple_radial_basis(legendre_basis(totdeg);isState=true)

Aspec, AAspec = degord2spec(radial; totaldegree = totdeg, 
                              order = ord, 
                              Lmax = maxL, wL = 1.)#, rSH = true)

l_basis, ps_basis, st_basis = equivariant_model(AAspec, radial, maxL; islong = false, isState = true)#, rSH = true);
# X = [ @SVector(rand(3)) for i in 1:56 ]
X = [ State(rr = @SVector(randn(3))) for i in 1:56 ]
B = l_basis(X, ps_basis, st_basis)[1]

# now extend the above BB basis to a model
len_BB = length(B) 

model = append_layer(l_basis, WrappedFunction(t -> real(t)); l_name=:real);
model = append_layer(model, LinearLayer(len_BB, 1); l_name=:dot);
model = append_layer(model, WrappedFunction(t -> t[1]); l_name=:get1);
         
ps, st = Lux.setup(rng, model);


##

## === benchmarks ===
@info("evaluate")

model(X, ps, st);
@btime $model($X, $ps, $st); # w.o. state 11.666 μs (22 allocations: 83.59 KiB) 
                             # w. state  11.916 μs (23 allocations: 85.05 KiB)

@profview let model = model, X = X, ps = ps, st = st
   for _ = 1:15000
      model(X, ps, st)
   end
end


##
@info("gradient w.r.t. X")

Zygote.gradient(X -> model(X, ps, st)[1], X)[1]

@btime Zygote.gradient(X -> $model(X, $ps, $st)[1], $X)[1]; # w.o. state 188.709 μs (1470 allocations: 881.34 KiB)
                                                            # w. state  904.833 μs (8200 allocations: 1.73 MiB)
                                                            # I think it just because the gradient should be a 3-vector rather than a state ?

@profview let model = model, X = X, ps = ps, st = st
   for _ = 1:5000
      Zygote.gradient(X -> model(X, ps, st)[1], X)[1]
   end
end

##

@info("gradient w.r.t. parameter")
Zygote.gradient(p -> model(X, p, st)[1], ps)[1]
@btime Zygote.gradient(p -> $model($X, p, $st)[1], $ps)[1] # w.o. state 190.708 μs (1472 allocations: 880.28 KiB)
                                                           # w. state  905.417 μs (8200 allocations: 1.73 MiB)

@profview let model = model, X = X, ps = ps, st = st
   for _ = 1:5000
      Zygote.gradient(p -> model(X, p, st)[1], ps)[1]
   end
end

##

using Optimisers, ReverseDiff

p_vec, rest = destructure(ps) 
X0 = X
Zygote.gradient(X -> model(X, rest(p_vec), st)[1], X0)[1][1][1]

gsq = let model = model, st = st, ps = ps, X0 = X 
   p_vec, rest = destructure(ps) 
   normsq = x -> sum(abs2, x)
   p -> sum( normsq, 
            # Zygote.gradient(X -> model(X, rest(p), st)[1], X0)[1] ) # FOR NON STATE CASE
            [(Zygote.gradient(X -> model(X, rest(p), st)[1], X0)[1])[i][1].rr for i in 1:length(X0)] )
end

p_vec, _ = destructure(ps)
gsq(p_vec)

@btime ReverseDiff.gradient($gsq, $p_vec)
# 190 us for grad            %    w. state 900 us for grad
# 21 ms for grad-grad        %    w. state 2.516 s for grad-grad
# ca factor 100.             %    ca factor 2800 ??

@profview let gsq = gsq, p_vec = p_vec
   for _ = 1:10
      ReverseDiff.gradient(gsq, p_vec)
   end
end
