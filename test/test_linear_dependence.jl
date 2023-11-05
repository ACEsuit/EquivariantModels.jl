using EquivariantModels, StaticArrays, Test, Polynomials4ML, LinearAlgebra
using ACEbase.Testing: print_tf
using EquivariantModels: Radial_basis, degord2spec
using Polynomials4ML:lux
using RepLieGroups

fcut(rcut::Float64,pin::Int=2,pout::Int=2) = r -> (r < rcut ? abs( (r/rcut)^pin - 1)^pout : 0)
ftrans(r0::Float64=.0,p::Int=2) = r -> ( (1+r0)/(1+r) )^p
rcut = 5.5
L = 0

@info("Testing linear independence of the L = $L equivariant basis")

for ord = 1:3
    for totdeg = 4:8
        radial = EquivariantModels.simple_radial_basis(MonoBasis(totdeg-1),r->sqrt(r)*fcut(rcut)(r),ftrans())

        Aspec, AAspec =  degord2spec(radial; totaldegree = totdeg, 
                                order = ord, 
                                Lmax = L, islong = false)
        luxchain, ps, st = equivariant_model(AAspec, radial, L; islong = false)
        F(X) = luxchain(X, ps, st)[1]

        T = L == 0 ? ComplexF64 : SVector{2L+1,ComplexF64}
        A = zeros(T,size(luxchain.layers.BB.op,1),10*size(luxchain.layers.BB.op,1))
        for i = 1:10*size(luxchain.layers.BB.op,1)
            local x = [ @SVector(rand(3)) for i in 1:10 ]
            x = [ State(rr = x[i]) for i in 1:length(x) ]
            A[:,i] = F(x)
        end
        print_tf(@test rank(A) == size(luxchain.layers.BB.op,1))
    end
end
println()