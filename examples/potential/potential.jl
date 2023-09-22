using EquivariantModels
using StaticArrays
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA

totaldegree = 6
order = 3 
maxL = 0 

Aspec, AAspec = degord2spec(; totaldegree = 6, 
                                  order = 3, 
                                  Lmax = 0 )

chain_xx2AA, ps1, st1 = xx2AA(AAspec, maxL);

chain_AA2B, ps2, st2 = equivariant_model(AAspec, maxL)

# A = randn(length(Aspec))
# 
# chain_A2B(A, ps, st)

X = [ @SVector(rand(3)) for i in 1:10 ]

chain_xx2AA(X, ps1, st1)

chain_AA2B(X, ps2, st2)
