using EquivariantModels
using EquivariantModels: degord2spec, specnlm2spec1p

totaldegree = 6
order = 3 
maxL = 0 

Aspec, AAspec = degord2spec(; totaldegree = 6, 
                                  order = 3, 
                                  Lmax = 0 )

chain_A2B, ps, st = equivariant_model(AAspec, maxL)
# 
# A = randn(length(Aspec))
# 
# chain_A2B(A, ps, st)
