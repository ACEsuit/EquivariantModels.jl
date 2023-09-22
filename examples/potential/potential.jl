using EquivariantModels
using EquivariantModels: degord2spec_nlm, specnlm2spec1p

totaldegree = 6
order = 3 
maxL = 0 

AAspec = degord2spec_nlm(totaldegree, order, maxL)
Aspec, maxn, maxl = specnlm2spec1p(AAspec)

# Change interface to 
# Aspec, AAspec = degord2spec_nlm(; totaldegree = 6, 
#                                   order = 3, 
#                                   Lmax = 0 )


chain_A2B, ps, st = equivariant_model(spec, maxL)

A = randn(length(Aspec))

chain_A2B(A, ps, st)