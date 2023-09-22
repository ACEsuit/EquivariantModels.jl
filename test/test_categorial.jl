using EquivariantModels: CategoricalBasis
using Polynomials4ML: evaluate, lux
using Lux
using Random

# define elements (categories)
elements = ['a', 'b', 'c']

# simply a basis
CatBasis = CategoricalBasis(elements)
out = evaluate(CatBasis, 'a')


# Luxity
l_CatBasis = lux(CatBasis)
ps, st = Lux.setup(MersenneTwister(1234), l_CatBasis)
l_out, st = l_CatBasis('a', ps, st)


@assert out == l_out
