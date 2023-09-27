using EquivariantModels: CategoricalBasis, SList, i2val, val2i
using Polynomials4ML: evaluate, lux
using Lux, Random, Test 
using ACEbase.Testing: println_slim, print_tf 

##

# define elements (categories)
elements = ['a', 'b', 'c']

# test slist 

@info("Test SList")
slist = SList(elements)
for (i, c) in enumerate(elements)
   print_tf(@test (i2val(slist, i) == c) )
   print_tf(@test (val2i(slist, c) == i) )
end

println()

## simply a basis

@info("Testing Categorical Basis")
catbasis = CategoricalBasis(elements)
out = evaluate(catbasis, 'a')
println_slim(@test (out == [true, false, false] ))
out = evaluate(catbasis, 'b')
println_slim(@test (out == [false, true, false] ))
out = evaluate(catbasis, 'c')
println_slim(@test (out == [false, false, true] ))

##

# Luxify
@info("Testing Luxified CategoricalBasis")

l_catbasis = lux(catbasis)
ps, st = Lux.setup(MersenneTwister(1234), l_catbasis)
for c in elements
   out = evaluate(catbasis, c)
   l_out, st2 = l_catbasis(c, ps, st)
   println_slim(@test out == l_out)
end

