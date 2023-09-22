using LinearAlgebra
using SparseArrays: SparseMatrixCSC, sparse
using RepLieGroups.O3: Rot3DCoeffs, Rot3DCoeffs_real, Rot3DCoeffs_long, re_basis, SYYVector
using Polynomials4ML: legendre_basis, RYlmBasis, natural_indices, degree
using Polynomials4ML.Utils: gensparse
using Lux: WrappedFunction
using Lux
using Random
using Polynomials4ML
using StaticArrays

export equivariant_model, equivariant_SYY_model, equivariant_luxchain_constructor, equivariant_luxchain_constructor_new

P4ML = Polynomials4ML

RPE_filter(L) = bb -> (length(bb) == 0) || ((abs(sum(b.m for b in bb)) <= L) && iseven(sum(b.l for b in bb)+L))
RPE_filter_long(L) = bb -> (length(bb) == 0) || (abs(sum(b.m for b in bb)) <= L)

"""
_rpi_A2B_matrix(cgen::Union{Rot3DCoeffs{L,T},Rot3DCoeffs_real{L,T},Rot3DCoeffs_long{L,T}},
                spec::Vector{Vector{NamedTuple}})
Return a sparse matrix for symmetrisation of AA basis of spec with equivariance specified by cgen
"""
function _rpi_A2B_matrix(cgen::Union{Rot3DCoeffs{L,T}, Rot3DCoeffs_real{L,T}, Rot3DCoeffs_long{L,T}}, spec) where {L,T}
   # allocate triplet format
   Irow, Jcol = Int[], Int[]
   if typeof(cgen) <: Rot3DCoeffs_long
      vals = SYYVector{L,(L+1)^2,ComplexF64}[]
   else
      vals =  L == 0 ? Float64[] : SVector{2L+1,ComplexF64}[]
   end
   # count the number of PI basis functions = number of rows
   idxB = 0
   # loop through all (zz, kk, ll) tuples; each specifies 1 to several B
   nnllset = []
   for i = 1:length(spec)
      # get the specification of the ith basis function, which is a tuple/vec of NamedTuples
      pib = spec[i]
      # skip it unless all m are zero, because we want to consider each
      # (nn, ll) block only once.
      # if !all(b.m == 0 for b in pib)
      #    continue
      # end
      # But we can not do this anymore for L≥1, so I add an nnllset
      
      # get the rotation-coefficients for this basis group
      # the bs are the basis functions corresponding to the columns
      
      nn = SVector([onep.n for onep in pib]...)
      ll = SVector([onep.l for onep in pib]...) # get a SVector of ll index
      
      if (nn,ll) in nnllset; continue; end

      # get the Mll indices and coeffs
      U, Mll = re_basis(cgen, ll)
      # conver the Mlls into basis functions (NamedTuples)
      
      rpibs = [_nlms2b(nn, ll, mm) for mm in Mll]
      
      if size(U, 1) == 0; continue; end
      # loop over the rows of Ull -> each specifies a basis function
      for irow = 1:size(U, 1)
         idxB += 1
         # loop over the columns of U / over brows
         for (icol, bcol) in enumerate(rpibs)
            # look for the index of basis bcol in spec
            bcol = sort(bcol)
            idxAA = LinearSearch(spec, bcol)
            if !isnothing(idxAA)
               push!(Irow, idxB)
               push!(Jcol, idxAA)
               push!(vals, U[irow, icol])
            end
         end
      end
      push!(nnllset,(nn,ll))
   end
   # create CSC: [   triplet    ]  nrows   ncols
   return sparse(Irow, Jcol, vals, idxB, length(spec))
end

## === Start building the chains ===

# TODO: symmetry group O(d)?
"""
xx2AA(spec_nlm, d=3, categories=[]; radial_basis=legendre_basis)
Construct a lux chain that maps a configuration to the corresponding the AA basis
spec_nlm: Specification of the AA bases
d: Input dimension
categories : A list of categories
radial_basis : specified radial basis, default using P4ML.legendre_basis
"""
function xx2AA(spec_nlm, d=3, categories=[]; radial_basis = legendre_basis) # Configuration to AA bases - this is what all chains have in common
   # from spec_nlm to all possible spec1p
   spec1p, lmax, nmax = specnlm2spec1p(spec_nlm)
   dict_spec1p = Dict([spec1p[i] => i for i = 1:length(spec1p)])
   Ylm = CYlmBasis(lmax)
   Rn = radial_basis(nmax)
   
   if !isempty(categories)
      # Read categories from x
      cat(x) = [ x[i].cat for i = 1:length(x) ]
      # Define categorical bases
      # δs = CateBasis(categories) # TODO: this is not yet in P4ML ??
   end
   
   spec1pidx = isempty(categories) ? getspec1idx(spec1p, Rn, Ylm) : getspec1idx(spec1p, Rn, Ylm, δs)
   bA = P4ML.PooledSparseProduct(spec1pidx)
   
   Spec = [ [dict_spec1p[spec_nlm[k][j]] for j = 1:length(spec_nlm[k])] for k = 1:length(spec_nlm) ]
   bAA = P4ML.SparseSymmProd(Spec)
   
   # wrapping into lux layers
   l_Rn = P4ML.lux(Rn)
   l_Ylm = P4ML.lux(Ylm)
   # l_δs = P4ML.lux(δs)
   l_bA = P4ML.lux(bA)
   l_bAA = P4ML.lux(bAA)
   
   # @assert Polynomials4ML.reconstruct_spec(l_bAA.basis) == Spec
   # The output of l_bAA may not be in the same order as Spec
   # Here we generate a permutation mapping to ensure this
   
   Spec_after = Polynomials4ML.reconstruct_spec(l_bAA.basis)
   dict = Dict([Spec_after[i] => i for i = 1 : length(Spec_after)])
   pos = [ dict[sort(Spec[i])] for i = 1:length(Spec) ]
   
   # formming model with Lux Chain
   _norm(x) = norm.(x)
   
   l_xnx = Lux.Parallel(nothing; normx = WrappedFunction(_norm), x = WrappedFunction(identity))
   l_embed = Lux.Parallel(nothing; Rn = l_Rn, Ylm = l_Ylm)
   
   luxchain = Chain(xnx = l_xnx, embed = l_embed, A = l_bA , AA = l_bAA, AA_sort = WrappedFunction(x -> x[pos]))
   ps, st = Lux.setup(MersenneTwister(1234), luxchain)
      
   return luxchain, ps, st
end

"""
equivariant_model(spec_nlm, d=3, categories=[]; radial_basis=legendre_basis)
Construct a lux chain that maps a configuration to the corresponding B^0 to B^L vectors, where L is the equivariance level
spec_nlm: Specification of the AA bases
L : Largest equivariance level
categories : A list of categories
radial_basis : specified radial basis, default using P4ML.legendre_basis
"""
function equivariant_model(spec_nlm, L::Int64, d=3, categories=[]; radial_basis=legendre_basis, group="O3", islong=true)
   # first filt out those unfeasible spec_nlm
   filter_init = islong ? RPE_filter_long(L) : RPE_filter(L)
   spec_nlm = spec_nlm[findall(x -> filter_init(x) == 1, spec_nlm)]
   
   # sort!(spec_nlm, by = x -> length(x))
   spec_nlm = closure(spec_nlm,filter_init)
   
   luxchain_tmp, ps_tmp, st_tmp = EquivariantModels.xx2AA(spec_nlm, d, categories; radial_basis = radial_basis)
   F(X) = luxchain_tmp(X, ps_tmp, st_tmp)[1]

   if islong
   # initialize C and spec_nlm
      C = Vector{Any}(undef, L+1)
      pos = Vector{Any}(undef, L+1)
   
      for l = 0:L
         filter = RPE_filter(l)
         cgen = Rot3DCoeffs(l) # TODO: this should be made group related

         tmp = spec_nlm[findall(x -> filter(x) == 1, spec_nlm)]
         C[l+1] = _rpi_A2B_matrix(cgen, tmp)
         pos[l+1] = findall(x -> filter(x) == 1, spec_nlm) # [ dict[tmp[j]] for j = 1:length(tmp)]
      end
   else
      cgen = Rot3DCoeffs(L) # TODO: this should be made group related
      C = _rpi_A2B_matrix(cgen, spec_nlm)
   end
   
   l_sym = islong ? Lux.Parallel(nothing, [WrappedFunction(x -> C[i] * x[pos[i]]) for i = 1:L+1]... ) : WrappedFunction(x -> C * x)

   # C - A2Bmap
   # luxchain = Chain(xx2AA = WrappedFunction(x -> F(x)), BB = l_sym)
   luxchain = Chain(xx2AA = luxchain_tmp, BB = l_sym)
   
   ps, st = Lux.setup(MersenneTwister(1234), luxchain)
   
   return luxchain, ps, st
end

# more constructors equivariant_model
equivariant_model(totdeg::Int64, ν::Int64, L::Int64, d=3, categories=[]; radial_basis=legendre_basis, group="O3", islong=true) = 
     equivariant_model(degord2spec(;totaldegree = totdeg, order = ν, Lmax=L, radial_basis = radial_basis, islong = islong)[2], L, d, categories; radial_basis, group, islong)

# With the _close function, the input could simply be an nnlllist (nlist,llist)
equivariant_model(nn::Vector{Int64}, ll::Vector{Int64}, L::Int64, d=3, categories=[]; radial_basis = legendre_basis, group = "O3", islong = true) = begin
   filter = islong ? RPE_filter_long(L) : RPE_filter(L)
   equivariant_model(_close(nn, ll, filter), L, d, categories; radial_basis, group, islong)
end

# ===== Codes that we might remove later =====

## The following are SYYVector-related codes - which we might want to either use or get rid of someday...

# This constructor builds a lux chain that maps a configuration to the corresponding B^L vector 
# or [B^0, B^1, ... B^L] vector, depending on whether islong == true
# What can be adjusted in its input are: (1) total polynomial degree; (2) correlation order; (3) largest L
# (4) weight of the order of spherical harmonics; (5) specified radial basis

function equivariant_SYY_model(spec_nlm, L::Int64, d=3, categories=[]; radial_basis=legendre_basis, group="O3")
   filter_init = RPE_filter_long(L)
   spec_nlm = spec_nlm[findall(x -> filter_init(x) == 1, spec_nlm)]
   
   # sort!(spec_nlm, by = x -> length(x))
   spec_nlm = closure(spec_nlm, filter_init)
   
   luxchain_tmp, ps_tmp, st_tmp = EquivariantModels.xx2AA(spec_nlm, d, categories; radial_basis = radial_basis)
   F(X) = luxchain_tmp(X, ps_tmp, st_tmp)[1]
   
   cgen = Rot3DCoeffs_long(L) # TODO: this should be made group related
   C = _rpi_A2B_matrix(cgen, spec_nlm)
   l_sym = WrappedFunction(x -> C * x)
   
   # C - A2Bmap
   # luxchain = Chain(xx2AA = WrappedFunction(x -> F(x)), BB = l_sym)
   luxchain = Chain(xx2AA = luxchain_tmp, BB = l_sym)
   
   ps, st = Lux.setup(MersenneTwister(1234), luxchain)
   
   return luxchain, ps, st
end

equivariant_SYY_model(totdeg::Int64, ν::Int64, L::Int64, d=3, categories=[]; radial_basis = legendre_basis,group = "O3") = 
   equivariant_SYY_model(degord2spec(;totaldegree = totdeg, order = ν, Lmax = L, radial_basis = radial_basis, islong=true)[2], L , d, categories; radial_basis, group)

equivariant_SYY_model(nn::Vector{Int64}, ll::Vector{Int64}, L::Int64, d=3, categories=[]; radial_basis=legendre_basis, group="O3") = 
   equivariant_SYY_model(_close(nn, ll, RPE_filter_long(L)), L, d, categories ; radial_basis, group)
   
## TODO: The following should eventually go into ACEhamiltonians.jl rather than this package

# mapping from long vector to spherical matrix
using RepLieGroups.O3: ClebschGordan
cg = ClebschGordan(ComplexF64)
function cgmatrix(L1, L2)
   cgm = zeros((2L1+1) * (2L2+1),(L1+L2+1)^2)
   for (i,(p,q)) in enumerate(collect(Iterators.product(-L1:L1, L2:-1:-L2)))
      ν = p+q
      for l = 0:L1+L2
         if abs(ν)<=l
            position = ν + l + 1
            cgm[i, l^2+position] = (-1)^q * cg(L1, p, L2, q, l, ν) 
            # cgm[i, l^2+position] = cg(L1,p,L2,q,l,ν) 
            # cgm[i, l^2+position] = (-1)^q * sqrt( (2L1+1) * (2L2+1) ) / 2 / sqrt(π * (2l+1)) * cg(L1,0,L2,0,l,0) * cg(L1,p,L2,q,l,ν) 
         end
      end
   end
   return sparse(cgm)
end

# This constructor builds a lux chain that maps a configuration to a LONG B^L vector ([B^0, B^1, ... B^L]), 
# and then all equivariant basis.
# What can be adjusted in its input are: (1) total polynomial degree; (2) correlation order; (3) largest L
# (4) weight of the order of spherical harmonics; (5) specified radial basis
function equivariant_luxchain_constructor(totdeg, ν, L; wL = 1, Rn = legendre_basis(totdeg))

   filter = RPE_filter_long(L)
   cgen = Rot3DCoeffs_long(L)

   Ylm = CYlmBasis(totdeg)
   
   spec1p = make_nlms_spec(Rn, Ylm; totaldegree = totdeg, admissible = (br, by) -> br + wL * by.l <= totdeg)
   spec1p = sort(spec1p, by = (x -> x.n + x.l * wL))
   spec1pidx = getspec1idx(spec1p, Rn, Ylm)
   
   # define sparse for n-correlations
   tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
   default_admissible = bb -> length(bb) == 0 || sum(b.n for b in bb) + wL * sum(b.l for b in bb) <= totdeg
   
   specAA = gensparse(; NU = ν, tup2b = tup2b, filter = filter, 
                        admissible = default_admissible,
                        minvv = fill(0, ν), 
                        maxvv = fill(length(spec1p), ν), 
                        ordered = true)

   spec = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]
   # map back to nlm
   spec_nlm = getspecnlm(spec1p, spec)
   
   C = _rpi_A2B_matrix(cgen, spec_nlm)

   # acemodel with lux layers
   bA = P4ML.PooledSparseProduct(spec1pidx)
   bAA = P4ML.SparseSymmProd(spec)

   # wrapping into lux layers
   l_Rn = P4ML.lux(Rn)
   l_Ylm = P4ML.lux(Ylm)
   l_bA = P4ML.lux(bA)
   l_bAA = P4ML.lux(bAA)

   # formming model with Lux Chain
   _norm(x) = norm.(x)

   l_xnx = Lux.Parallel(nothing; normx = WrappedFunction(_norm), x = WrappedFunction(identity))
   l_embed = Lux.Parallel(nothing; Rn = l_Rn, Ylm = l_Ylm)
   
   l1l2set = [(l1,l2) for l1 = 0:L for l2 = 0:L-l1]
   cgmat_set = [cgmatrix(l1,l2) for (l1,l2) in l1l2set ]
   
   _block(x,cgmat,l1,l2) = reshape.(Ref(cgmat) .* [ x[j][1:size(cgmat)[2]] for j = 1:length(x) ], 2l1+1, 2l2+1)
   _dropzero(x) = x[findall(X -> X>1e-12, norm.(x))]
   
   l_seperate = Lux.Parallel(nothing, [WrappedFunction(x -> _dropzero(_block(x,cgmat_set[i],l1l2set[i][1],l1l2set[i][2]))) for i = 1:length(cgmat_set)]... )

   # C - A2Bmap
   luxchain = Chain(xnx = l_xnx, embed = l_embed, A = l_bA , AA = l_bAA, BB = WrappedFunction(x -> C * x), blocks = l_seperate)#, rAA = WrappedFunction(ComplexF64))
   ps, st = Lux.setup(MersenneTwister(1234), luxchain)
                        
   return luxchain, ps, st
end

# This constructor builds a lux chain that maps a configuration to B^0, B^1, ... B^L first, 
# and then deduces all equivariant basis.
# What can be adjusted in its input are: (1) total polynomial degree; (2) correlation order; (3) largest L
# (4) weight of the order of spherical harmonics; (5) specified radial basis
function equivariant_luxchain_constructor_new(totdeg, ν, L; wL = 1, Rn = legendre_basis(totdeg))
   Ylm = CYlmBasis(totdeg)

   spec1p = make_nlms_spec(Rn, Ylm; totaldegree = totdeg, admissible = (br, by) -> br + wL * by.l <= totdeg)
   spec1p = sort(spec1p, by = (x -> x.n + x.l * wL))
   spec1pidx = getspec1idx(spec1p, Rn, Ylm)

   # define sparse for n-correlations
   tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
   default_admissible = bb -> length(bb) == 0 || sum(b.n for b in bb) + wL * sum(b.l for b in bb) <= totdeg

   # initialize C and spec_nlm
   C = Vector{Any}(undef,L+1)
   # spec_nlm = Vector{Any}(undef,L+1)
   spec = Vector{Any}(undef,L+1)

   # Spec = []
   
   for l = 0:L
      filter = RPE_filter(l)
      cgen = Rot3DCoeffs(l)
      specAA = gensparse(; NU = ν, tup2b = tup2b, filter = filter, 
                        admissible = default_admissible,
                        minvv = fill(0, ν), 
                        maxvv = fill(length(spec1p), ν), 
                        ordered = true)

      spec[l+1] = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]
      # Spec = Spec ∪ spec

      spec_nlm = getspecnlm(spec1p, spec[l+1])
      C[l+1] = _rpi_A2B_matrix(cgen, spec_nlm)
   end
   # return C, spec, spec_nlm
   # acemodel with lux layers
   bA = P4ML.PooledSparseProduct(spec1pidx)
   
   #
   filter = RPE_filter_long(L)
   specAA = gensparse(; NU = ν, tup2b = tup2b, filter = filter, 
                     admissible = default_admissible,
                     minvv = fill(0, ν), 
                     maxvv = fill(length(spec1p), ν), 
                     ordered = true)

   Spec = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]
   dict = Dict([Spec[i] => i for i = 1:length(Spec)])
   pos = [ [dict[spec[k][j]] for j = 1:length(spec[k])] for k = 1:L+1 ]
   # @show typeof(pos)
   # return C, spec, Spec
   
   bAA = P4ML.SparseSymmProd(Spec)
   
   # wrapping into lux layers
   l_Rn = P4ML.lux(Rn)
   l_Ylm = P4ML.lux(Ylm)
   l_bA = P4ML.lux(bA)
   l_bAA = P4ML.lux(bAA)
   
   # formming model with Lux Chain
   _norm(x) = norm.(x)
   
   l_xnx = Lux.Parallel(nothing; normx = WrappedFunction(_norm), x = WrappedFunction(identity))
   l_embed = Lux.Parallel(nothing; Rn = l_Rn, Ylm = l_Ylm)
   l_seperate = Lux.Parallel(nothing, [WrappedFunction(x -> C[i] * x[pos[i]]) for i = 1:L+1]... )
   
   function _vectorize(cc)
      ccc = []
      for i = 1:length(cc)
         ccc = [ccc; cc[i]...]
      end
      return sparse(complex.(ccc))
   end
   
   function _block(x,l1,l2)
      @assert l1+l2 <= length(x) - 1
      init = iseven(l1+l2) ? 0 : 1
      fea_set = init:2:l1+l2
      A = Vector{Any}(undef,l1+l2+1)
      for i = 0:l1+l2
         if i in fea_set
            A[i+1] = x[i+1]
         else
            A[i+1] = [zeros(ComplexF64,2i+1)]
         end
      end
      cc = Iterators.product(A...) |> collect
      c = [ _vectorize(cc[i]) for i = 1:length(cc) ]
      return reshape.( Ref(cgmatrix(l1,l2)) .* c, 2l1+1, 2l2+1 )
   end
   
   # l_condensed = WrappedFunction(x -> _block(x,0,1))
   l_condensed = Lux.Parallel(nothing, [WrappedFunction(x -> _block(x,l1,l2)) for l1 in 0:L for l2 in 0:L-l1]... )
   
   
   # C - A2Bmap
   luxchain = Chain(xnx = l_xnx, embed = l_embed, A = l_bA , AA = l_bAA, BB = l_seperate, inter = WrappedFunction(x -> [x[i] for i = 1:length(x)]), B = l_condensed)#, rAA = WrappedFunction(ComplexF64))
   ps, st = Lux.setup(MersenneTwister(1234), luxchain)
   
   return luxchain, ps, st
end