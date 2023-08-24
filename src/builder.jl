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

export luxchain_constructor, luxchain_constructor_multioutput, equivariant_luxchain_constructor

# a should be a set, return the position of each element
function _invmap(a::AbstractVector)
   inva = Dict{eltype(a), Int}()
   for i = 1:length(a) 
      inva[a[i]] = i 
   end
   return inva 
end

function dropnames(namedtuple::NamedTuple, names::Tuple{Vararg{Symbol}}) 
   keepnames = Base.diff_names(Base._nt_names(namedtuple), names)
   return NamedTuple{keepnames}(namedtuple)
end

function getspec1idx(spec1, bRnl, bYlm)
   spec1idx = Vector{Tuple{Int, Int}}(undef, length(spec1))
   spec_Rnl = natural_indices(bRnl); 
   spec_Rnl = [(n = i, ) for i in spec_Rnl]
   inv_Rnl = _invmap(spec_Rnl)
   
   spec_Ylm = natural_indices(bYlm); inv_Ylm = _invmap(spec_Ylm)

   spec1idx = Vector{Tuple{Int, Int}}(undef, length(spec1))
   for (i, b) in enumerate(spec1)
      spec1idx[i] = (inv_Rnl[dropnames(b, (:m, :l))], inv_Ylm[(l=b.l, m=b.m)])
   end
   return spec1idx
end

function make_nlms_spec(bRn, bYlm;
            totaldegree::Integer = -1,
            admissible = nothing, 
            nnuc = 0)
   
   spec_Rn = natural_indices(bRn)
   spec_Ylm = natural_indices(bYlm)
   
   spec1 = []
   for (iR, br) in enumerate(spec_Rn), (iY, by) in enumerate(spec_Ylm)
      if admissible(br, by) 
         push!(spec1, (n = br, l = by.l, m = by.m))
      end
   end
   return spec1 
end

function _nlms2b(nn, ll, mm)
    return [(n = n, l = l, m = m) for (n, l, m) in zip(nn, ll, mm)]
end

function get_natural_spec(spec, spec1p)
   new_spec = []
   for t in spec
      push!(new_spec, [spec1p[i] for i in t])
   end
   return new_spec
end

function LinearSearch(arr, t)
   for (idx, x) in enumerate(arr)
       if sort(x) == sort(t) # length(setdiff(x,t)) == 0
           return idx
       end
   end
   @error("No t found")
end


function rand_rot() 
   K = randn(3,3)
   K = K - K' 
   return exp(K) 
end

function _rpi_A2B_matrix(cgen::Union{Rot3DCoeffs{L,T},Rot3DCoeffs_real{L,T},Rot3DCoeffs_long{L,T}},
                         spec) where {L,T}
   # allocate triplet format
   Irow, Jcol = Int[], Int[]
   if typeof(cgen) <: Rot3DCoeffs_long
      vals = SYYVector{L,(L+1)^2}[]
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
            push!(Irow, idxB)
            push!(Jcol, idxAA)
            push!(vals, U[irow, icol])
         end
      end
      push!(nnllset,(nn,ll))
   end
   # create CSC: [   triplet    ]  nrows   ncols
   return sparse(Irow, Jcol, vals, idxB, length(spec))
end


function getspecnlm(spec1p, spec)
    new_spec = []
    for b in spec
        nlms = [spec1p[b1p_idx] for b1p_idx in b]
        push!(new_spec, nlms)
    end
    return new_spec
end

P4ML = Polynomials4ML
RPI_filter(L) = bb -> (length(bb) == 0) || ((abs(sum(b.m for b in bb)) <= L) && iseven(sum(b.l for b in bb)+L))
RPI_filter_long(L) = bb -> (length(bb) == 0) || (abs(sum(b.m for b in bb)) <= L)

# This constructor builds a lux chain that maps a configuration to the corresponding B^L vector 
# or [B^0, B^1, ... B^L] vector, depending on whether islong == true
# What can be adjusted in its input are: (1) total polynomial degree; (2) correlation order; (3) largest L
# (4) weight of the order of spherical harmonics; (5) specified radial basis
function luxchain_constructor(totdeg,ν,L; wL = 1, Rn = legendre_basis(totdeg), islong = true)
   if islong
      filter = RPI_filter_long(L)
      cgen = Rot3DCoeffs_long(L)
   else
      filter = RPI_filter(L)
      cgen = Rot3DCoeffs(L)
   end
   
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

   # C - A2Bmap
   luxchain = Chain(xnx = l_xnx, embed = l_embed, A = l_bA , AA = l_bAA, BB = WrappedFunction(x -> C * x))#, rAA = WrappedFunction(ComplexF64))
   ps, st = Lux.setup(MersenneTwister(1234), luxchain)
                        
   return luxchain, ps, st
end

# mapping from long vector to spherical matrix
using RepLieGroups.O3: ClebschGordan
cg = ClebschGordan(ComplexF64)
function cgmatrix(L1,L2)
   cgm = zeros((2L1+1)*(2L2+1),(L1+L2+1)^2)
   for (i,(p,q)) in enumerate(collect(Iterators.product(-L1:L1, L2:-1:-L2)))
      ν = p+q
      for l = 0:L1+L2
         if abs(ν)<=l
            position = ν + l + 1
            # cgm[i, l^2+position] = (-1)^q * cg(L1,p,L2,q,l,ν) 
            cgm[i, l^2+position] = cg(L1,p,L2,q,l,ν) 
            # cgm[i, l^2+position] = (-1)^q * 3 / 2 / sqrt(π * (2l+1)) * cg(L1,0,L2,0,l,0) * cg(L1,p,L2,q,l,ν) 
         end
      end
   end
   return sparse(cgm)
end

# This constructor builds a lux chain that maps a configuration to a LONG B^L vector ([B^0, B^1, ... B^L]), 
# and then all equivariant basis.
# What can be adjusted in its input are: (1) total polynomial degree; (2) correlation order; (3) largest L
# (4) weight of the order of spherical harmonics; (5) specified radial basis
function equivariant_luxchain_constructor(totdeg,ν,L; wL = 1, Rn = legendre_basis(totdeg))

   filter = RPI_filter_long(L)
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

# This constructor builds a lux chain that maps a configuration to the corresponding B^0 to B^L vectors 
# What can be adjusted in its input are: (1) total polynomial degree; (2) correlation order; (3) largest L
# (4) weight of the order of spherical harmonics; (5) specified radial basis
function luxchain_constructor_multioutput(totdeg,ν,L; wL = 1, Rn = legendre_basis(totdeg))
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
      filter = RPI_filter(l)
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
   filter = RPI_filter_long(L)
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
   
   # C - A2Bmap
   luxchain = Chain(xnx = l_xnx, embed = l_embed, A = l_bA , AA = l_bAA, BB = l_seperate)#, rAA = WrappedFunction(ComplexF64))
   ps, st = Lux.setup(MersenneTwister(1234), luxchain)
   
   return luxchain, ps, st
end

# This constructor builds a lux chain that maps a configuration to B^0, B^1, ... B^L first, 
# and then deduces all equivariant basis.
# What can be adjusted in its input are: (1) total polynomial degree; (2) correlation order; (3) largest L
# (4) weight of the order of spherical harmonics; (5) specified radial basis
function equivariant_luxchain_constructor_new(totdeg,ν,L; wL = 1, Rn = legendre_basis(totdeg))
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
      filter = RPI_filter(l)
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
   filter = RPI_filter_long(L)
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
