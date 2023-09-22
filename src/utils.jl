using Polynomials4ML: natural_indices


"""
_invmap(a::AbstractVector)
Return a dictionary that maps the elements of a to their indices
"""
function _invmap(a::AbstractVector)
   inva = Dict{eltype(a), Int}()
   for i = 1:length(a) 
      inva[a[i]] = i 
   end
   return inva 
end

"""
dropnames(namedtuple::NamedTuple, names::Tuple{Vararg{Symbol}})
Return a new NamedTuple with tuples elements in "names" dropped
"""
function dropnames(namedtuple::NamedTuple, names::Tuple{Vararg{Symbol}}) 
   keepnames = Base.diff_names(Base._nt_names(namedtuple), names)
   return NamedTuple{keepnames}(namedtuple)
end

"""
getspec1idx(spec1, bRnl, bYlm)
Return a vector of tuples of indices of spec1 w.r.t actual indices (i.e. 1, 2, 3, ...) of bRnl and bYlm
"""
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

"""
make_nlms_spec(bRnl, bYlm)
Return a vector of tuples of indices of spec1 w.r.t naural indices (i.e. (n = ..., l = ..., m = ...) ) of bRnl and bYlm
"""
function make_nlms_spec(bRn, bYlm;
            totaldegree::Int64 = -1,
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

"""
_nlms2b(nn, ll, mm)
Return a list of NamedTuples of (n, l, m) from nn, ll, mm
"""
function _nlms2b(nn, ll, mm)
    return [(n = n, l = l, m = m) for (n, l, m) in zip(nn, ll, mm)]
end

"""
get_natural_spec(spec, spec1p)
Get a readible version of spec
"""
function get_natural_spec(spec, spec1p)
   new_spec = []
   for t in spec
      push!(new_spec, [spec1p[i] for i in t])
   end
   return new_spec
end

"""
getspecnlm(spec1p, spec)
Get a readible version of spec
"""
function getspecnlm(spec1p, spec)
    new_spec = []
    for b in spec
        nlms = [spec1p[b1p_idx] for b1p_idx in b]
        push!(new_spec, nlms)
    end
    return new_spec
end

"""
LinearSearch(arr, t)
Return the index of t in arr, note that we use sort(t) == sort(x) instead of t == x
"""
function LinearSearch(arr, t)
   for (idx, x) in enumerate(arr)
       if sort(x) == sort(t) # length(setdiff(x,t)) == 0
           return idx
       end
   end
   @show t
   @error("No t found")
end

"""
specnlm2spec1p(spec_nlm)
From a list of AA specifications to all A specifications needed
"""
function specnlm2spec1p(spec_nlm)
    spec1p = union(spec_nlm...)
    lmax = [ spec1p[i].l for i = 1:length(spec1p) ] |> maximum
    nmax = [ spec1p[i].n for i = 1:length(spec1p) ] |> maximum
    return spec1p, lmax, nmax + 1
end

"""
closure(spec_nlm,filter)
Make a spec_nlm to be a "complete" set to be symmetrised w.r.t to the filter
"""
function closure(spec_nlm, filter)
   specnlm = Vector{Vector{NamedTuple}}()
   nl_list = []
   for spec in spec_nlm
      n_list = [ spec[i].n for i = 1:length(spec) ]
      l_list = [ spec[i].l for i = 1:length(spec) ]
      if (n_list, l_list) ∉ nl_list
         push!(nl_list, (n_list, l_list))
         push!(specnlm, _close(n_list, l_list, filter)...)
      end
   end
   return sort.(specnlm) |> unique
end

function _close(nn::Vector{Int64}, ll::Vector{Int64}, filter)
   spec = Vector{Vector{NamedTuple}}()
   mm = CartesianIndices(ntuple(i -> -ll[i]:ll[i], length(ll))) |> collect
   for m in mm 
      spec_tmp = [(n=nn[i], l = ll[i], m = m.I[i]) for i = 1:length(ll)]
      if filter(spec_tmp)
         push!(spec, spec_tmp)
      end
   end
   return spec |> unique
end

"""
degord2spec(;totaldegree, order, Lmax, radial_basis = legendre_basis, wL = 1, islong = true)
Return a list of AA specifications and A specifications
"""
function degord2spec(;totaldegree, order, Lmax, radial_basis = legendre_basis, wL = 1, islong = true)
   Rn = radial_basis(totaldegree)
   Ylm = CYlmBasis(totaldegree)

   spec1p = make_nlms_spec(Rn, Ylm; totaldegree = totaldegree, admissible = (br, by) -> br + wL * by.l <= totaldegree)
   spec1p = sort(spec1p, by = (x -> x.n + x.l * wL))
   spec1pidx = getspec1idx(spec1p, Rn, Ylm)

   # define sparse for n-correlations
   tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
   default_admissible = bb -> length(bb) == 0 || sum(b.n for b in bb) + wL * sum(b.l for b in bb) <= totaldegree

   # to construct SS, SD blocks
   filter_ = islong ? RPE_filter_long(Lmax) : RPE_filter(Lmax)

   specAA = gensparse(; NU = order, tup2b = tup2b, filter = filter_, 
                        admissible = default_admissible,
                        minvv = fill(0, order), 
                        maxvv = fill(length(spec1p), order), 
                        ordered = true)

   spec = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]
   # map back to nlm
   AAspec = getspecnlm(spec1p, spec)
   Aspec = specnlm2spec1p(AAspec)[1]
   return Aspec, AAspec # Aspecgetspecnlm(spec1p, spec)
end