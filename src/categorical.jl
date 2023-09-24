using Polynomials4ML
using Polynomials4ML: AbstractPoly4MLBasis
using StaticArrays: SVector

export CategoricalBasis

# -------------------------

struct SList{N, T}
   list::SVector{N, T}

   function SList{N, T}(list::SVector{N, T})  where {N, T} 
      if isabstracttype(T)
         error("`SList` can only contain a single type")
      end
      return new(list)
   end
end

SList(list::AbstractArray) = SList(SVector(list...))
SList(args...) = SList(SVector(args...))
SList(list::SVector{N, T}) where {N, T} = SList{N, T}(list)

Base.length(list::SList) = length(list.list)
Base.rand(list::SList) = rand(list.list)

i2val(list::SList, i::Integer) = list.list[i]

# should we implement Base.getindex instead / in addition? 

Base.iterate(list::SList, args...) = iterate(list.list, args...)

function val2i(list::SList, val)
   for j = 1:length(list)
      if list.list[j] == val
         return j
      end
   end
   error("val = $val not found in this list")
end


# write_dict(list::SList{N,T}) where {N, T} = 
#       Dict( "__id__" => "ACE_SList", 
#                  "T" => write_dict(T),
#               "list" => list.list )

# function read_dict(::Val{:ACE_SList}, D::Dict) 
#    list = D["list"]
#    T = read_dict(D["T"])
#    svector = SVector{length(list), T}((T.(list))...)
#    return SList(svector)
# end



# -------------------------

@doc raw"""
`CategoricalBasis` : defines the discrete 1p basis 
```math 
   \phi_q(u) = \delta{u, U_q},
```
where ``U_q, q = 1, \dots, Q`` are finitely many possible values that the 
variable ``u`` may take. Suppose, e.g., we allow the values `[:a, :b, :c]`, 
then 
```julia 
P = CategoricalBasis([:a, :b, :c]; varsym = :u, idxsym = :q)
evaluate(P, State(u = :a))     # Bool[1, 0, 0]
evaluate(P, State(u = :b))     # Bool[0, 1, 0]
evaluate(P, State(u = :c))     # Bool[0, 0, 1]
```
If we evaluate it with an unknown state we get an error: 
```julia 
evaluate(P, State(u = :x))   
# Error : val = x not found in this list
```

Warning : the list of categories is internally stored as an SVector 
which means that lookup scales linearly with the number of categories
"""
struct CategoricalBasis{LEN, T} <: AbstractPoly4MLBasis
   categories::SList{LEN, T}
   meta::Dict{String, Any}
end

Base.length(B::CategoricalBasis) = length(B.categories)

CategoricalBasis(categories::AbstractArray, meta = Dict{String, Any}() ) = 
      CategoricalBasis(SList(categories), meta)

      
#Polynomials4ML._outsym(x::Char) = :char
Polynomials4ML._outsym(x::T) where T = :T
Polynomials4ML._out_size(basis::CategoricalBasis{LEN, T}, x::T) where {LEN, T} = (LEN,)
Polynomials4ML._out_size(basis::CategoricalBasis{LEN, T}, x::Vector{T}) where {LEN, T} = (length(x),LEN)
Polynomials4ML._valtype(basis::CategoricalBasis{LEN, T}, x::Union{T,Vector{T}}) where {LEN, T} = Bool

# should the output be somethign like this?
# struct Ei 
#    i::Int
# end
# getindex(e::Ei, j::Int) = (j == e.i)

# the next few functions need to be adapted to P4ML / Lux  

function Polynomials4ML.evaluate(basis::CategoricalBasis{LEN, T}, X::T) where {LEN,T}   
   # some abstract vector 
   A = Vector{Bool}(undef, LEN)
   return evaluate!(A, basis, X)
end

function Polynomials4ML.evaluate(basis::CategoricalBasis{LEN, T}, X::Vector{T}) where {LEN,T}
   A = Matrix{Bool}(undef, length(X), LEN)
   for i = 1:length(X)
      A[i,:] = evaluate!(A[i,:], basis, X[i])
   end
   return A
end

function Polynomials4ML.evaluate!(A, basis::CategoricalBasis{LEN, T}, X::T) where {LEN,T}  
   fill!(A, false)
   A[val2i(basis.categories, X)] = true
   return A
end

function Polynomials4ML.evaluate!(A, basis::CategoricalBasis{LEN, T}, X::Vector{T}) where {LEN,T}  
   fill!(A, false)
   for i = 1:length(X)
      A[i,val2i(basis.categories, X[i])] = true
   end
   return A
end

# natural_indices 
Polynomials4ML.natural_indices(basis::CategoricalBasis) = basis.categories.list

Base.rand(basis::CategoricalBasis) = rand(basis.categories)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
# probably we don't need the rest, but keep around for now

# do 
# function get_spec(basis::CategoricalBasis, i)
#    return NamedTuple{(_isym(basis),)}((i2val(basis.categories, i),))
# end

# get_spec(basis::CategoricalBasis) = [ get_spec(basis, i) for i = 1:length(basis) ]


# write_dict(B::CategoricalBasis) = 
#       Dict( "__id__" => "ACE_CategoricalBasis", 
#             "categories" => write_dict(B.categories), 
#             "VSYM" => String(_varsym(B)), 
#             "ISYM" => String(_isym(B)), 
#             "label" => B.label)

# read_dict(::Val{:ACE_CategoricalBasis}, D::Dict)  = 
#    CategoricalBasis( read_dict(D["categories"]), 
#                   Symbol(D["VSYM"]), Symbol(D["ISYM"]), 
#                   D["label"] )
