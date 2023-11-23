import ChainRulesCore: rrule
using LuxCore, LinearOperators
using LuxCore: AbstractExplicitLayer

struct ConstLinearLayer{T} <: AbstractExplicitLayer
    op::T
end

(l::ConstLinearLayer{T})(x::AbstractVector) where T = l.op * x

(l::ConstLinearLayer{T})(x::AbstractMatrix) where T = begin
    Tmp = l(x[1,:])
    for i = 2:size(x,1)
        Tmp = [Tmp l(x[i,:])]
    end
    return Tmp'
 end

 (l::ConstLinearLayer)(x::AbstractArray,ps,st) = (l(x), st)

 # NOTE: the following rrule is kept because there is an issue with SparseArray
function rrule(::typeof(LuxCore.apply), l::ConstLinearLayer, x::AbstractVector)
    val = l(x)
    function pb(A)
        return NoTangent(), NoTangent(), l.op' * A[1], (op = A[1] * x',), NoTangent()
    end
    return val, pb
end

function rrule(::typeof(LuxCore.apply), l::ConstLinearLayer, x::AbstractArray,ps,st)
    val = l(x,ps,st)
    function pb(A)
        return NoTangent(), NoTangent(), l.op' * A[1], (op = A[1] * x',), NoTangent()
    end
    return val, pb
end

# fallback to generic matmul
(l::ConstLinearLayer)(x) = l.op * x

##

# === connection with ChainRulesCore === 
# sparse matrix interface
function rrule(::typeof(LuxCore.apply), l::ConstLinearLayer{<: AbstractSparseMatrixCSC}, x, ps, st)
   val = l(x,ps,st)
   function pb(A)
      #out = similar(x)
      T = eltype(A[1][1])
      out = zeros(T, size(x))
      genmul!(out, l.op', A[1], dot)
      return NoTangent(), NoTangent(), out, NoTangent(), NoTangent()
   end
   return val, pb
end

# fallback
function rrule(::typeof(LuxCore.apply), l::ConstLinearLayer, x, ps, st)
   val = l(x,ps,st)
   function pb(A)
      return  NoTangent(), NoTangent(), vec_matmul(l.op', A[1]), NoTangent(), NoTangent()
   end
   return val, pb
end

vec_matmul(A::AbstractMatrix{<: Number}, B::AbstractVecOrMat{<: Number}) = A * B
vec_matmul(A::AbstractArray, B::AbstractVecOrMat{<: SVector}) = sum([dot(conj(ai), bj) for ai in A[:, k], bj in B[k, :]] for k = 1:size(A, 2) )


##

# === construction with LinearOperator === 
function _linear_operator_L(L, C, pos, len)
   if L == 0
      T = ComplexF64
      fL = let C=C, idx=pos#, T=T
        (res, aa) -> genmul!(res, C, aa[idx], *);# try; mul!(res, C, aa[idx]); catch; mul!(zeros(T,size(C,1)), C, aa[idx]); end
      end
   else
      T = SVector{2L+1,ComplexF64} 
      fL = let C=C, idx=pos#, T=T
         (res, aa) -> genmul!(res, C, aa[idx], *)
      end
   end
   return LinearOperator{T}(size(C,1), len, false, false, fL, nothing, nothing; S = Vector{T})
end

# === sparse matmul implementation from ACE.jl ===
# https://github.com/ACEsuit/ACE.jl/blob/main/src/symmbasis.jl

function genmul!(C, A::AbstractSparseMatrixCSC, B, mulop)
   size(A, 2) == size(B, 1) || throw(DimensionMismatch())
   size(A, 1) == size(C, 1) || throw(DimensionMismatch())
   size(B, 2) == size(C, 2) || throw(DimensionMismatch())
   nzv = nonzeros(A)
   rv = rowvals(A)
   fill!(C, zero(_valtype(A,B)))
   for k in 1:size(C, 2)
     @inbounds for col in 1:size(A, 2)
            αxj = B[col,k]
            for j in nzrange(A, col)
               C[rv[j], k] += mulop(nzv[j], αxj)
            end
         end
   end
   return C
end


function genmul!(C, xA::Adjoint{<:Any,<:AbstractSparseMatrixCSC}, B, mulop)
   A = xA.parent
   size(A, 2) == size(C, 1) || throw(DimensionMismatch())
   size(A, 1) == size(B, 1) || throw(DimensionMismatch())
   size(B, 2) == size(C, 2) || throw(DimensionMismatch())
   nzv = nonzeros(A)
   rv = rowvals(A)
   fill!(C, zero(_valtype(A,B)))
   for k in 1:size(C, 2)
      @inbounds for col in 1:size(A, 2)
         tmp = zero(_valtype(A,B))
         for j in nzrange(A, col)
            tmp += mulop(nzv[j], B[rv[j],k])
         end
      C[col,k] += tmp
      end
   end
   return C
end
