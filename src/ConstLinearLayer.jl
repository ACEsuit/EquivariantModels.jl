import ChainRulesCore: rrule
using LuxCore, LinearOperators
using LuxCore: AbstractExplicitLayer
using ObjectPools: unwrap, release!

struct ConstLinearLayer{T} <: AbstractExplicitLayer
   op::T
end

(l::ConstLinearLayer{T})(x::AbstractVector) where T = begin
   TT = promote_type(eltype(l.op[1]), eltype(x))
   C = zeros(TT, size(l.op, 1))
   genmul!(C, l.op, unwrap(x), *)
   release!(x)
   return C
end

(l::ConstLinearLayer{T})(x::AbstractMatrix) where T = begin
   TT = promote_type(eltype(l.op[1]), eltype(x))
   C = zeros(TT, (size(x, 1), size(l.op, 1)))
   genmul!(C, l.op, unwrap(x), *)
   release!(x)
   return C
end

(l::ConstLinearLayer)(x::AbstractArray, ps, st) = (l(x), st)

function rrule(::typeof(LuxCore.apply), l::ConstLinearLayer, x::AbstractArray, ps, st)
   val = l(x,ps,st)
   function pb(A)
      return NoTangent(), NoTangent(), l.op' * A[1], (op = A[1] * x',), NoTangent()
   end
   return val, pb
end

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


# === sparse matmul implementation ===

using SparseArrays: AbstractSparseMatrixCSC, nonzeros, rowvals, nzrange
using LinearAlgebra: Transpose

function genmul!(C, A::AbstractSparseMatrixCSC, B, mulop)
   size(A, 2) == size(B, 1) || throw(DimensionMismatch())
   size(A, 1) == size(C, 1) || throw(DimensionMismatch())
   size(B, 2) == size(C, 2) || throw(DimensionMismatch())
   nzv = nonzeros(A)
   rv = rowvals(A)
   fill!(C, zero(eltype(C)))
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


function genmul!(C, xA::Transpose{<:Any,<:AbstractSparseMatrixCSC}, B, mulop)
   A = xA.parent
   size(A, 2) == size(C, 1) || throw(DimensionMismatch())
   size(A, 1) == size(B, 1) || throw(DimensionMismatch())
   size(B, 2) == size(C, 2) || throw(DimensionMismatch())
   nzv = nonzeros(A)
   rv = rowvals(A)
   fill!(C, zero(eltype(C)))
   for k in 1:size(C, 2)
      @inbounds for col in 1:size(A, 2)
         tmp = zero(eltype(C))
         for j in nzrange(A, col)
            tmp += mulop(nzv[j], B[rv[j],k])
         end
      C[col,k] += tmp
      end
   end
   return C
end
