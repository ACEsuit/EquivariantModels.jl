using LuxCore, LinearOperators
using LuxCore: AbstractExplicitLayer
using ObjectPools: unwrap, release!
using SparseArrays: AbstractSparseMatrixCSC, nonzeros, rowvals, nzrange
using LinearAlgebra: Adjoint

import ChainRulesCore: rrule

struct ConstLinearLayer{T} <: AbstractExplicitLayer
   op::T
end

# === evaluation interface === 
_valtype(op::AbstractMatrix{<: Number}, x) = promote_type(eltype(op), eltype(x))
_valtype(op::AbstractMatrix{<: AbstractVector}, x::AbstractArray{<: Number}) = SVector{length(op[1]), promote_type(eltype(op[1]), eltype(x[1][1]))}
_valtype(op::AbstractMatrix{<: AbstractVector}, x::AbstractArray{<: AbstractVector}) = promote_type(eltype(op[1]), eltype(x[1][1]))

(l::ConstLinearLayer)(x::AbstractArray, ps, st) = (l(x), st)

# sparse linear op interface
(l::ConstLinearLayer{<: AbstractSparseMatrixCSC})(x::AbstractVector) = begin
   TT =_valtype(l.op, x)
   out = zeros(TT, size(l.op, 1))
   genmul!(out, l.op, unwrap(x), *)
   release!(x)
   return out
end

(l::ConstLinearLayer{<: AbstractSparseMatrixCSC})(x::AbstractMatrix) = begin
   TT = _valtype(l.op, x)
   out = zeros(TT, (size(l.op, 1), size(x, 2)))
   genmul!(out, l.op, unwrap(x), *)
   release!(x)
   return out
end

# fallback to generic matmul
(l::ConstLinearLayer)(x) = l.op * x

##

# === connection with ChainRulesCore === 
# sparse matrix interface
function rrule(::typeof(LuxCore.apply), l::ConstLinearLayer{<: AbstractSparseMatrixCSC}, x, ps, st)
   val = l(x,ps,st)
   function pb(A)
      T = eltype(x)
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
   fill!(C, zero(_valtype(A, B)))
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


function genmul!(C, xA::Adjoint{<:Any, <:AbstractSparseMatrixCSC}, B, mulop)
   A = xA.parent
   size(A, 2) == size(C, 1) || throw(DimensionMismatch())
   size(A, 1) == size(B, 1) || throw(DimensionMismatch())
   size(B, 2) == size(C, 2) || throw(DimensionMismatch())
   nzv = nonzeros(A)
   rv = rowvals(A)
   TAB = _valtype(A, B)
   fill!(C, zero(TAB))
   for k in 1:size(C, 2)
      @inbounds for col in 1:size(A, 2)
         tmp = zero(TAB)
         for j in nzrange(A, col)
            tmp += mulop(nzv[j], B[rv[j],k])
         end
      C[col,k] += tmp
      end
   end
   return C
end