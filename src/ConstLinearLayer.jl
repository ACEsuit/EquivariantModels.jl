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

function _linear_operator_L(L, C, pos, len)
    if L == 0
       T = ComplexF64
       fL = let C=C, idx=pos#, T=T
           (res, aa) -> mul!(res, C, aa[idx]);# try; mul!(res, C, aa[idx]); catch; mul!(zeros(T,size(C,1)), C, aa[idx]); end
           end
    else
       T = SVector{2L+1,ComplexF64} 
       fL = let C=C, idx=pos#, T=T
           (res, aa) -> begin
               res[:] .= C * aa[idx]
           end
       end
    end
    return LinearOperator{T}(size(C,1), len, false, false, fL, nothing, nothing; S = Vector{T})
 end