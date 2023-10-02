import ChainRulesCore: rrule
using LuxCore
using LuxCore: AbstractExplicitLayer

struct ConstLinearLayer{T} <: AbstractExplicitLayer # where {in_dim,out_dim,T}
    W::AbstractMatrix{T}  
    in_dim::Integer
    out_dim::Integer
end

ConstLinearLayer(W::AbstractMatrix{T}) where T = ConstLinearLayer(W,size(W,2),size(W,1))

(l::ConstLinearLayer)(x::AbstractVector) = l.in_dim == length(x) ? l.W * x : error("x has a wrong length!")

(l::ConstLinearLayer)(x::AbstractMatrix) = begin
    Tmp = l(x[1,:])
    for i = 2:size(x,1)
        Tmp = [Tmp l(x[i,:])]
    end
    return Tmp'
 end

function rrule(::typeof(LuxCore.apply), l::ConstLinearLayer, x::AbstractVector)
    val = l(x)
    function pb(A)
        return NoTangent(), NoTangent(), l.W' * A[1], (W = A[1] * x',), NoTangent()
    end
    return val, pb
end

(l::ConstLinearLayer)(x::AbstractArray,ps,st) = l(x)

function rrule(::typeof(LuxCore.apply), l::ConstLinearLayer, x::AbstractArray,ps,st)
    val = l(x)
    function pb(A)
        return NoTangent(), NoTangent(), l.W' * A[1], (W = A[1] * x',), NoTangent()
    end
    return val, pb
end

# function rrule(::typeof(LuxCore.apply), l::ConstLinearLayer, x::AbstractMatrix, ps, st)
#     val = l(x, ps, st)
#     function pb(A)
#        return NoTangent(), NoTangent(), l.W' * A[1], (W = A[1] * x',), NoTangent()
#     end
#     return val, pb
#  end