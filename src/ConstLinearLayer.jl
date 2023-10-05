import ChainRulesCore: rrule
using LuxCore
using LuxCore: AbstractExplicitLayer

struct ConstLinearLayer <: AbstractExplicitLayer
    op
end

(l::ConstLinearLayer)(x::AbstractVector) = l.op * x[1:size(l.op,2)]

(l::ConstLinearLayer)(x::AbstractMatrix) = begin
    Tmp = l(x[1,:])
    for i = 2:size(x,1)
        Tmp = [Tmp l(x[i,:])]
    end
    return Tmp'
 end

 (l::ConstLinearLayer)(x::AbstractArray,ps,st) = (l(x), st)

 # NOTE: the following rrule is kept because there is a issue with SparseArray
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