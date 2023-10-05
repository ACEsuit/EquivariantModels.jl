import ChainRulesCore: rrule
using LuxCore
using LuxCore: AbstractExplicitLayer

struct ConstLinearLayer <: AbstractExplicitLayer # where {in_dim,out_dim,T}
    op #::AbstractMatrix{T}  
    position::Union{Vector{Int64}, UnitRange{Int64}}
end

ConstLinearLayer(op) = ConstLinearLayer(op,1:size(op,2))
# ConstLinearLayer(op, pos::Union{Vector{Int64}, UnitRange{Int64}}) = ConstLinearLayer(op,pos)

(l::ConstLinearLayer)(x::AbstractVector) = l.op * x[l.position]

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
        return NoTangent(), NoTangent(), l.op' * A[1], (op = A[1] * x',), NoTangent()
    end
    return val, pb
end

(l::ConstLinearLayer)(x::AbstractArray,ps,st) = (l(x), st)

function rrule(::typeof(LuxCore.apply), l::ConstLinearLayer, x::AbstractArray,ps,st)
    val = l(x,ps,st)
    function pb(A)
        return NoTangent(), NoTangent(), l.op' * A[1], (op = A[1] * x',), NoTangent()
    end
    return val, pb
end

# function rrule(::typeof(LuxCore.apply), l::ConstLinearLayer, x::AbstractMatrix, ps, st)
#     val = l(x, ps, st)
#     function pb(A)
#        return NoTangent(), NoTangent(), l.op' * A[1], (op = A[1] * x',), NoTangent()
#     end
#     return val, pb
#  end