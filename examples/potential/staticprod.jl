import Polynomials4ML: _static_prod_ed, _pb_grad_static_prod

function _static_prod_ed(b::NTuple{N, Any}) where N
   b2 = b[2:N]
   p2, g2 = _static_prod_ed(b2)
   return b[1] * p2, tuple(p2, ntuple(i -> b[1] * g2[i], N-1)...)
end

function _static_prod_ed(b::NTuple{1, Any})
   return b[1], (one(T),)
end

function _pb_grad_static_prod(∂::NTuple{N, Any}, b::NTuple{N, Any}) where N
    ∂2 = ∂[2:N]
    b2 = b[2:N]
    p2, g2, u2 = _pb_grad_static_prod(∂2, b2)
    return b[1] * p2, 
           tuple(p2, ntuple(i -> b[1] * g2[i], N-1)...), 
           tuple(sum(∂2 .* g2), ntuple(i -> ∂[1] * g2[i] + b[1] * u2[i], N-1)...)
 end
 
function _pb_grad_static_prod(∂::NTuple{1, Any}, b::NTuple{1, Any})
   return b[1], (one(T),), (zero(T),)
end
 