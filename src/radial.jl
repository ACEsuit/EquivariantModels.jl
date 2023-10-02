using Polynomials4ML: natural_indices, ScalarPoly4MLBasis, lux
using LuxCore: AbstractExplicitContainerLayer, AbstractExplicitLayer

struct Radial_basis <:AbstractExplicitContainerLayer{(:Rnl,)}
    Rnl::AbstractExplicitLayer
    # make it meta or just leave it as a NameTuple ?
    Radialspec::Vector{NamedTuple}
 end

Radial_basis(Rnl::AbstractExplicitLayer, spec_Rnl::Union{Vector{Int}, UnitRange{Int64}}) = 
         Radial_basis(Rnl, [(n = i, ) for i in spec_Rnl])

Radial_basis(Rnl::AbstractExplicitLayer) = 
         try 
            Radial_basis(Rnl,natural_indices(Rnl.basis)) 
         catch 
            try 
               Radial_basis(Rnl,natural_indices(Rnl.layers.poly.basis)) 
            catch
               error("The specification of this Radial_basis should be given explicitly!")
            end
         end

# it is in its current form just for the purpose of testing - a more specific example can be found in forces.jl
function simple_radial_basis(basis::ScalarPoly4MLBasis,f_cut::Function=r->1,f_trans::Function=r->1; spec = nothing)
   if isnothing(spec)
      try 
         spec = natural_indices(basis)
      catch 
         error("The specification of this Radial_basis should be given explicitly!")
      end
   end
   
   return Radial_basis(Chain(trans = WrappedFunction(xx -> [f_trans(norm(x)) for x in xx]), 
                cutoff = WrappedFunction(xx -> f_cut.(xx)),
                poly = lux(basis), ), spec)
end