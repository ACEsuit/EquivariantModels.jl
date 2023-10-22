using Polynomials4ML: natural_indices, ScalarPoly4MLBasis, lux
using LuxCore: AbstractExplicitContainerLayer, AbstractExplicitLayer

 struct Radial_basis{T <: AbstractExplicitLayer} <:AbstractExplicitContainerLayer{(:Rnl, )}
    Rnl::T
    # make it meta or just leave it as a NamedTuple ?
    Radialspec::Vector #{NamedTuple} #TODO: double check this...
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
function simple_radial_basis(basis::ScalarPoly4MLBasis,f_cut::Function=r->1,f_trans::Function=r->r; spec = nothing)
   if isnothing(spec)
      try 
         spec = natural_indices(basis)
      catch 
         error("The specification of this Radial_basis should be given explicitly!")
      end
   end

   _norm(x) = norm(x.rr)
   return Radial_basis(Chain(trans = WrappedFunction(x -> f_trans.(_norm.(x))), evaluation = Lux.BranchLayer(poly = lux(basis), cutoff = WrappedFunction(x -> f_cut.(x))), env = WrappedFunction(x -> x[1].*x[2]), ), spec)

end