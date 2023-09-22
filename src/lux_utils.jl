using Lux: Chain
using LuxCore: AbstractExplicitLayer

export concat_chain, append_layer

# overiding the iterate method of Chain
Base.iterate(chain::Chain, state = 1) = state > length(chain) ? nothing : chain[keys(chain)[state]]

model = Lux.Chain(((; zip(keys(chain_AA2B), values(chain_AA2B))...)..., dot = Polynomials4ML.LinearLayer(length(B[1]), 1))...)

# TODO: can we do it inplaces or it is not needed anyways?
append_layer(chain1::Chain, layer::AbstractExplicitLayer; l_name = Symbol("layer$(length(chain1) + 1)")) = Chain(; (; zip(keys(chain1), values(chain1))...)..., (; zip((l_name,), (layer, ))...)...)
concat_chain(chain1::Chain, chain2::Chain) = Chain(; (; zip(keys(chain1), values(chain1))...)..., (; zip(keys(chain2), keys(chain2))...)...)
