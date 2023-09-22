using Lux: Chain
using LuxCore: AbstractExplicitLayer

export concat_chain, append_layer

append_layer(chain1::Chain, layer::AbstractExplicitLayer; l_name = Symbol("layer$(length(chain1) + 1)")) = Chain(; (; zip(keys(chain1), values(chain1))...)..., (; zip((l_name,), (layer, ))...)...)
concat_chain(chain1::Chain, chain2::Chain) = Chain(; (; zip(keys(chain1), values(chain1))...)..., (; zip(keys(chain2), keys(chain2))...)...)
