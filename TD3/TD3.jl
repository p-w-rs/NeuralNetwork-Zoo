module TD3

using Flux, ChainRulesCore, Random

export Actor, Critic

function Actor(s_dim, a_dim, a_low, a_high; mult=1)
	h_dim = s_dim*a_dim*mult
	return (
		make_actor(s_dim, a_dim, h_dim, a_high),
		make_actor(s_dim, a_dim, h_dim, a_high)
	)
end

function make_actor(s_dim, a_dim, h_dim, scale)
	return Chain(
	    Dense(s_dim => h_dim, selu; init=lecun_normal), AlphaDropout(0.5),
	    Dense(h_dim => h_dim, selu; init=lecun_normal), AlphaDropout(0.5),
	    Dense(h_dim => h_dim, selu; init=lecun_normal), AlphaDropout(0.5),
	    Dense(h_dim => a_dim, hardtanh; init=lecun_normal), Flux.Scale(scale, false, identity)
	)
end

function Critic(s_dim, a_dim; mult=1)
	h_dim = s_dim*a_dim*mult
	return (
		make_critic(s_dim, a_dim, h_dim),
		make_critic(s_dim, a_dim, h_dim)
	)
end

function make_critic(s_dim, a_dim, h_dim)
	Parallel(vcat; 
		Q1=Chain(
			Parallel(vcat;
				s=Chain(Dense(s_dim => h_dim, selu; init=lecun_normal), AlphaDropout(0.5)),
				a=Chain(Dense(a_dim => h_dim, selu; init=lecun_normal), AlphaDropout(0.5))
			),
			Dense(h_dim*2, h_dim, selu; init=lecun_normal), AlphaDropout(0.5),
			Dense(h_dim, h_dim, selu; init=lecun_normal), AlphaDropout(0.5),
			Dense(h_dim, 1, identity; init=lecun_normal)
		),
		Q2=Chain(
			Parallel(vcat,
				Chain(Dense(s_dim => h_dim, selu; init=lecun_normal), AlphaDropout(0.5)),
				Chain(Dense(a_dim => h_dim, selu; init=lecun_normal), AlphaDropout(0.5))
			),
			Dense(h_dim*2, h_dim, selu; init=lecun_normal), AlphaDropout(0.5),
			Dense(h_dim, h_dim, selu; init=lecun_normal), AlphaDropout(0.5),
			Dense(h_dim, 1, identity; init=lecun_normal)
		)
	)
end

function lecun_normal(rng::AbstractRNG, dims::Integer...; gain::Real=1)
  std = Float32(gain)*sqrt(1.0f0 / first(Flux.nfan(dims...))) # calculates the standard deviation based on the `fan_in` value
  return Flux.truncated_normal(rng, dims...; mean=0, std=std)
end

lecun_normal(dims::Integer...; kwargs...) = lecun_normal(Random.default_rng(), dims...; kwargs...)
lecun_normal(rng::AbstractRNG=default_rng(); init_kwargs...) = (dims...; kwargs...) -> lecun_normal(rng, dims...; init_kwargs..., kwargs...)

ChainRulesCore.@non_differentiable lecun_normal(::Any...)


end