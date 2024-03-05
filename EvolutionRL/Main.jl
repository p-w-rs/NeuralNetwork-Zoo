using Evolutionary, Flux, StatsBase
using BSON: @save

include("Drone2DEnv.jl")
using .Drone2DEnv

include("NNLibExtra.jl")
using .NNLibExtra

env = Env()
const a_low = -1.0
const a_high = 1.0
const a_dim = env.act_dim
const s_dim = env.obs_dim
const h_dim = s_dim*a_dim*2

model = Chain(
	LSTM(s_dim => h_dim),
	LSTM(h_dim => h_dim),
	LSTM(h_dim => a_dim),
)
M = typeof(model)
construct = Flux.destructure(model)[2]

function loss(params::Vector{Float32})
	env = Env()
	policy = construct(params)
	mx, mn, p, avg, std = evaluate(env, policy, 5)
	return p
end

function evaluate(env::Env, policy::M, n::Int)
	rewards = zeros(Float32, n)
	for i in 1:n
		ps = zeros(Float32, s_dim)
		cs = reset!(env, i)
		r = 0f0
		d = false
		reward = 0f0
		while !d
			a = policy(cs)
			s, r, d = step!(env, a[:])
			ps, cs = d ? (zeros(Float32, s_dim), reset!(env)) : (cs, s)
			reward = r
		end
		rewards[i] = reward
	end
	return maximum(rewards), minimum(rewards), prod(rewards), mean(rewards), std(rewards)
end

result = Evolutionary.optimize(
	loss, Flux.destructure(model)[1],
	CMAES(),
	Evolutionary.Options(
		iterations=1000,
		show_trace=true,
		parallelization=:thread
	)
)
g = Evolutionary.minimizer(result)
@save "drone2d.bson" g



