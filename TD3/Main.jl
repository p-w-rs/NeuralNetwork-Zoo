using Flux, ParameterSchedulers, StatsBase, ProgressMeter, CUDA
using BSON: @save
device = cpu#CUDA.functional() ? gpu : cpu

include("Gym.jl")
using .Gym

include("ReplayBuffer.jl")
using .ReplayBuffer

include("TD3.jl")
using .TD3

n = parse(Int, ARGS[1])


#---------------------------#
#-- Gym Environmnet Setup --#
#---------------------------#
env = Env("LunarLander-v2"; continuous=true, render_mode="rgb_array")
const a_low = env.low
const a_high = env.high
const a_dim = env.a_dim
const s_dim = env.s_dim


#----------------------#
#-- Hyper Parameters --#
#----------------------#
const max_t_steps = 100000

const γ = 0.99 # Reward discount factor
const ρ = 0.99 # Polyak averaging coefficient (for target network updates)
const ηθ = Triangle(λ0=0.0001, λ1=0.01, period=max_t_steps)
const ηϕ = Triangle(λ0=0.0002, λ1=0.02, period=max_t_steps)

const update_every = 16
const update_number = 1

const eval_every = div(max_t_steps, 50)
const eval_n = 20

const batch_size = 1024
const buffer_size = 1_000_000

const μ = (a_low + a_high) / 2
const σ = Exp(λ=(a_high[end] - a_low[end]) / 2, γ=1f0-(1f0/max_t_steps))
const ϵ(t) = (randn(Float32, (a_dim, 1)) .+ μ) .* σ(t)


#---------------------------------#
#-- Twin Delayed DDPG Algorithm --#
#---------------------------------#
function td3()
	global env, a_low, a_high, a_dim, s_dim, γ, ρ, η, max_t_steps, update_every, update_number, policy_delay, eval_every, eval_n, batch_size, buffer_size, device
	
	D = Buffer(s_dim, a_dim, buffer_size)
	μθ, μθₜ = Actor(s_dim, a_dim, a_low, a_high; mult=4)
	Qϕ, Qϕₜ = Critic(s_dim, a_dim; mult=6)

	optθ = Flux.setup(AdaGrad(ηθ(1)), μθ)
	optϕ = Flux.setup(AdaGrad(ηϕ(1)), Qϕ)

	fill_buffer(D)
	eval_policy = evaluate()
	s = reset!(env)
	@showprogress for t in 1:max_t_steps
		a = clamp.(μθ(s) + ϵ(t), a_low, a_high)
		s′, r, d = step!(env, a)
		store!(D, s, a, r, s′, d)
		s = d ? reset!(env) : s′
		
		t % update_every == 0 &&
			update!(D, μθ, μθₜ, optθ, Qϕ, Qϕₜ, optϕ, t)

		t % eval_every == 0 &&
			eval_policy(μθ, μθₜ, optθ, Qϕ, Qϕₜ, optϕ)
	end
	eval_policy(μθ, μθₜ, optθ, Qϕ, Qϕₜ, optϕ; render=true)
end

function update!(D, μθ, μθₜ, optθ, Qϕ, Qϕₜ, optϕ, t)
	global env, a_low, a_high, a_dim, s_dim, γ, ρ, η, max_t_steps, update_every, update_number, policy_delay, eval_every, eval_n, batch_size, buffer_size, device

	Flux.trainmode!(μθ); Flux.trainmode!(μθₜ); Flux.trainmode!(Qϕ); Flux.trainmode!(Qϕₜ);
	μθ, μθₜ, optθ, Qϕ, Qϕₜ, optϕ = (μθ, μθₜ, optθ, Qϕ, Qϕₜ, optϕ) |> device
	for j in 1:update_number
		B = batch!(D, batch_size)
		(s, a, r, s′, d) = B |> device

		# compute target q value as the minimum of the two caritic targets
		a′ = μθₜ(s′)
		y = r .+ γ.*(1 .- d).*minimum(Qϕₜ((s, a′), (s, a′)), dims=1)
		
		# update critic network
		grads = Flux.gradient(Qϕ) do Qϕ
			Flux.mse(Qϕ((s, a), (s, a)), vcat(y, y))
		end
		Flux.update!(optϕ, Qϕ, grads[1])
		Flux.adjust!(optϕ, ηϕ(t))
		
		# update actor network
		grads = Flux.gradient(μθ) do μθ
			a = μθ(s)
			-mean( minimum(Qϕ((s, a), (s, a)), dims=1) )
		end
		Flux.update!(optθ, μθ, grads[1])
		Flux.adjust!(optθ, ηθ(t))

		#update Qϕₜ <- ρ*Qϕₜ + (1-ρ)*Qϕ
		for (param_t, param) in zip(Flux.params(Qϕₜ), Flux.params(Qϕ))
		    param_t .= ρ.*param_t .+ (1-ρ).*param
		end
		
		#update μθₜ <- ρ*μθₜ + (1-ρ)*μθ
		for (param_t, param) in zip(Flux.params(μθₜ), Flux.params(μθ))
		    param_t .= ρ.*param_t .+ (1-ρ).*param
		end
	end
	μθ, μθₜ, optθ, Qϕ, Qϕₜ, optϕ = (μθ, μθₜ, optθ, Qϕ, Qϕₜ, optϕ) |> cpu
	Flux.testmode!(μθ); Flux.testmode!(μθₜ); Flux.testmode!(Qϕ); Flux.testmode!(Qϕₜ);
end

function fill_buffer(D)
	global env, a_low, a_high, a_dim, s_dim, γ, ρ, η, max_t_steps, update_every, update_number, policy_delay, eval_every, eval_n, batch_size, buffer_size, device

	s = reset!(env)
	@showprogress for t in 1:batch_size*10
		a = action(env)
		s′, r, d = step!(env, a)
		store!(D, s, a, r, s′, d)
		s = d ? reset!(env) : s′
	end
end

function evaluate()
	global env, a_low, a_high, a_dim, s_dim, γ, ρ, η, max_t_steps, update_every, update_number, policy_delay, eval_every, eval_n, batch_size, buffer_size, device
	local_static_mean = -Inf

	function run_sim_loop(μθ, μθₜ, optθ, Qϕ, Qϕₜ, optϕ; render=false)
		rewards = zeros(eval_n)
		for j in 1:eval_n
			s = reset!(env)
			d = false
			reward = 0.0
			while !d
				render && render!(env)
				a = μθ(s)
				s′, r, d = step!(env, a)
				s = d ? reset!(env) : s′
				reward += r
			end
			rewards[j] = reward
		end
		mn, st = mean(rewards), std(rewards)

		println()
		println("Eval Mean: $mn +/- $st")
		if mn >= local_static_mean
			println("New Best Policy!")
			local_static_mean = mn
			@save "save/models.bson" μθ μθₜ optθ Qϕ Qϕₜ optϕ
			@save "save/best$n.txt" mn st
		end
		
		return local_static_mean
	end

	return run_sim_loop
end


#----------#
#-- MAIN --#
#----------#
td3()