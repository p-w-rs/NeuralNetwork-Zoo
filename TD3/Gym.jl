module Gym

using PyCall

gym = pyimport("gymnasium")
gym_wrappers = pyimport("gymnasium.wrappers")

export Env, action, reset!, step!, render!, close!

struct Env
	env
	s_dim
	a_dim
	high
	low
end

function Env(env_name; kwargs...)::Env
	env = gym_wrappers.FlattenObservation(gym.make(
		env_name; kwargs...
	))
	s_dim = convert(Int, env.observation_space.shape[end])
	a_dim = convert(Int, env.action_space.shape[end])
	high = env.action_space.high
	low = env.action_space.low
	return Env(env, s_dim, a_dim, high, low)
end

function action(env::Env)
	env.env.action_space.sample()
end

function reset!(env::Env)
	state, info = env.env.reset()
	return state
end

function step!(env::Env, action)
	state, reward, terminal, trunc, info = env.env.step(action[:])
	done = terminal || trunc
	reward = done ? 0 : reward
	return state, reward, done
end

function render!(env::Env)
	env.env.render()
end

function close!(env::Env)
	env.env.close()
end

end