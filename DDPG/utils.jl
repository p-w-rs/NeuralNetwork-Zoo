module Utils

export lecun_normal, get_env_details, a_rand, act!

using Flux, ChainRulesCore, Random, PyCall

gym = pyimport("gymnasium")
gym_wrappers = pyimport("gymnasium.wrappers")

function lecun_normal(rng::AbstractRNG, dims::Integer...; gain::Real=1)
  std = Float32(gain)*sqrt(1.0f0 / first(Flux.nfan(dims...))) # calculates the standard deviation based on the `fan_in` value
  return Flux.truncated_normal(rng, dims...; mean=0, std=std)
end

lecun_normal(dims::Integer...; kwargs...) = lecun_normal(Random.default_rng(), dims...; kwargs...)
lecun_normal(rng::AbstractRNG=default_rng(); init_kwargs...) = (dims...; kwargs...) -> lecun_normal(rng, dims...; init_kwargs..., kwargs...)

ChainRulesCore.@non_differentiable lecun_normal(::Any...)

function get_env_details(env_name)
  env = gym_wrappers.FlattenObservation(gym.make(env_name, continuous=true))
  state_dim = convert(Int, env.observation_space.shape[1])
  action_dim = convert(Int, env.action_space.shape[1])
  action_scl = env.action_space.high
  env.close()
  return state_dim, action_dim, action_scl
end

function a_rand(env)
  env.action_space.sample()
end

function act!(env)
  action = env.action_space.sample()
  state, reward, term, trunc, info = env.step(action)
  terminal = term || trunc
  reward = terminal ? 0 : reward
  return state, reward, terminal
end

function act!(env, action)
  state, reward, term, trunc, info = env.step(action)
  terminal = term || trunc
  reward = terminal ? 0 : reward
  return state, reward, terminal
end

end
