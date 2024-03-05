#=== Package Imports and Globals ===#
using PyCall, Flux, CUDA, ParameterSchedulers, Random, ProgressMeter, Plots
using StatsBase: mean
using BSON: @save

include("optimiser.jl")
using .SchedOpt

include("replayBuffer.jl")
using .ReplayBuffer

gym = pyimport("gymnasium")
gym_wrappers = pyimport("gymnasium.wrappers")

device = CUDA.functional() ? gpu : cpu

#=== Functions ===#

function create_model(in_dim, out_dim, mult)
    h_dim = mult*in_dim*out_dim
    init=Flux.glorot_normal
    return Chain(
        Dense(in_dim => h_dim, mish; init=init), Dropout(0.2),
        Dense(h_dim => h_dim, mish; init=init), Dropout(0.2),
        Dense(h_dim => h_dim, mish; init=init), Dropout(0.2),
        Dense(h_dim => out_dim)
    )
end

function get_env_details(name)
    env = gym_wrappers.FlattenObservation(gym.make(name))
    state_dim = convert(Int, env.observation_space.shape[1])
    action_dim = convert(Int, env.action_space.n)
    env.close()
    return state_dim, action_dim
end

function act!(env, action)
    state, reward, term, trunc, info = env.step(action-1)
    terminal = term || trunc
    reward = term ? -1 : reward
    return state, reward, terminal
end

function random_action(env)
    return convert(Int, env.action_space.sample())+1
end

function trainQ()
    global Q, Q̂, opt, buffer
    Q |> device; Q̂ |> device; loss = [0.0]

    s, a, r, s′, t = get_batch(buffer, batch_size) |> device
    grads = Flux.gradient(Q) do Q
        q = sum(Q(s).*a, dims=1)
        q_t = r .+ γ.*maximum(Q̂(s′), dims=1).*t
        loss = Flux.mse(q, q_t; agg=identity)
        mean(loss)
    end
    Flux.update!(opt, Q, grads[1])
    Q |> cpu; Q̂ |> cpu
end

function pre_loop(env, n)
    global buffer
    step = 1
    while step <= n
        s, info = env.reset()
        t = false
        while !t
            a = random_action(env)
            s′, r, t = act!(env, a)
            store!(buffer, s, a, r, s′, t)
            s = s′
            step += 1
        end
    end
end

function train_loop(env)
    global Q, Q̂, opt, buffer
    train_reward_per_episode = Float64[]
    test_reward_per_episdoe = Float64[]
    best_test = 0
    best_Q = deepcopy(Q)
    
    prog = ProgressThresh(0, "Training:"); step = 1
    while step <= n_steps
        ProgressMeter.update!(prog, n_steps-step)
        s, info = env.reset()
        t = false
        total_r = 0.0
        
        while !t
            Flux.testmode!(Q); Flux.testmode!(Q̂);
            q = Q(s)
            a = ϵ(step) >= rand() ? random_action(env) : argmax(q)
            s′, r, t = act!(env, a)
            total_r += r
            store!(buffer, s, a, r, s′, t)
            s = s′

            Flux.trainmode!(Q); Flux.trainmode!(Q̂);
            trainQ()
            step += 1
        end
        push!(train_reward_per_episode, total_r)
        push!(test_reward_per_episdoe, mean([eval_loop(env) for _ in 1:10]))
        if test_reward_per_episdoe[end] >= best_test
            Flux.testmode!(Q)
            best_test = test_reward_per_episdoe[end]
            best_Q = deepcopy(Q)
        end
        
        if step % div(n_steps, 25) == 0
            Q̂ = deepcopy(Q)
        end
    end
    return best_Q, train_reward_per_episode, test_reward_per_episdoe
end

function eval_loop(env)
    s, info = env.reset()
    t = false
    total_r = 0.0
    Flux.testmode!(Q)

    while !t
        s′, r, t = act!(env, argmax(Q(s)))
        total_r += r
        s = s′
    end
    return total_r
end

#=== Hyper Params ===#

env_name = "CartPole-v1"

n_steps = 10000
steps = 1:n_steps

η = Triangle(λ0=0.0001, λ1=0.01, period=div(n_steps, 1))        # Learning rate schedule
ϵ = Step(λ=1.0, γ=0.75, step_sizes=div(n_steps, 20))            # Exploration probability schedule
γ = 0.99                                                        # Future reward discount factor

batch_size = 32                                                 # Friends don't let friends use batch size > 32 - Yann LeCun
replay_buffer_size = n_steps                                    # How many samples to keep from played games

envd = get_env_details(env_name)
state_dim = envd[1]                                             # The input size of the environment
action_dim = envd[2]                                            # The number of possible action in the environment

#=== Initializations ===#

Q = create_model(state_dim, action_dim, 64)
Q̂ = deepcopy(Q)
opt = Flux.setup(AdamW(), Q)
buffer = Buffer(state_dim, action_dim, replay_buffer_size)

savefig(plot(steps, η.(steps)), "eta.png")
savefig(plot(steps, ϵ.(steps)), "explore.png")

#=== Main ===#

env = gym_wrappers.FlattenObservation(gym.make(env_name))
pre_loop(env, replay_buffer_size)
Q, train_r, test_r = train_loop(env)
savefig(plot(train_r), "train_reward.png")
savefig(plot(test_r), "test_reward.png")
env.close()

env = gym_wrappers.FlattenObservation(gym.make(env_name, render_mode="human"))
final_r = mean([eval_loop(env) for _ in 1:3])
println(final_r)
env.close()

@save "$(env_name)_Q.bson" Q

