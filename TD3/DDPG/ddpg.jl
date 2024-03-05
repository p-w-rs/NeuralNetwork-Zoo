#=== Package Imports and Globals ===#
using PyCall, Flux, CUDA, ParameterSchedulers, Random, ProgressMeter, Plots
using ParameterSchedulers: Stateful, next!
using StatsBase: mean
using BSON: @save

include("utils.jl")
using .Utils

include("replayBuffer.jl")
using .ReplayBuffer

include("ouNoise.jl")
using .OUNoise

gym = pyimport("gymnasium")
gym_wrappers = pyimport("gymnasium.wrappers")
device = CUDA.functional() ? gpu : cpu

#== Setup Section ==#
env_name = "LunarLander-v2"
state_dim, act_dim, scale = get_env_details(env_name)
h_dim = state_dim*act_dim*2
n_steps = 100000
γ = 0.99

batch_size = 32
buffer_size = 100000
buffer = Buffer(state_dim, act_dim, buffer_size)
noiser = Noiser(act_dim, scale)

Π = Chain(
    Dense(state_dim => h_dim, selu; init=lecun_normal), AlphaDropout(0.5),
    Dense(h_dim => h_dim, selu; init=lecun_normal), AlphaDropout(0.5),
    Dense(h_dim => h_dim, selu; init=lecun_normal), AlphaDropout(0.5),
    Dense(h_dim => act_dim, hardtanh; init=lecun_normal), Flux.Scale(scale, false, identity)
)
Πₜ = deepcopy(Π)
Πη = Triangle(λ0=0.0001, λ1=0.01, period=div(n_steps, 2))
optΠ = Flux.setup( RMSProp(Πη(1)), Π )

Q = Chain(
    Parallel(vcat,
        s=Chain(Dense(state_dim => h_dim, selu; init=lecun_normal), AlphaDropout(0.5)),
        a=Chain(Dense(act_dim => h_dim, selu; init=lecun_normal), AlphaDropout(0.5)),
    ),
    Dense(h_dim*2, h_dim, selu), AlphaDropout(0.5),
    Dense(h_dim, h_dim, selu), AlphaDropout(0.5),
    Dense(h_dim, 1)
)
Qₜ = deepcopy(Q)
Qη = Triangle(λ0=0.0003, λ1=0.03, period=div(n_steps, 2))
optQ = Flux.setup( RMSProp(Qη(1)), Q )

#== Training Functions ==#
function pre_train(env)
    global Π, Πₜ, Πη, optΠ, Q, Qₜ, Qη, optQ, buffer, noiser
    step = 1
    while step <= buffer_size
        s, _ = env.reset()
        t = false
        while !t
            a = a_rand(env)
            s′, r, t = act!(env, a)
            q = Q((s, a))
            q_t = r .+ γ.*Qₜ((s′, a_rand(env))).*t
            store!(buffer, s, a, r, s′, t; w=Flux.mse(q, q_t))
            s = s′
            step += 1
        end
    end
end

function train_loop(env)
    global Π, Πₜ, Πη, optΠ, Q, Qₜ, Qη, optQ, buffer, noiser
    train_reward_per_episode = Float64[]
    test_reward_per_episdoe = Float64[]
    best_test = 0
    best_ΠQ = (deepcopy(Π), deepcopy(Q))
    
    prog = ProgressThresh(0, "Training:"); step = 1
    while step <= n_steps
        ProgressMeter.update!(prog, n_steps-step)
        s, info = env.reset()
        t = false
        total_r = 0.0
        
        env_t = 0
        reset!(noiser)
        while !t
            Flux.testmode!(Π); Flux.testmode!(Q);
            a = noisyAction!(noiser, Π(s)[:], env_t)
            s′, r, t = act!(env, a)
            total_r += r
            #q = Q((s, a))
            #q_t = r .+ γ.*Qₜ((s′, Πₜ(s')[:])).*t
            store!(buffer, s, a, r, s′, t)
            s = s′

            trainModel(step)
            env_t += 1
            step += 1
        end
        push!(train_reward_per_episode, total_r)
        push!(test_reward_per_episdoe, mean([eval_loop(env) for _ in 1:10]))
        if test_reward_per_episdoe[end] >= best_test
            Flux.testmode!(Π); Flux.testmode!(Q);
            best_test = test_reward_per_episdoe[end]
            best_ΠQ = (deepcopy(Π), deepcopy(Q))
        end

        if step % 20 == 0
            Πₜ, Qₜ = deepcopy(Π), deepcopy(Q)
            #reset_wv!(buffer; w=1.0)
        end
    end
    return best_ΠQ[1], best_ΠQ[2], train_reward_per_episode, test_reward_per_episdoe
end

function trainModel(step)
    global Π, Πₜ, Πη, optΠ, Q, Qₜ, Qη, optQ, buffer, noiser
    Flux.trainmode!(Π); Flux.trainmode!(Q);
    Π |> device; Q |> device; loss = [0.0]

    s, a, r, s′, t = get_batch!(buffer, batch_size) |> device
    grads = Flux.gradient(Q) do Q
        q = Q((s, a))
        q_t = r .+ γ.*Qₜ((s′, Πₜ(s′))).*t
        loss = Flux.mse(q, q_t; agg=identity)
        mean(loss)
    end
    #update_wv!(buffer, loss[:])
    Flux.update!(optQ, Q, grads[1])
    Flux.adjust!(optQ, Qη(step))

    grads = Flux.gradient(Π) do Π
        -mean(Q((s, Π(s))))
    end
    Flux.update!(optΠ, Π, grads[1])
    Flux.adjust!(optΠ, Πη(step))
    Π |> cpu; Q |> cpu
end

function eval_loop(env)
    global Π, Πₜ, Πη, optΠ, Q, Qₜ, Qη, optQ, buffer, noiser
    s, _ = env.reset()
    t = false
    total_r = 0.0
    Flux.testmode!(Π); Flux.testmode!(Q);

    while !t
        s′, r, t = act!(env,  Π(s)[:])
        total_r += r
        s = s′
    end
    return total_r
end

#== Main Section ==#
env = gym_wrappers.FlattenObservation(gym.make(env_name, continuous=true))
pre_train(env)
Π, Q, train_r, test_r = train_loop(env)
savefig(plot(train_r), "train_reward.png")
savefig(plot(test_r), "test_reward.png")
env.close()

env = gym_wrappers.FlattenObservation(gym.make(env_name, continuous=true, render_mode="human"))
final_r = mean([eval_loop(env) for _ in 1:10])
println(final_r)
env.close()

@save "$(env_name)_ΠQ.bson" Π Q

