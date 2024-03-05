function trainModel()
    global Π, V, Πₜ, Vₜ, optΠ, optV, buffer
    Π |> device; V |> device; loss = [0.0]

    s, a, r, s′, t = get_batch(buffer, batch_size) |> device
    grads = Flux.gradient(V) do V
        q = V((s, a))
        q_t = r .+ γ.*Vₜ((s′, Πₜ(s′))).*t
        loss = Flux.mse(q, q_t; agg=identity)
        mean(loss)
    end
    update_wv!(buffer, loss[:])
    Flux.update!(optV, V, grads[1])

    grads = Flux.gradient(Π) do Π
        -mean(V((s, Π(s))))
    end
    Flux.update!(optΠ, Π, grads[1])
    Π |> cpu; V |> cpu
end

function train_loop(env)
    global Π, V, Πₜ, Vₜ, optΠ, optV, buffer, noiser
    train_reward_per_episode = Float64[]
    test_reward_per_episdoe = Float64[]
    best_test = 0
    best_ΠV = (deepcopy(Π), deepcopy(V))
    
    prog = ProgressThresh(0, "Training:"); step = 1
    while step <= n_steps
        ProgressMeter.update!(prog, n_steps-step)
        s, info = env.reset()
        t = false
        total_r = 0.0
        
        env_t = 0
        reset!(noiser)
        while !t
            Flux.testmode!(Π); Flux.testmode!(V);
            a = ϵ(step) >= rand() ? random_action(env) : Π(s)[:]
            a = noisyAction!(noiser, a, env_t)
            s′, r, t = act!(env, a)
            total_r += r
            q = V((s, a))
            q_t = r .+ γ.*Vₜ((s′, random_action(env))).*t
            store!(buffer, s, a, r, s′, t; w=Flux.mse(q, q_t))
            s = s′

            Flux.trainmode!(Π); Flux.trainmode!(V);
            trainModel()
            env_t += 1
            step += 1
        end
        push!(train_reward_per_episode, total_r)
        push!(test_reward_per_episdoe, mean([eval_loop(env) for _ in 1:10]))
        if test_reward_per_episdoe[end] >= best_test
            Flux.testmode!(Π); Flux.testmode!(V);
            best_test = test_reward_per_episdoe[end]
            best_ΠV = (deepcopy(Π), deepcopy(V))
        end

        if step % 2 == 0
            Πₜ, Vₜ = deepcopy(Π), deepcopy(Vₜ)
            reset_wv!(buffer; w=2.0)
        end
    end
    return best_ΠV[1], best_ΠV[2], train_reward_per_episode, test_reward_per_episdoe
end

function eval_loop(env)
    s, info = env.reset()
    t = false
    total_r = 0.0
    Flux.testmode!(Π); Flux.testmode!(V);

    while !t
        s′, r, t = act!(env,  Π(s)[:])
        total_r += r
        s = s′
    end
    return total_r
end

#=== Hyper Params ===#

env_name = "LunarLander-v2"

n_steps = 200000
steps = 1:n_steps

η = Triangle(λ0=0.0001, λ1=0.01, period=div(n_steps, 1))        # Learning rate schedule
ϵ = Step(λ=1.0, γ=0.75, step_sizes=div(n_steps, 6))            # Exploration probability schedule
γ = 0.99                                                        # Future reward discount factor

batch_size = 32                                                 # Friends don't let friends use batch size > 32 - Yann LeCun
replay_buffer_size = n_steps                                    # How many samples to keep from played games

envd = get_env_details(env_name)
state_dim = envd[1]                                             # The input size of the environment
action_dim = envd[2]                                            # The number of possible action in the environment
action_scl = envd[3]                                                   

#=== Initializations ===#

Π, V = create_model(state_dim, action_dim, action_scl, 128)
Πₜ, Vₜ = deepcopy(Π), deepcopy(V)
optΠ = Flux.setup(AdamW(0.0001), Π)
optV = Flux.setup(AdamW(0.001), V)
buffer = Buffer(state_dim, action_dim, replay_buffer_size)
noiser = Noiser(action_dim, action_scl)

savefig(plot(steps, η.(steps)), "eta.png")
savefig(plot(steps, ϵ.(steps)), "explore.png")

#=== Main ===#

env = gym_wrappers.FlattenObservation(gym.make(env_name))
pre_loop(env, replay_buffer_size)
Π, V, train_r, test_r = train_loop(env)
savefig(plot(train_r), "train_reward.png")
savefig(plot(test_r), "test_reward.png")
env.close()

env = gym_wrappers.FlattenObservation(gym.make(env_name, render_mode="human"))
final_r = mean([eval_loop(env) for _ in 1:3])
println(final_r)
env.close()

@save "$(env_name)_ΠV.bson" Π V
