module OUNoise
#=
Modified from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
=#
using Random

export Noiser, reset!, evolveState!, noisyAction!

mutable struct Noiser
    μ
    θ
    σ
    σ_max
    σ_min
    ϵ
    state
    action_dim
    low
    high
end

function Noiser(action_dim, scale; mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000)
    state = ones(action_dim) * mu
    Noiser(mu, theta, max_sigma, max_sigma, min_sigma, decay_period,
        state, action_dim, .-scale, scale
    )
end

function reset!(ou::Noiser)
    ou =  Noiser(ou.action_dim, ou.high)
    return ou
end

function evolveState!(ou::Noiser)
    x = ou.state
    dx = ou.θ .* (ou.μ .- x) .+ ou.σ .* randn(ou.action_dim)
    ou.state = x .+ dx
    return ou.state
end

function noisyAction!(ou::Noiser, action, t=0)
    ou_state = evolveState!(ou)
    ou.σ = ou.σ_max .- (ou.σ_max .- ou.σ_min) .* min(1.0, t ./ ou.ϵ)
    return clamp.(action, ou.low, ou.high)
end

end
