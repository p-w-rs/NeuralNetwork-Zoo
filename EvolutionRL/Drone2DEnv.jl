module Drone2DEnv

using Distributions, Plots, Random

export World, Drone2D, Env, reset!, step!, render!

const WIDTH::Int64 = 10 # meters & pixels
const HEIGHT::Int64 = 10 # meters & pixels
const Δt::Float32 = 0.015f0 # integration timestep s
const SIMULATION_TIME_LIMIT::Float32 = 5.0 # seconds
const FRAME_RATE::Int = 60 # frames per second

mutable struct World
    gravity::Float32 # acceleration due to gravity m/s^2
    wind::Float32 # acceleration due to wind m/s^2
    turbulence::Float32 # acceleration due to turbulence m/s^2
end

mutable struct Drone2D
    x::Float32 # x position
    y::Float32 # y position
    θ::Float32 # angle in radians
    ẋ::Float32 # x velocity
    ẏ::Float32 # y velocity
    θ̇::Float32 # angular velocity
    x̄::Float32 # x acceleration
    ȳ::Float32 # y acceleration
    θ̄::Float32 # angular acceleration

    a::Float32 # thrust amplification in Newtons
    m::Float32 # mass in kg
    l::Float32 # distance between drone rotors and physical width of drone in m
end

function Drone2D(a=7f0, m=0.8f0, l=0.4f0)
    return Drone2D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0f0, 0f0, 0f0, m * (a*9.806f0), m, l)
end

mutable struct Env
    world::World
    drone::Drone2D
    target::Vector{Float32}
    gravity::Normal{Float32}
    wind::Normal{Float32}
    turbulence::Normal{Float32}
    obs_dim::Int64
    act_dim::Int64
    frame_count::Int
    total_time::Float32
end

function Env(g=-9.806f0, wind=true, turbulence=true)
    gravity_dist = Normal(g)
    wind_dist = wind ? Normal(0.0f0, 5.0f0) : Normal(0f0, 0f0)
    turbulence_dist = turbulence ? Normal(0.0f0, 1.0f0) : Normal(0f0, 0f0)

    world = World(rand(gravity_dist), rand(wind_dist), rand(turbulence_dist))
    target = [rand(Uniform(-WIDTH/2,WIDTH/2)), rand(Uniform(-HEIGHT/2,HEIGHT/2))]
    
    Env(world, Drone2D(), target, gravity_dist, wind_dist, turbulence_dist, 2, 2, 0, 0f0)
end

function reset!(env::Env, seed::Union{Int, Nothing}=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end

    env.drone = Drone2D()
    env.world = World(rand(env.gravity), rand(env.wind), rand(env.turbulence))
    env.target = [rand(Uniform(-WIDTH/2,WIDTH/2)), rand(Uniform(-HEIGHT/2,HEIGHT/2))]
    env.frame_count = 0
    env.total_time = 0f0
    return first(step!(env, [0f0, 0f0]))
end

function reward(drone::Drone2D, target::Vector{Float32})
    # Euclidean distance to the target
    distance = sqrt((drone.x - target[1])^2 + (drone.y - target[2])^2)
    return distance # Adding 1 to avoid division by zero
end

function done(env::Env)
    drone = env.drone
    off_screen = drone.x < -WIDTH/2 || drone.x > WIDTH/2 || drone.y < -HEIGHT/2 || drone.y > HEIGHT/2
    time_limit_reached = env.total_time >= SIMULATION_TIME_LIMIT
    return off_screen || time_limit_reached
end

function step!(env::Env, a::Vector{Float32})
    drone = env.drone
    world = env.world
    target = env.target

    # Calculate net thrust and torque
    a_mod = a .* drone.a
    net_thrust = a_mod[1] + a_mod[2]
    torque = (a_mod[2] - a_mod[1]) * drone.l

    # Calculate thrust components based on drone's angle
    horizontal_thrust = net_thrust * sin(drone.θ)
    vertical_thrust = net_thrust * cos(drone.θ)

    # Update accelerations
    drone.x̄ = (horizontal_thrust / drone.m)# + world.wind
    drone.ȳ = (vertical_thrust / drone.m) + world.gravity
    drone.θ̄ = (torque / drone.m)# + world.turbulence

    # Verlet integration
    drone.x += (drone.ẋ * Δt) + (0.5f0 * drone.x̄ * Δt^2)
    drone.y += (drone.ẏ * Δt) + (0.5f0 * drone.ȳ * Δt^2)
    drone.θ += (drone.θ̇ * Δt) + (0.5f0 * drone.θ̄ * Δt^2)

    # Update velocities using average acceleration
    drone.ẋ += drone.x̄ * Δt
    drone.ẏ += drone.ȳ * Δt
    drone.θ̇ += drone.θ̄ * Δt

    # Increment total simulation time
    env.total_time += Δt

    # Cartesian coordinates of the target relative to the drone
    relative_x = target[1] - drone.x
    relative_y = target[2] - drone.y

    # Convert to polar coordinates
    r = sqrt(relative_x^2 + relative_y^2) # radius
    θ_target = atan(relative_y, relative_x) # angle in radians

    # Update state with polar coordinates
    state = Float32[r, drone.θ]
    r = reward(drone, target)
    d = done(env)

    # Return the new state, reward, and done status
    return (state, r, d)
end

function render!(env::Env)
    # Only render at the specified frame rate
    if env.frame_count % Int(round(1/(Δt*FRAME_RATE))) != 0
        env.frame_count += 1
        return
    end
    env.frame_count += 1

    drone = env.drone
    plt = plot(xlim=(-WIDTH/2, WIDTH/2), ylim=(-HEIGHT/2, HEIGHT/2), aspect_ratio=:equal)
    
    # Draw the target
    scatter!([env.target[1]], [env.target[2]], label="Target", color=:black, markersize=drone.l*6)

    # The drone's position is (drone.x, drone.y) and its angle is drone.θ
    drone_pos = [drone.x, drone.y]
    # Increase the length of the bars by adjusting the multiplier for drone.l
    bar_length = drone.l # Adjust this value to make the bars longer
    drone_top = [drone.x + cos(drone.θ) * bar_length, drone.y + sin(drone.θ) * bar_length]
    drone_bottom = [drone.x - cos(drone.θ) * bar_length, drone.y - sin(drone.θ) * bar_length]

    # Draw the drone body
    scatter!([drone_pos[1]], [drone_pos[2]], label="Drone", color=:blue, markersize=drone.l*8)

    # Draw the horizontal bars above and below the drone
    bar_width = drone.l
    bar_height = drone.l/4 # Adjust this value for the thickness of the bars
    plot!([drone.x - bar_width/2, drone.x + bar_width/2], [drone.y + bar_height, drone.y + bar_height], line=(:solid, 1), color=:green, label="Top Bar")
    plot!([drone.x - bar_width/2, drone.x + bar_width/2], [drone.y - bar_height, drone.y - bar_height], line=(:solid, 1), color=:red, label="Bottom Bar")

    display(plt)
end

end #module

