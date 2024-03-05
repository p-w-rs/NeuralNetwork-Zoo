module Ribs

using PyCall

ribs = pyimport("ribs")
np = pyimport("numpy")

GridArchive = ribs.archives.GridArchive
EvolutionStrategyEmitter = ribs.emitters.EvolutionStrategyEmitter
Scheduler = ribs.schedulers.Scheduler

export Optimizer, GridArchive, EvolutionStrategyEmitter, Scheduler, opt_step!

struct Optimizer
	scheduler
	qfunc::Function
	dfunc::Function
end

function opt_step!(opt::Optimizer)
	solutions = opt.scheduler.ask()
	objective_batch = [
		opt.qfunc(Float32.(solutions[i, :])) for i in 1:size(solutions, 1)
	]
	measures_batch = vcat([
		opt.dfunc(Float32.(solutions[i, :])) for i in 1:size(solutions, 1)
	]...)
	opt.scheduler.tell(objective_batch, measures_batch)
	return objective_batch, measures_batch
end

end

#=
function qfunc(params::Vector{Float32})::Float32
	env = Env("LunarLander-v2"; continuous=true, enable_wind=true)
	policy = construct(vcat(params, [1f0, 1f0]))
	mn, std = evaluate(env, policy, 20)
	close!(env)
	return mn
end

function evaluate(env::Env, policy::M, n::Int)::Tuple{Float32, Float32}
	rewards = zeros(Float32, n)
	for i in 1:n
		ps = zeros(Float32, s_dim)
		cs = reset!(env)
		d = false
		reward = 0f0
		Flux.reset!(policy)
		while !d
			a = policy(cs)
			s, r, d = step!(env, a)
			ps, cs = d ? (zeros(Float32, s_dim), reset!(env)) : (cs, s)
			reward += r
		end
		rewards[i] = reward
	end
	return mean(rewards), std(rewards)
end

function dfunc(params::Vector{Float32})
	[norm(params, 1) norm(params, 2)]
end

archive = GridArchive(
    solution_dim=construct.length-2,
    dims=[20, 20],
    ranges=[(min_bound, max_bound), (min_bound, max_bound)],
    learning_rate=0.01,
    threshold_min=-1000,
    qd_score_offset=-1000
)
result_archive = GridArchive(
    solution_dim=construct.length-2,
    dims=[20, 20],
    ranges=[(min_bound, max_bound), (min_bound, max_bound)]
)
emitters = [
    EvolutionStrategyEmitter(
        archive=archive,
        x0=rand(Float32, construct.length-2),
        sigma0=0.5,
        ranker="imp",
        selection_rule="mu",
        restart_rule="basic",
        batch_size=36
    ) for _ in 1:15
]
scheduler = Scheduler(archive, emitters, result_archive=result_archive)

opt = Optimizer(scheduler, qfunc, dfunc)
for _ in 1:50
	objective_batch, measures_batch = opt_step!(opt)
	cur_best = maximum(objective_batch)
	println(cur_best+1000)
end
=#