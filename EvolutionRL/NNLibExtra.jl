module NNLibExtra

using Flux, ChainRulesCore, Random
using Flux: @functor

export DenseStochastic, DenseHeb, lecun_normal



# Dense layer with mean and std for weights
struct DenseStochastic{F, M<:AbstractMatrix}
    mean_w::M
    std_w::M
    σ::F

    function DenseStochastic(mean_w::M, std_w::M, σ::F = identity) where {M<:AbstractMatrix, F}
        new{F, M}(mean_w, std_w, σ)
    end
end

function DenseStochastic((in, out)::Pair{<:Integer, <:Integer}, σ=identity;
               init_mean=Flux.glorot_uniform, init_std=Flux.ones32)
    mean_w = init_mean(out, in)
    std_w = init_std(out, in)
    DenseStochastic(mean_w, std_w, σ)
end

@functor DenseStochastic

function (a::DenseStochastic)(x::AbstractArray)
    w = (a.mean_w .+ randn(Float32, size(a.mean_w))) .* a.std_w
    activation = a.σ.(w * x)
    return activation
end

function Base.show(io::IO, l::DenseStochastic)
    print(io, "DenseStochastic(", size(l.mean_w, 2), " => ", size(l.mean_w, 1))
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end



# Dense layer with hebbian learning rules
struct DenseHeb{F, M<:AbstractMatrix, B}
  weights::M
  bias::B
  σ::F

  η::M
  A::M
  B::M
  C::M
  D::M
end

function DenseHeb((in, out)::Pair{<:Integer, <:Integer}, σ=identity;
               init_w=Flux.glorot_normal, init_b=Flux.ones32)
  DenseHeb(
    init_w(out, in), init_b(out), σ,
    init_w(out, in), init_w(out, in), init_w(out, in), init_w(out, in), init_w(out, in)
  )
end

@functor DenseHeb

function (m::DenseHeb)(x)
  y = m.σ.(m.weights * x .+ m.bias)
  z = y*x'
  Δw = m.η .* ((m.A .* z) .+ (m.B .* x') .+ (m.C .* y) .+ m.D)
  m.weights .+= Δw #clamp.(m.weights .+ Δw, -1.0, 1.0)
  return y
end

function Base.show(io::IO, l::DenseHeb)
    print(io, "DenseHeb(", size(l.weights, 2), " => ", size(l.weights, 1))
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end



# weight initializers
function lecun_normal(rng::AbstractRNG, dims::Integer...; gain::Real=1)
  std = Float32(gain)*sqrt(1.0f0 / first(Flux.nfan(dims...))) # calculates the standard deviation based on the `fan_in` value
  return Flux.truncated_normal(rng, dims...; mean=0, std=std)
end

lecun_normal(dims::Integer...; kwargs...) = lecun_normal(Random.default_rng(), dims...; kwargs...)
lecun_normal(rng::AbstractRNG=default_rng(); init_kwargs...) = (dims...; kwargs...) -> lecun_normal(rng, dims...; init_kwargs..., kwargs...)

ChainRulesCore.@non_differentiable lecun_normal(::Any...)

end #module
