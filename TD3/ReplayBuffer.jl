module ReplayBuffer

using StatsBase: sample, weights

export Buffer, store!, batch!, update_wv!, reset_wv!

mutable struct Buffer
   s::Matrix{Float32}
   a::Matrix{Float32}
   r::Matrix{Float32}
   s′::Matrix{Float32}
   t::Matrix{UInt32}
   wv::Vector{Float64}
   lastIdxs::Vector{Int64}
   idx::Int64
   len::Int64
   maxl::Int64
end

function Buffer(state_dim, action_dim, maxl)
    return Buffer(
        zeros(Float32, state_dim, maxl),
        zeros(Float32, action_dim, maxl),
        zeros(Float32, 1, maxl),
        zeros(Float32, state_dim, maxl),
        zeros(UInt32, 1, maxl),
        ones(Float32, maxl),
        [], 1, 0, maxl
    )
end

function store!(buffer, s, a, r, s′, t; w=1.0)
    buffer.s[:, buffer.idx] .= s
    buffer.a[:, buffer.idx] .= a
    buffer.r[:, buffer.idx] .= r
    buffer.s′[:, buffer.idx] .= s′
    buffer.t[:, buffer.idx] .= t
    buffer.wv[buffer.idx] = w
    buffer.idx = buffer.idx+1 > buffer.maxl ? 1 : buffer.idx+1
    buffer.len = min(buffer.len+1, buffer.maxl)
end

function batch!(buffer, n)
    idxs = 1:buffer.len
    idxs = sample(idxs, weights(1 ./ buffer.wv[idxs]), n; replace=false, ordered=false)
    buffer.lastIdxs = idxs
    return buffer.s[:, idxs], buffer.a[:, idxs], buffer.r[:, idxs], buffer.s′[:, idxs], buffer.t[:, idxs]
end

function update_wv!(buffer, wv)
    buffer.wv[buffer.lastIdxs] .= wv
end

function reset_wv!(buffer; w=1.0)
    buffer.wv[:] .= w
end

end
