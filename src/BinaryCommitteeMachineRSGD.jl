module BinaryCommitteeMachineRSGD

using ExtractMacro

typealias IVec Vector{Int}
typealias BVec BitVector
typealias BVec2 Vector{BVec}
typealias Vec Vector{Float64}
typealias Vec2 Vector{Vec}

function dot_prod(a::BVec, b::BVec, l::Int64)
    ac = a.chunks
    bc = b.chunks
    @inbounds @simd for i = 1:length(ac)
	l -= 2 * count_ones(ac[i] $ bc[i])
    end
    return l
end

function add_cx_to_y!(c::Float64, x::Vec, y::Vec)
    @inbounds @simd for i = 1:length(y)
	y[i] += c * x[i]
    end
    return y
end

type Patterns
    N::Int
    M::Int
    p_tr::BVec2
    o_tr::IVec
    function Patterns(N::Integer, M::Integer)
	p_tr = [bitrand(N) for μ = 1:M]
	o_tr = rand(-1:2:1, M)
	return new(N, M, p_tr, o_tr)
    end
end

type Params
    N::Int64
    K::Int64
    y::Int64
    τ::Int64
    η::Float64
    λ::Float64
    γ::Float64
end

type Grads
    N::Int64
    K::Int64
    ΔH::Vec2
    function Grads(N::Int64, K::Int64)
	ΔH = [zeros(Float64, N) for k = 1:K]
	return new(N, K, ΔH)
    end
end

type Weights
    N::Int64
    K::Int64
    J::BVec2
    H::Vec2
    function Weights(N::Int64, K::Int64)
	H = [1.0 * rand(-1:2.:1, N) for k = 1:K]
	J = [H[k] .> 0 for k = 1:K]
	return new(N, K, J, H)
    end
end

type Net
    N::Int64
    K::Int64
    W::Weights
    Δ::Grads
    function Net()
	return new(0, 0, Weights(0,0), Grads(0,0))
    end
end

function reset_grads!(net::Net, params::Params)
    @extract params : N K
    @extract net    : Δ
    @extract Δ      : ΔH
    @inbounds for k = 1:K
	fill!(ΔH[k], 0.0)
    end
end

function reset_net_mean!(netc::Net, nets::Vector{Net}, params::Params)
    @extract params : N K y
    W = Weights(N, K)
    Δ = Grads(N, K)

    for k = 1:K
        W.H[k] = zeros(Float64, N)
	for r = 1:y
	    add_cx_to_y!(1/y, 2. * nets[r].W.J[k] - 1, W.H[k])
	end
	W.J[k] = W.H[k] .> 0
    end

    netc.N = N
    netc.K = K
    netc.W = W
    netc.Δ = Δ
    return netc
end

function reset_net!(net::Net, params::Params)
    @extract params : N K
    W = Weights(N, K)
    Δ = Grads(N, K)

    net.N = N
    net.K = K
    net.W = W
    net.Δ = Δ

    return net
end

function update_net!(net::Net)
    @extract net : N K W Δ
    @extract W   : H J
    @extract Δ   : ΔH

    @inbounds for k = 1:K
	Hk = H[k]
	Jk = J[k]
	add_cx_to_y!(1., ΔH[k], H[k])
	for i = 1:N
	    Jk[i] = Hk[i] > 0
	end
    end
end

function forward_net!(netr, p::BVec, h::IVec, t::IVec)
    @extract netr : N K W
    @extract W    : J
    @inbounds for k = 1:K
	h[k] = dot_prod(p, J[k], N)
	t[k] = 2 * (h[k] > 0) - 1
    end
    hout = sum(t)
    tout = 2 * (hout > 0) - 1

    return hout, tout
end

function forward_net(netr, p::BVec)
    h = Array(Int, netr.K)
    t = Array(Int, netr.K)
    hout, tout = forward_net!(netr, p, h, t)
    return h, t, hout, tout
end

forward_net(netr::Net, ps::BVec2) = [forward_net(netr, p) for p in ps]

let wrongh = Int[], indh = Int[], sortedindh = Int[]
    global compute_gd!
    function compute_gd!(net::Net, patterns::Patterns, μ::Int64, h::IVec, t::IVec, hout::Int64, params::Params)
	@extract net      : N K W Δ
	@extract W        : H
	@extract Δ        : ΔH
	@extract patterns : p_tr o_tr
	@extract params   : η

	p = p_tr[μ]
	o = o_tr[μ]

	n_h = (- o * hout + 1) ÷ 2
	for k = 1:K
	    h[k] * o > 0 && continue
	    push!(wrongh, -o * h[k])
	    push!(indh, k)
	end
	resize!(sortedindh, length(wrongh))
	sortperm!(sortedindh, wrongh)

	for kk = 1:n_h
	    k = indh[sortedindh[kk]]
	    ΔHk = ΔH[k]
	    ΔHtemp = o * (2. * p - 1.)
	    add_cx_to_y!(η, ΔHtemp, ΔHk)
	end
	empty!(wrongh)
	empty!(indh)
	empty!(sortedindh)
    end
end

function kickboth!(net::Net, netc::Net, params::Params, δH::Vec)
    @extract params : λ
    @extract net    : N K W
    @extract netc   : Wc=W
    @extract W      : H J
    @extract Wc     : Hc=H Jc=J

    @inbounds for k = 1:K
	Jck = Jc[k]
	Jk = J[k]
	for i = 1:N
	    δH[i] = Jck[i] - Jk[i]
	end
	Hk = H[k]
	Hck = Hc[k]
	add_cx_to_y!(λ, δH, Hk)
	add_cx_to_y!(-λ, δH, Hck)
	for i = 1:N
	    Jk[i] = Hk[i] > 0
	    Jck[i] = Hck[i] > 0
	end
    end
end

function kickboth_traced!(net::Net, netc::Net, params::Params, δH::Vec, old_J::BVec2, cavity::Bool = false)
    @extract params : N K y γ λ
    @extract net    : N K W
    @extract netc   : Wc=W
    @extract W      : H J
    @extract Wc     : Hc=H Jc=J

    corr = cavity ? tanh(γ * y) : 1.0
    @inbounds for k = 1:K
	Jck = Jc[k]
	Jk = J[k]
	Hk = H[k]
	Hck = Hc[k]
	if γ ≥ 5
	    for i = 1:N
		δH[i] = sign(Hck[i]) - (2 * Jk[i] - 1)
	    end
	else
	    for i = 1:N
		δH[i] = (tanh(γ * y * Hck[i]) - corr * (2 * Jk[i] - 1))
	    end
	end
	add_cx_to_y!(λ, δH, Hk)
	old_Jk = old_J[k]
	for i = 1:N
	    old_Jki = old_Jk[i]
	    new_Jki = Hk[i] > 0
	    Jk[i] = new_Jki
	    Hck[i] += 2 * (new_Jki - old_Jki) / y
	    Jck[i] = Hck[i] > 0
	end
    end
end

function kickboth_traced_cont!(net::Net, netc::Net, params::Params, δH::Vec, old_J::BVec2)
    @extract params : N K y γ λ
    @extract net    : N K W
    @extract netc   : Wc=W
    @extract W      : H J
    @extract Wc     : Hc=H Jc=J


    @inbounds for k = 1:K
	Jck = Jc[k]
	Jk = J[k]
	Hk = H[k]
	Hck = Hc[k]
	for i = 1:N
	    Wi = 2 * Jk[i] - 1
	    δH[i] = Hck[i] - Wi
	end
	add_cx_to_y!(λ, δH, Hk)
	old_Jk = old_J[k]
	for i = 1:N
	    old_Jki = old_Jk[i]
	    new_Jki = Hk[i] > 0
	    Jk[i] = new_Jki
	    Hck[i] += 2 * (new_Jki - old_Jki) / y
	    Jck[i] = Hck[i] > 0
	end
    end
end

function compute_err(net::Net, ps::BVec2, os::IVec)
    @extract net : K

    h = Array(Int, K)
    t = Array(Int, K)
    errs = 0
    for (p, o) in zip(ps, os)
	_, tout = forward_net!(net, p, h, t)
	errs += tout ≠ o
    end
    return errs
end

function compute_err(net::Net, p::BVec, o::Int64)
    _, _, _, tout = forward_net(net, p)
    return tout ≠ o
end

compute_err(net::Net, patterns::Patterns) = compute_err(net, patterns.p_tr, patterns.o_tr)

function epoch!(net::Net, patterns::Patterns, patt_perm::IVec, a0::Integer, params::Params)
    @extract params   : τ
    @extract patterns : M p_tr o_tr

    reset_grads!(net, params)
    for a = a0:(a0+τ-1)
	μ = patt_perm[mod1(a,M)]
	p = p_tr[μ]
	o = o_tr[μ]
	h, t, hout, tout = forward_net(net, p)
	tout == o && continue
	compute_gd!(net, patterns, μ, h, t, hout, params)
    end

    update_net!(net)

    return
end

function compute_dist(net1::Net, net2::Net)
    @extract net1 : N K W1=W
    @extract net2 : W2=W
    @extract W1   : J1=J
    @extract W2   : J2=J

    return sum([sum(j1 $ j2) for (j1,j2) in zip(J1,J2)])
end

function init_outfile(outfile::AbstractString, y::Int)
    !isempty(outfile) && isreadable(outfile) && error("outfile exists: $outfile")
    !isempty(outfile) && open(outfile, "w") do outf
	println(outf, "#epoch err(Wc) err(best) | ", join(["err(W$i)" for i = 1:y], " "), " | λ γ | ", join(["d(W$i)" for i = 1:y], " "))
    end
end

function report(ep::Int, errc, minerrc, errs::Vector, minerrs::Vector, dist::Vector{Int}, η::Float64, λ::Float64, γ::Float64, quiet::Bool, outfile::AbstractString)
    if !quiet
	println("ep: $ep λ: $λ γ: $γ η: $η")
	println("  errc: $minerrc [$errc]")
	println("  errs: $(minimum(minerrs)) $errs (mean=$(mean(errs)))")
	println("  dist = $dist (mean=$(mean(dist)))")
    end

    if !isempty(outfile)
        open(outfile, "a") do outf
            @printf(outf, "%i %i %i |", ep, errc, min(minerrc, minimum(minerrs)))
            for ek in errs
                @printf(outf, " %i", ek)
            end
            @printf(outf, " | %f %f |", λ, γ)
	    for dk in dist
		@printf(outf, " %f", dk)
	    end
            @printf(outf, "\n")
        end
    end
end

function main(; N::Integer = 51,
                K::Integer = 1,
                max_epochs::Real = 1_000,
                M::Integer = 10,
                center::Bool = false,
                init_equal::Bool = true,
                seed::Integer = 1,
		seed_run::Integer = 0,
                outfile::AbstractString = "",
                waitcenter::Bool = false,
                τ::Integer = 5,
                f_λ::Float64 = 1.0,
                f_γ::Float64 = 1.0,
		γpolicy::Symbol = :additive,
                ep_scope::Integer = 1,
                η::Float64 = 2.0,
		f_η::Float64 = 1.0,
                λ::Float64 = 0.1,
                γ::Float64 = Inf,
                y::Integer = 1,
		formula::Symbol = :tanh,
		quiet::Bool = false)

    srand(seed)

    formula ∈ [:tanh, :cavity, :simpler] || thorw(ArgumentError("formula must be either :tanh, :cavity or :simpler, given : $formula"))
    γpolicy ∈ [:multiplicative, :additive] || throw(ArgumentError("γpolicy must be either :multiplicative or :additive, given: $γpolicy"))

    λ == 0 && waitcenter && warn("λ=$λ waitcenter=$waitcenter")

    patterns = Patterns(N, M)

    params = Params(N, K, y, τ, η, λ, γ)

    seed_run ≠ 0 && srand(seed_run)

    netc = Net()
    nets = [Net() for r = 1:y]

    if center || init_equal
	reset_net!(netc, params)
    end

    for r = 1:y
	if init_equal
	    nets[r] = deepcopy(netc)
	else
	    reset_net!(nets[r], params)
	end
    end

    !center && reset_net_mean!(netc, nets, params)

    old_J = deepcopy(netc.W.J)

    errc = compute_err(netc, patterns.p_tr, patterns.o_tr)
    minerrc = errc

    errs = [compute_err(net, patterns.p_tr, patterns.o_tr) for net in nets]
    minerrs = copy(errs)

    dist = [compute_dist(netc, net) for net in nets]

    init_outfile(outfile, y)
    report(0, errc, minerrc, errs, minerrs, dist, η, λ, γ, quiet, outfile)

    ep_increase = τ / M
    ep = 0.0
    δH = Array(Float64, N)
    patt_perm = [randperm(M) for r = 1:y]
    a0 = ones(Int, y)
    ok = errc == 0 || (!waitcenter && any(x->x==0, errs))

    minerr = min(minerrc, minimum(minerrs))
    while !ok && (ep < max_epochs)
	for r in randperm(y)
	    net = nets[r]
	    for k = 1:K
		copy!(old_J[k], net.W.J[k])
	    end
	    epoch!(net, patterns, patt_perm[r], a0[r], params)
	    a0[r] += τ
	    if !center
		if formula == :tanh || formula == :cavity
		    kickboth_traced!(net, netc, params, δH, old_J, formula == :cavity)
		elseif formula == :simpler
		    kickboth_traced_cont!(net, netc, params, δH, old_J)
		end
	    elseif λ > 0
		kickboth!(net, netc, params, δH)
	    end
	end
	ep += ep_increase
	if abs(ep - ep_scope * round(ep / ep_scope)) < ep_increase / 2
	    if γpolicy == :multiplicative
		γ *= f_γ
	    elseif γpolicy == :additive
		γ += f_γ
	    end
	    λ *= f_λ
	    η *= f_η
	    params.η = η
	    params.λ = λ
	    params.γ = γ
	end
	if abs(ep - round(ep)) < ep_increase / 2
	    errc = compute_err(netc, patterns)
	    minerrc = min(minerrc, errc)
	    errc == 0 && (ok = true)
	    for r = 1:y
		net = nets[r]
		shuffle!(patt_perm[r])
		a0[r] = 1
		errs[r] = compute_err(net, patterns)
		minerrs[r] = min(minerrs[r], errs[r])
		errs[r] == 0 && !waitcenter && (ok = true)
		dist[r] = compute_dist(netc, net)
	    end
	    minerr = min(minerrc, minimum(minerrs))
	    report(round(Int,ep), errc, minerrc, errs, minerrs, dist, η, λ, γ, quiet, outfile)
	end
    end

    !quiet && println(ok ? "SOLVED" : "FAILED")

    return ok, round(Int,ep), minerr
end

end # module
