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
    ξ::BVec2
    σ::IVec
    function Patterns(N::Integer, M::Integer)
        ξ = [bitrand(N) for μ = 1:M]
        σ = rand(-1:2:1, M)
        return new(N, M, ξ, σ)
    end
end

type PatternsPermutation
    M::Int
    perm::IVec
    a::Int
    batch::Int
    PatternsPermutation(M::Integer, batch::Integer) = new(M, randperm(M), 1, batch)
end

function get_batch(pp::PatternsPermutation)
    @extract pp : M perm a batch

    b = min(a + batch - 1, M)
    if b == M
	shuffle!(perm)
	pp.a = 1
    else
	pp.a = b + 1
    end
    return a:b
end


type Params
    N::Int64
    K::Int64
    y::Int64
    η::Float64
    λ::Float64
    γ::Float64
end

type Grads
    N::Int64
    ΔH::Vec2
    function Grads(N::Int64, K::Int64)
        ΔH = [zeros(Float64, N) for k = 1:K]
        return new(N, ΔH)
    end
end

reset_grads!(Δ::Grads) = map!(x->fill!(x, 0.0), Δ.ΔH)

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

reset_grads!(net::Net) = reset_grads!(net.Δ)

function reset_net_mean!(netc::Net, nets::Vector{Net}, params::Params)
    @extract params : N K y
    W = Weights(N, K)
    Δ = Grads(N, K)

    for k = 1:K
	#W.H[k] = sum([(2.0 * net.W.J[k] - 1) / y for net in nets])
	W.H[k] = 2/y * sum([net.W.J[k] for net in nets]) .- 1
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

function forward_net!(netr, ξμ::BVec, h::IVec, τ::IVec)
    @extract netr : N K W
    @extract W    : J
    @inbounds for k = 1:K
        h[k] = dot_prod(ξμ, J[k], N)
        τ[k] = 2 * (h[k] > 0) - 1
    end
    hout = sum(τ)
    τout = 2 * (hout > 0) - 1

    return hout, τout
end

function forward_net(netr, ξμ::BVec)
    h = Array(Int, netr.K)
    τ = Array(Int, netr.K)
    hout, τout = forward_net!(netr, ξμ, h, τ)
    return h, τ, hout, τout
end

forward_net(netr::Net, ξ::BVec2) = [forward_net(netr, ξμ) for ξμ in ξ]

let wrongh = Int[], indh = Int[], sortedindh = Int[]
    global compute_gd!
    function compute_gd!(net::Net, patterns::Patterns, μ::Int64, h::IVec, τ::IVec, hout::Int64, params::Params)
        @extract net      : N K W Δ
        @extract W        : H
        @extract Δ        : ΔH
	@extract patterns : ξμ=ξ[μ] σμ=σ[μ]
        @extract params   : η

        n_h = (-σμ * hout + 1) ÷ 2
        for k = 1:K
            h[k] * σμ > 0 && continue
            push!(wrongh, -σμ * h[k])
            push!(indh, k)
        end
        resize!(sortedindh, length(wrongh))
        sortperm!(sortedindh, wrongh)

        for kk = 1:n_h
            k = indh[sortedindh[kk]]
            ΔHk = ΔH[k]
            ΔHtemp = σμ * (2.0 * ξμ - 1.0)
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

function kickboth_traced!(net::Net, netc::Net, params::Params, δH::Vec, old_J::BVec2, corrected::Bool = false)
    @extract params : N K y γ λ
    @extract net    : N K W
    @extract netc   : Wc=W
    @extract W      : H J
    @extract Wc     : Hc=H Jc=J

    correction = corrected ? tanh(γ * y) : 1.0
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
                δH[i] = (tanh(γ * y * Hck[i]) - correction * (2 * Jk[i] - 1))
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

function kickboth_traced_continuous!(net::Net, netc::Net, params::Params, δH::Vec, old_J::BVec2)
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

function compute_err(net::Net, ξ::BVec2, σ::IVec)
    @extract net : K

    h = Array(Int, K)
    τ = Array(Int, K)
    errs = 0
    for (ξμ, σμ) in zip(ξ, σ)
        _, τout = forward_net!(net, ξμ, h, τ)
        errs += τout ≠ σμ
    end
    return errs
end

function compute_err(net::Net, ξμ::BVec, σμ::Int64)
    _, _, _, τout = forward_net(net, ξμ)
    return τout ≠ σμ
end

compute_err(net::Net, patterns::Patterns) = compute_err(net, patterns.ξ, patterns.σ)

function subepoch!(net::Net, patterns::Patterns, patt_perm::PatternsPermutation, params::Params)
    @extract patterns  : ξ σ
    @extract patt_perm : batch

    reset_grads!(net)
    for μ in get_batch(patt_perm)
	ξμ, σμ = ξ[μ], σ[μ]
        h, τ, hout, τout = forward_net(net, ξμ)
        τout == σμ && continue
        compute_gd!(net, patterns, μ, h, τ, hout, params)
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

function report(ep::Int, errc, minerrc, errs::Vector, minerrs::Vector, dist::Vector{Int}, params::Params, quiet::Bool, outfile::AbstractString)
    @extract params : η λ γ
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
                M::Integer = 10,
                y::Integer = 1,

                η::Float64 = 2.0,
                λ::Float64 = 0.1,
                γ::Float64 = Inf,
                ηfactor::Float64 = 1.0,
                λfactor::Float64 = 1.0,
                γstep::Float64 = 1.0,
                batch::Integer = 5,

                formula::Symbol = :simple,

                seed::Integer = 1,
                seed_run::Integer = 0,

                max_epochs::Real = 1_000,
                init_equal::Bool = true,
                waitcenter::Bool = false,
                center::Bool = false,

                outfile::AbstractString = "",
                quiet::Bool = false)

    srand(seed)

    formula ∈ [:simple, :corrected, :continuous] || throw(ArgumentError("formula must be either :simple, :corrected or :continuous, given : $formula"))

    λ == 0 && waitcenter && warn("λ=$λ waitcenter=$waitcenter")

    patterns = Patterns(N, M)

    params = Params(N, K, y, η, λ, γ)

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

    errc = compute_err(netc, patterns.ξ, patterns.σ)
    minerrc = errc

    errs = [compute_err(net, patterns.ξ, patterns.σ) for net in nets]
    minerrs = copy(errs)

    dist = [compute_dist(netc, net) for net in nets]

    init_outfile(outfile, y)
    report(0, errc, minerrc, errs, minerrs, dist, params, quiet, outfile)

    sub_epochs = (M + batch - 1) ÷ batch
    patt_perm = [PatternsPermutation(M, batch) for r = 1:y]
    δH = Array(Float64, N)

    minerr = min(minerrc, minimum(minerrs))

    ok = errc == 0 || (!waitcenter && minerr == 0)
    ep = 0

    while !ok && (ep < max_epochs)
	ep += 1
	for subep = 1:sub_epochs, r in randperm(y)
	    net = nets[r]
	    for k = 1:K
		copy!(old_J[k], net.W.J[k])
	    end
	    subepoch!(net, patterns, patt_perm[r], params)
	    if !center
		if formula == :simple || formula == :corrected
		    kickboth_traced!(net, netc, params, δH, old_J, formula == :corrected)
		elseif formula == :continuous
		    kickboth_traced_continuous!(net, netc, params, δH, old_J)
		end
	    elseif params.λ > 0
		kickboth!(net, netc, params, δH)
	    end
	end

	errc = compute_err(netc, patterns)
	minerrc = min(minerrc, errc)
	errc == 0 && (ok = true)
	for r = 1:y
	    net = nets[r]
	    errs[r] = compute_err(net, patterns)
	    minerrs[r] = min(minerrs[r], errs[r])
	    errs[r] == 0 && !waitcenter && (ok = true)
	    dist[r] = compute_dist(netc, net)
	end
	minerr = min(minerrc, minimum(minerrs))
	report(ep, errc, minerrc, errs, minerrs, dist, params, quiet, outfile)

	params.η *= ηfactor
	params.λ *= λfactor
	params.γ += γstep
    end

    !quiet && println(ok ? "SOLVED" : "FAILED")

    return ok, ep, minerr
end

end # module
