module BinaryCommitteeMachineRSGD

export Patterns, Net, replicatedSGD

using ExtractMacro

typealias IVec Vector{Int}
typealias BVec BitVector
typealias BVec2 Vector{BVec}
typealias Vec Vector{Float64}
typealias Vec2 Vector{Vec}

# same as (2a-1) ⋅ (2b-1)
# note: no length checks
function pm1dot(a::BVec, b::BVec)
    # one way to write it avoiding allocations:
    # 4 * (a ⋅ b) - 2sum(a) - 2sum(b) + length(a)

    # ugly but slightly faster (length(a)-2sum(a$b) without allocations):
    l = length(a)
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
    function Patterns(ξ::Vector, σ::Vector)
        M = length(ξ)
        length(σ) == M || throw(ArgumentError("inconsistent vector lengths: ξ=$M σ=$(length(σ))"))
        M ≥ 1 || throw(ArgumentError("empty patterns – use Patterns(N, 0) if this is what you intended"))
        N = length(first(ξ))
        all(ξμ->length(ξμ)==N, ξ) || throw(ArgumentError("patterns length must all be the same, found: $(unique(map(length, ξ)))"))
        ξ = BVec[ξμ .> 0 for ξμ in ξ]
        σ = Int[2 * (σμ > 0) - 1 for σμ in σ]
        return new(N, M, ξ, σ)
    end
    function Patterns(N::Integer, M::Integer)
        N ≥ 1 && isodd(N) || throw(ArgumentError("N must be positive and odd, given: $N"))
        M ≥ 0 || throw(ArgumentError("M cannot be negative, given: $M"))
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
    y::Int64
    η::Float64
    λ::Float64
    γ::Float64
end

function update!(params::Params, ηfactor::Real, λfactor::Real, γstep::Real)
    params.η *= ηfactor
    params.λ *= λfactor
    params.γ += γstep
end

type Net
    N::Int64
    K::Int64
    J::BVec2
    H::Vec2
    ΔH::Vec2
    old_J::BVec2
    δH::Vec
    function Net(N, K)
	H = [rand(-1:2.:1, N) for k = 1:K]
	J = [H[k] .> 0 for k = 1:K]
        ΔH = [zeros(Float64, N) for k = 1:K]
        old_J = [copy(Jk) for Jk in J]
        return new(N, K, J, H, ΔH, old_J)
    end
    function Net(H::Vec2)
	K = length(H)
	K ≥ 1 || throw(ArgumentError("empty initial vector H"))
	N = length(H[1])
	all(h->length(h)==N, H) || throw(ArgumentError("invalid initial vector H, lengths are not all equal: $(map(h->length(h), H))"))
	J = [H[k] .> 0 for k = 1:K]
	ΔH = [zeros(Float64, N) for k = 1:K]
        old_J = [copy(Jk) for Jk in J]
        return new(N, K, J, H, ΔH, old_J)
    end
end

reset_grads!(net::Net) = map!(x->fill!(x, 0.0), net.ΔH)
save_J!(net::Net) = map(k->copy!(net.old_J[k], net.J[k]), 1:net.K)
init_δH!(net::Net) = net.δH = Array(Float64, net.N)

Base.copy(net::Net) = Net(deepcopy(net.H))

function mean_net(nets::Vector{Net})
    y = length(nets)
    y ≥ 1 || throw(ArgumentError("empty nets vector"))
    K = nets[1].K
    all(net->net.K == K, nets) || throw(ArgumentError("heterogeneous nets: $(map(net->net.K, nets))"))

    Net([2/y * sum([net.J[k] for net in nets]) .- 1 for k = 1:K])
end

function update_net!(net::Net)
    @extract net : N K H J ΔH

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
    @extract netr : N K J
    @inbounds for k = 1:K
        h[k] = pm1dot(ξμ, J[k])
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
        @extract net      : N K H ΔH
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

function kickboth!(net::Net, netc::Net, params::Params)
    @extract params : λ
    @extract net    : N K H J
    @extract netc   : Hc=H Jc=J δH

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

function kickboth_traced!(net::Net, netc::Net, params::Params, corrected::Bool = false)
    @extract params : y γ λ
    @extract net    : N K H J old_J
    @extract netc   : Hc=H Jc=J δH

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

function kickboth_traced_continuous!(net::Net, netc::Net, params::Params)
    @extract params : y γ λ
    @extract net    : N K H J old_J
    @extract netc   : Hc=H Jc=J δH

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
    @extract net1 : J1=J
    @extract net2 : J2=J

    return sum([sum(j1 $ j2) for (j1,j2) in zip(J1,J2)])
end

function init_outfile(outfile::AbstractString, y::Int)
    !isempty(outfile) && isfile(outfile) && error("outfile exists: $outfile")
    !isempty(outfile) && open(outfile, "w") do outf
        println(outf, "#epoch err(Wc) err(best) | ", join(["err(W$i)" for i = 1:y], " "), " | λ γ | ", join(["d(W$i)" for i = 1:y], " "))
    end
end

function report(ep::Int, errc::Int, minerrc::Int, errs::IVec, minerrs::IVec, dist::IVec, params::Params, quiet::Bool, outfile::AbstractString)
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

function replicatedSGD(; N::Integer = 101, M::Integer = 10, seed::Integer = 1, kw...)
    N ≥ 1 && isodd(N) || throw(ArgumentError("N must be positive and odd, given: $N"))
    M ≥ 0 || throw(ArgumentError("M cannot be negative, given: $M"))
    srand(seed)

    srand(seed)
    patterns = Patterns(N, M)

    replicatedSGD(patterns; kw...)
end

function replicatedSGD(patterns::Patterns;
                       K::Integer = 1,
                       y::Integer = 1,

                       η::Float64 = 2.0,
                       λ::Float64 = 0.1,
                       γ::Float64 = Inf,
                       ηfactor::Float64 = 1.0,
                       λfactor::Float64 = 1.0,
                       γstep::Float64 = 1.0,
                       batch::Integer = 5,

                       formula::Symbol = :simple,

                       seed_run::Integer = 0,

                       max_epochs::Real = 1_000,
                       init_equal::Bool = true,
                       waitcenter::Bool = false,
                       center::Bool = false,

                       outfile::AbstractString = "",
                       quiet::Bool = false)

    @extract patterns : N M
    K ≥ 1 && isodd(K) || throw(ArgumentError("K must be positive and odd, given: $K"))
    y ≥ 1 || throw(ArgumentError("y must be positive, given: $y"))
    batch ≥ 1 || throw(ArgumentError("batch must be positive, given: $batch"))
    formula ∈ [:simple, :corrected, :continuous] || throw(ArgumentError("formula must be either :simple, :corrected or :continuous, given : $formula"))
    max_epochs ≥ 0 || throw(ArgumentError("max_epochs cannot be negative, given: $max_epochs"))

    λ == 0 && waitcenter && warn("λ=$λ waitcenter=$waitcenter")

    seed_run ≠ 0 && srand(seed_run)

    params = Params(y, η, λ, γ)

    local netc::Net
    nets = Array(Net, y)

    if center || init_equal
	netc = Net(N, K)
    end

    for r = 1:y
        if init_equal
            nets[r] = copy(netc)
        else
	    nets[r] = Net(N, K)
        end
    end

    !center && (netc = mean_net(nets))
    init_δH!(netc)

    errc = compute_err(netc, patterns.ξ, patterns.σ)
    minerrc = errc

    errs = [compute_err(net, patterns.ξ, patterns.σ) for net in nets]
    minerrs = copy(errs)

    minerr = min(minerrc, minimum(minerrs))

    dist = [compute_dist(netc, net) for net in nets]

    init_outfile(outfile, y)
    report(0, errc, minerrc, errs, minerrs, dist, params, quiet, outfile)

    sub_epochs = (M + batch - 1) ÷ batch
    patt_perm = [PatternsPermutation(M, batch) for r = 1:y]

    ok = errc == 0 || (!waitcenter && minerr == 0)
    ep = 0
    while !ok && (ep < max_epochs)
	ep += 1
	for subep = 1:sub_epochs, r in randperm(y)
	    net = nets[r]
            save_J!(net)
	    subepoch!(net, patterns, patt_perm[r], params)
	    if !center
		if formula == :simple || formula == :corrected
		    kickboth_traced!(net, netc, params, formula == :corrected)
		elseif formula == :continuous
                    kickboth_traced_continuous!(net, netc, params)
		end
	    elseif params.λ > 0
                kickboth!(net, netc, params)
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

        update!(params, ηfactor, λfactor, γstep)
    end

    !quiet && println(ok ? "SOLVED" : "FAILED")

    return ok, ep, minerr
end

end # module
