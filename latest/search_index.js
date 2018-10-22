var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#BinaryCommitteeMachineRSGD.jl-documentation-1",
    "page": "Home",
    "title": "BinaryCommitteeMachineRSGD.jl documentation",
    "category": "section",
    "text": "This package implements the Replicated Stochastic Gradient Descent algorithm for committee machines with binary weights described in the paper Unreasonable Effectiveness of Learning Neural Networks: From Accessible States and Robust Ensembles to Basic Algorithmic Schemes by Carlo Baldassi, Christian Borgs, Jennifer Chayes, Alessandro Ingrosso, Carlo Lucibello, Luca Saglietti and Riccardo Zecchina, Proc. Natl. Acad. Sci. U.S.A. 113: E7655-E7662 (2016), doi:10.1073/pnas.1608103113.The package requires Julia 0.7 or later."
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "To install the module, use these commands from within Julia:julia> using Pkg\n\njulia> Pkg.clone(\"https://github.com/carlobaldassi/BinaryCommitteeMachineRSGD.jl\")Dependencies will be installed automatically."
},

{
    "location": "index.html#BinaryCommitteeMachineRSGD.Patterns",
    "page": "Home",
    "title": "BinaryCommitteeMachineRSGD.Patterns",
    "category": "type",
    "text": "Patterns(N, M)\n\nGenerates M random ±1 patterns of length N.\n\nPatterns(ξ, σ)\n\nEncapsulates the input patterns ξ and their associated desired outputs σ for use in replicatedSGD. The inputs ξ must be given as a vector of vectors, while the outputs σ must be given as a vector. In both cases, they are converted to ±1 values using their sign (more precisely, using x > 0 ? 1 : -1).\n\n\n\n\n\n"
},

{
    "location": "index.html#BinaryCommitteeMachineRSGD.replicatedSGD",
    "page": "Home",
    "title": "BinaryCommitteeMachineRSGD.replicatedSGD",
    "category": "function",
    "text": "replicatedSGD(patterns::Patterns; keywords...)\n\nRuns the replicated Stochastic Gradient Descent algorithm over the given patterns (see Patterns). It automatically detects the size of the input and initializes a system of interacting binary committee machines which collectively try to learn the patterns.\n\nThe function returns three values: a Bool with the success status, the number of epochs, and the minimum error achieved.\n\nThe available keyword arguments (note that the defaults are mostly not sensible, they must be collectively tuned):\n\nK (default=1): number of hidden units for each committee machine (size of the hidden layer)\ny (default=1): number of replicas\nη (default=2): initial value of the step for the energy (loss) term gradient\nλ (default=0.1): initial value of the step for the interaction gradient (called η′ in the paper)\nγ (default=Inf): initial value of the interaction strength\nηfactor (default=1): factor used to update η after each epoch\nλfactor (default=1): factor used to update λ after each epoch\nγstep (default=0.01): additive step used to update γ after each epoch\nbatch (default=5): minibatch size\nformula (default=:simple): used to choose the interaction update scheme when center=false; see below for available values\nseed (default=0): random seed; if 0, it is not used\nmax_epochs (default=1000): maximum number of epochs\ninit_equal (default=true): whether to initialize all replicated networks equally\nwaitcenter (default=false): whether to only exit successfully if the center replica has solved the problem\ncenter (default=false): whether to explicity use a central replica (if false, it is traced out)\noutfile (default=\"\"): name of a file where to output the results; if empty it\'s ignored\nquiet (default=false): whether to output information on screen\n\nThe possible values of the formula option are:\n\n:simple (the default): uses the simplest traced-out center formula (eq. (C7) in the paper)\n:corrected: applies the correction of eq. (C9) to the formula of eq. (C7)\n:continuous: version in which the center is continuous and traced-out\n:hard: same as :simple but uses a hard tanh, for improved performance\n\nExample of a good parameter configuration (for a committee with K=5 and N*K=1605 synapses overall, working at α=M/(NK)=0.5):\n\nok, epochs, minerr = replicatedSGD(Patterns(321, 802), K=5, y=7, batch=80, λ=0.75, γ=0.05, γstep=0.001, formula=:simple)\n\n\n\n\n\n"
},

{
    "location": "index.html#Usage-1",
    "page": "Home",
    "title": "Usage",
    "category": "section",
    "text": "The module is loaded as any other Julia module:julia> using BinaryCommitteeMachineRSGDThe code basically provides a single function which generates a system of interacting replicated committee machines and tries to learn some patterns. The function and the patterns constructor are documented below.PatternsreplicatedSGD"
},

]}
