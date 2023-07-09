# SciML Tools
using OrdinaryDiffEq, SciMLSensitivity, Optimization, OptimizationOptimisers, OptimizationOptimJL

# Standard Libraries
using LinearAlgebra, Statistics, Random, Distributions

# External Libraries
using ComponentArrays, Lux, Zygote, Plots, StableRNGs
gr()

# Set a random seed for reproducible behaviour
rng = StableRNG(1111);


#################################

function sir!(du, u, p, t)
    β, γ, N = p
    du[1] = -β*u[1]*u[2]/N
    du[2] = β*u[1]*u[2]/N - γ*u[2]
    du[3] = γ*u[2]
end

p_ODE = [0.6, 0.3, 1.0]
u0 = [0.999, 0.001, 0.0]
tspan = (0.0, 40.0);

prob = ODEProblem(sir!, u0, tspan, p_ODE)
sol = solve(prob, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = 1.0);
     


# multiplicative log-normal noise
t = sol.t
X = Array(sol)
dist = LogNormal(0, 0.01)
noise = rand(rng, dist, size(X))
X_noisy = noise .* X;
     


########## Define the model ##########


# Define activation function
rbf(x) = exp.(-(x .^ 2))

# Multi-layer feed-forward neural network
model = Chain(Dense(3, 5, rbf), Dense(5, 5, rbf), Dense(5, 5, rbf), Dense(5, 3, init_weight=Lux.zeros32))

# Get initial parameters and state variables of the model
p_init, st = Lux.setup(rng, model);


# Define the hybrid model ### TODO ###
β, γ, N = p_ODE

function ude_dynamics!(du, u, p, t)
    nn = model(u, p.nn, st)[1]   # Network prediction    
    du[1] = nn[1]
    du[2] = -p.γ*u[2] + nn[2] 
    du[3] = p.γ*u[2] + nn[3]
end

γ_init = 1.0

ps = (nn = p_init, γ = γ_init)
ps = ComponentVector(ps);

prob_ude = ODEProblem(ude_dynamics!, u0, tspan, ps);
 

function predict(p)   # p consists of the NN parameters and the mechanistic parameter of the ODE
    _prob = remake(prob_ude, u0=u0, tspan=tspan, p=p)
    Array(solve(_prob, Tsit5(), abstol = 1e-6, reltol = 1e-6, saveat = t,
    sensealg = ForwardDiffSensitivity()))
end;
     

function loss(p)
    X̂ = predict(p)
    mean(abs2, X_noisy .- X̂)   # MSE loss
end;
     

losses = Float64[]

function callback(p, l)
    push!(losses, l)
    n_iter = length(losses)
    if n_iter % 50 == 0   
        println("Loss after $(losses[end])")
    end
    return false
end;


########### train ude ##############

adtype = AutoZygote()   # automatic differentiation
optf = OptimizationFunction((x,p) -> loss(x), adtype)
optprob = OptimizationProblem(optf, ps);
     
res1 = Optimization.solve(optprob, ADAM(), callback = callback, maxiters = 100);



optprob2 = OptimizationProblem(optf, res1.u)
res2 = solve(optprob2, LBFGS(), callback = callback, maxiters = 1000)
p_trained = res2.u;
     
