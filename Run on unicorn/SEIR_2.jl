##### 1. Import packages #####
# SciML Tools
using OrdinaryDiffEq, SciMLSensitivity, Optimization, OptimizationOptimisers, OptimizationOptimJL

# Standard Libraries
using LinearAlgebra, Statistics, Random, Distributions

# External Libraries
using ComponentArrays, Lux, Zygote, Plots, StableRNGs, JLD, JLD2
gr()

# Set a random seed for reproducible behavior
rng = StableRNG(1234)


##### 2. Define hyperparameter grid #####
reg_ω_grid = ((nothing, nothing), ("L2 θ", 0.01), ("L2 θ", 0.005), ("L2 θ", 0.001), ("L2 θ", 0.0005), ("L2 θ", 0.0001), 
              ("Integral", 0.01), ("Integral", 0.005), ("Integral", 0.001), ("Integral", 0.0005), ("Integral", 0.0001))
α_init_grid = (0.05, 0.2, 0.5, 0.7)
γ_init_grid = (0.1, 0.3, 0.6, 0.8)
adam_epochs_grid = (7500, 7000, 6000, 5000, 4000)
bfgs_epochs_grid = (22500, )
experiments = collect(Iterators.product(reg_ω_grid, α_init_grid, γ_init_grid, adam_epochs_grid, bfgs_epochs_grid))

# Select hyperparameters via slurm array id
array_nr = parse(Int, ARGS[1])
reg_ω, α_init, γ_init, adam_epochs, bfgs_epochs = experiments[array_nr]


##### 3. Define the SEIR model and sample one data set #####
function seir!(du, u, p, t)
    α, β, γ, N = p
    du[1] = -β*u[1]*u[3]/N
    du[2] = β*u[1]*u[3]/N - α*u[2]
    du[3] = α*u[2] - γ*u[3]
    du[4] = γ*u[3]
end

p_ode = [0.3, 1.0, 0.4, 1.0]
u0 = [0.995, 0.004, 0.001, 0.0]
tspan = (0.0, 50.0)

prob = ODEProblem(seir!, u0, tspan, p_ode)
sol = solve(prob, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = 1.0)

t = sol.t
X = Array(sol)
dist = LogNormal(0, 0.1)
noise = rand(rng, dist, size(X[3:4, begin:5:end]))
X_noisy = noise .* X[3:4, begin:5:end]

save("Time 2.jld", "t", t)
save("True trajectory 2.jld", "X", X)
save("Noisy data 2.jld", "X_noisy", X_noisy)

plot(t, X[3:4, :]', xlabel = "Time", ylabel = "Number of individuals", color = [:magenta :green],
     label = ["True trajectory" nothing], size=(600,400), margin=10Plots.mm)
scatter!(t[begin:5:end], X_noisy', color = [:magenta :green], label = ["Noisy data" nothing])
savefig("Data 2.pdf")


##### 4. Set up the UDE #####
# Define activation function
rbf(x) = exp.(-(x .^ 2))

# Multi-layer feed-forward neural network
model = Chain(Dense(4, 8, rbf), Dense(8, 8, rbf), Dense(8, 8, rbf), Dense(8, 4))

# Get initial parameters and state variables of the model
θ_init, st = Lux.setup(rng, model)
save("θ_init 2.jld", "θ_init", θ_init)

# Define the hybrid model
function ude_dynamics!(du, u, p, t)
    nn = Lux.apply(model, u, p.nn, st)[1]    
    du[1] = nn[1]
    du[2] = -p.α*u[2] + nn[2]
    du[3] = p.α*u[2] - p.γ*u[3] + nn[3] 
    du[4] = p.γ*u[3] + nn[4]
end


##### 5. Method to train the UDE using different regularization techniques and initial values #####
function train_ude(reg_ω, α_init, γ_init, adam_epochs, bfgs_epochs)
    # Set up the training
    p_init = (nn = θ_init, α = α_init, γ = γ_init)
    p_init = ComponentVector(p_init)
    prob_ude = ODEProblem(ude_dynamics!, u0, tspan, p_init)

    function predict(p, saveat = t[begin:5:end])   # p contains NN parameters θ and mechanistic parameters α, γ
        _prob = remake(prob_ude, p = p)
        Array(solve(_prob, Tsit5(), abstol = 1e-6, reltol = 1e-6, saveat = saveat,
                    sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
    end

    # Define the loss function
    function loss(x, p)   # x: optimized variable, p: parametrization of the objective
        if p.reg == nothing
            X̂ = predict(x)
            mean(abs2, X_noisy .- X̂[3:4, :])
        elseif p.reg == "L2 θ"
            X̂ = predict(x)
            mean(abs2, X_noisy .- X̂[3:4, :]) + p.ω * norm(x.nn)
        elseif p.reg == "Integral"
            X̂_fine = predict(x, t)
            mean(abs2, X_noisy .- X̂̂_fine[3:4, begin:5:end]) + p.ω * norm(Lux.apply(model, X̂_fine[:, begin:end-1], x.nn, st)[1], 1)
        end
    end;
    
    # Set up histories for loss and parameter estimates
    losses = Float64[]
    α_hist = Float64[]
    γ_hist = Float64[]
    function callback(p, l)
        push!(losses, l)
        push!(α_hist, p.α)
        push!(γ_hist, p.γ)
        return false
    end
    
    # Set up the optimization problem
    adtype = AutoZygote()
    optf = OptimizationFunction(loss, adtype)
    reg, ω = reg_ω
    optprob = OptimizationProblem(optf, p_init, (reg = reg, ω = ω))
    
    # Hybrid training procedure combining Adam and BFGS
    res1 = solve(optprob, ADAM(), callback = callback, maxiters = adam_epochs)
    optprob2 = OptimizationProblem(optf, res1.u, (reg = reg, ω = ω))
    res2 = solve(optprob2, LBFGS(), callback = callback, maxiters = bfgs_epochs)
    p_trained = res2.u
    
    result = (reg = reg, ω = ω, adam_epochs = adam_epochs, bfgs_epochs = bfgs_epochs, losses = losses, α_init = α_init,
              α_hist = α_hist, γ_init = γ_init, γ_hist = γ_hist, p_trained = p_trained)
    save("Results 2/result_$(reg)_$(ω)_$(α_init)_$(γ_init)_$(adam_epochs)_$(bfgs_epochs).jld2", "result", result)
    return result
end 


##### 6. Method to visualize the results #####
function nrmse(y_true, y_pred)
    mse = mean(abs2, y_pred .- y_true)
    rmse = sqrt(mse)
    nrmse = rmse / (maximum(y_true) - minimum(y_true))
    return nrmse
end

function visualize(result) 
    # Unpack result
    reg, ω, adam_epochs, bfgs_epochs, losses, α_init, α_hist, γ_init, γ_hist, p_trained = result
        
    # Convergence plot
    p1 = plot(1:adam_epochs, losses[1:adam_epochs], yaxis=:log10, xlabel="Iterations", ylabel="Loss", label="ADAM",
              color=:blue)
    plot!(adam_epochs+1:length(losses), losses[adam_epochs+1:end], yaxis=:log10, label="BFGS", color=:red)
    
    # Parameter estimate plot
    p2 = plot(1:adam_epochs, α_hist[1:adam_epochs], xlabel="Iterations", ylabel="Estimated α", label="ADAM", color=:blue)
    plot!(adam_epochs+1:length(α_hist), α_hist[adam_epochs+1:end], label="BFGS", color=:red)
    plot!(1:length(α_hist), p_ode[1]*ones(length(α_hist)), linestyle = :dash, label="Ground truth", color=:grey10)
    p3 = plot(1:adam_epochs, γ_hist[1:adam_epochs], xlabel="Iterations", ylabel="Estimated γ", label="ADAM", color=:blue)
    plot!(adam_epochs+1:length(γ_hist), γ_hist[adam_epochs+1:end], label="BFGS", color=:red)
    plot!(1:length(γ_hist), p_ode[3]*ones(length(γ_hist)), linestyle = :dash, label="Ground truth", color=:grey10)
    
    # Data fit
    prob_ude_trained = ODEProblem(ude_dynamics!, u0, tspan, p_trained)
    X̂ = Array(solve(prob_ude_trained, Tsit5(), abstol = 1e-6, reltol = 1e-6, saveat = t))
    p4 = plot(t, X[3:4, :]', xlabel = "Time", ylabel = "Number of individuals", color = [:magenta :green],
              label = ["True trajectory" nothing])
    plot!(t, X̂[3:4, :]', linestyle = :dash, color = [:magenta :green], label = ["UDE approximation" nothing])
    scatter!(t[begin:5:end], X_noisy', color = [:magenta :green], label = ["Noisy data" nothing])
    
    # Approximation of true trajectory
    p5 = plot(t, X', xlabel = "Time", ylabel = "Number of individuals", color = [:blue :goldenrod2 :magenta :green],
              label = ["True trajectory" nothing nothing nothing])
    plot!(t, X̂', linestyle = :dash, color = [:blue :goldenrod2 :magenta :green], 
          label = ["UDE approximation" nothing nothing nothing])
    
    # True residual dynamics (i.e. interactions of predictors) along the true trajectory
    α, β, γ, N = p_ode
    dynamics_true = [-β/N * (X[1,:] .* X[3,:])'; β/N * (X[1,:] .* X[3,:])'; zeros(length(t))'; zeros(length(t))']
    # Neural network guess along the true trajectory
    dynamics_nn = Lux.apply(model, X, p_trained.nn, st)[1]
    # Compare true and learned residual dynamics
    p6 = plot(t, dynamics_true', xlabel = "Time", ylabel = "Residual dynamics",
              color = [:blue :goldenrod2 :magenta :green], label = ["True" nothing nothing nothing])
    plot!(t, dynamics_nn', color = [:blue :goldenrod2 :magenta :green], linestyle = :dash, 
          label = ["NN(S,E,I,R)" nothing nothing nothing])
    # Compute and store NRMSE of learned residual dynamics
    nrmse_score = nrmse(dynamics_true, dynamics_nn)
    save("Results 2/NRMSE_$(reg)_$(ω)_$(α_init)_$(γ_init)_$(adam_epochs)_$(bfgs_epochs).jld", "nrmse_score", nrmse_score)

    # Combined plot  
    plot(p1, p2, p3, p4, p5, p6, size=(1000,1200), layout=(3,2), margin=16Plots.mm,
         plot_title="Regularization: $(reg), ω: $(ω), NRMSE: $(round(nrmse_score, digits = 4))")
    savefig("Plots 2/Plot_$(reg)_$(ω)_$(α_init)_$(γ_init)_$(adam_epochs)_$(bfgs_epochs).pdf")
end


##### 7. Train using selected hyperparameters #####
result = train_ude(reg_ω, α_init, γ_init, adam_epochs, bfgs_epochs)
visualize(result)
