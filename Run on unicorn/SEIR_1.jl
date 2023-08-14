##### 1. Import packages #####
# SciML Tools
using OrdinaryDiffEq, SciMLSensitivity, Optimization, OptimizationOptimisers, OptimizationOptimJL

# Standard Libraries
using LinearAlgebra, Statistics, Random, Distributions

# External Libraries
using ComponentArrays, Lux, Zygote, Plots, StableRNGs, JLD, JLD2
gr()

# Set a random seed for reproducible behavior
rng = StableRNG(10)


##### 2. Define the SEIR model and sample one data set #####
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
dist = LogNormal(0, 0.05)
noise = rand(rng, dist, size(X[:, begin:5:end]))
X_noisy = noise .* X[:, begin:5:end]

save("Time 1.jld", "t", t)
save("True trajectory 1.jld", "X", X)
save("Noisy data 1.jld", "X_noisy", X_noisy)

plot(t, X', xlabel = "Time", ylabel = "Number of individuals", color = [:blue :goldenrod2 :magenta :green], 
     label = ["True trajectory" nothing nothing nothing], size=(600,400), margin=10Plots.mm)
scatter!(t[begin:5:end], X_noisy', color = [:blue :goldenrod2 :magenta :green], 
     label = ["Noisy data" nothing nothing nothing])
savefig("Data 1.pdf")

# Also, create 1 test data set by varying initial values
u0_test = [0.980, 0.012, 0.007, 0.001]
prob_test = ODEProblem(seir!, u0_test, tspan, p_ode)
X_test = Array(solve(prob_test, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = t))
save("Test data 1.jld", "X_test", X_test)


##### 3. Set up the UDE #####
# Define activation function
rbf(x) = exp.(-(x .^ 2))

# Multi-layer feed-forward neural network
model = Chain(Dense(4, 8, rbf), Dense(8, 8, rbf), Dense(8, 8, rbf), Dense(8, 4))

# Get initial parameters and state variables of the model
θ_init, st = Lux.setup(rng, model)
save("θ_init 1.jld", "θ_init", θ_init)

# Define the hybrid model
function ude_dynamics!(du, u, p, t)
    nn = Lux.apply(model, u, p.nn, st)[1]    
    du[1] = nn[1]
    du[2] = nn[2] 
    du[3] = -p.γ*u[3] + nn[3] 
    du[4] = p.γ*u[3] + nn[4]
end


##### 4. Method to train the UDE using different regularization techniques and initial values #####
function train_ude(reg_ω, γ_init, adam_epochs, bfgs_epochs)
    # Set up the training
    p_init = (nn = θ_init, γ = γ_init)
    p_init = ComponentVector(p_init)
    prob_ude = ODEProblem(ude_dynamics!, u0, tspan, p_init)

    function predict(p, saveat = t[begin:5:end])   # p contains NN parameters θ and mechanistic parameter γ of the ODE
        _prob = remake(prob_ude, p = p)
        Array(solve(_prob, Tsit5(), abstol = 1e-6, reltol = 1e-6, saveat = saveat,
                    sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
    end
   
    # Define the loss function
    function loss(x, p)   # x: optimized variable, p: parametrization of the objective
        if p.reg == nothing
            X̂ = predict(x)
            if size(X̂) == size(X_noisy)
                mean(abs2, X_noisy .- X̂)  
            else
                Inf
            end
        elseif p.reg == "L2 θ"
            X̂ = predict(x)
            if size(X̂) == size(X_noisy)
                mean(abs2, X_noisy .- X̂) + p.ω * norm(x.nn)
            else
                Inf
            end
        elseif p.reg == "Integral"
            X̂_fine = predict(x, t)
            if size(X̂_fine) == size(X)
                X̂ = X̂_fine[:, begin:5:end]
                mean(abs2, X_noisy .- X̂) + p.ω * norm(Lux.apply(model, X̂_fine[:, begin:end-1], x.nn, st)[1], 1)
            else 
                Inf
            end
        end
    end;
    
    # Set up histories for loss and parameter estimates
    losses = Float64[]
    γ_hist = Float64[]
    function callback(p, l)
        push!(losses, l)
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
    
    result = (reg = reg, ω = ω, adam_epochs = adam_epochs, bfgs_epochs = bfgs_epochs, losses = losses, γ_init = γ_init, 
              γ_hist = γ_hist, p_trained = p_trained)
    save("Results 1/result_$(reg)_$(ω)_$(γ_init)_$(adam_epochs)_$(bfgs_epochs)_$(array_nr).jld2", "result", result)
    return result
end 


##### 5. Method to visualize the results #####
function nrmse(y_true, y_pred)
    mse = mean(abs2, y_pred .- y_true)
    rmse = sqrt(mse)
    nrmse = rmse / (maximum(y_true) - minimum(y_true))
    return nrmse
end

function visualize(result) 
    # Unpack result
    reg, ω, adam_epochs, bfgs_epochs, losses, γ_init, γ_hist, p_trained = result
        
    # Convergence plot
    p1 = plot(1:adam_epochs, losses[1:adam_epochs], yaxis=:log10, xlabel="Iterations", ylabel="Loss", label="ADAM",
              color=:blue)
    plot!(adam_epochs+1:length(losses), losses[adam_epochs+1:end], yaxis=:log10, label="BFGS", color=:red)
    
    # Parameter estimate plot
    p2 = plot(1:adam_epochs, γ_hist[1:adam_epochs], xlabel="Iterations", ylabel="Estimated γ", label="ADAM", color=:blue)
    plot!(adam_epochs+1:length(γ_hist), γ_hist[adam_epochs+1:end], label="BFGS", color=:red)
    plot!(1:length(γ_hist), p_ode[3]*ones(length(γ_hist)), linestyle = :dash, label="Ground truth", color=:grey10)
    
    # Data fit
    prob_ude_trained = ODEProblem(ude_dynamics!, u0, tspan, p_trained)
    X̂ = Array(solve(prob_ude_trained, Tsit5(), abstol = 1e-6, reltol = 1e-6, saveat = t))
    p3 = plot(t, X̂', xlabel = "Time", ylabel = "Number of individuals", color = [:blue :goldenrod2 :magenta :green],
              label = ["UDE approximation" nothing nothing nothing])
    scatter!(t[begin:5:end], X_noisy', color = [:blue :goldenrod2 :magenta :green], 
             label = ["Noisy data" nothing nothing nothing])
    
    # Approximation of true trajectory
    p4 = plot(t, X', xlabel = "Time", ylabel = "Number of individuals", color = [:blue :goldenrod2 :magenta :green], 
              label = ["True trajectory" nothing nothing nothing])
    plot!(t, X̂', linestyle = :dash, color = [:blue :goldenrod2 :magenta :green], 
          label = ["UDE approximation" nothing nothing nothing])
    
    # True residual dynamics (i.e. interactions of predictors) along the true trajectory
    α, β, γ, N = p_ode
    dynamics_true = [-β/N * (X[1,:] .* X[3,:])'; β/N * (X[1,:] .* X[3,:])' - α * X[2,:]'; α * X[2,:]'; zeros(length(t))'] 
    # Neural network guess along the true trajectory
    dynamics_nn = Lux.apply(model, X, p_trained.nn, st)[1]
    # Compare true and learned residual dynamics
    p5 = plot(t, dynamics_true', xlabel = "Time", ylabel = "Residual dynamics", 
              color = [:blue :goldenrod2 :magenta :green], label = ["True" nothing nothing nothing])
    plot!(t, dynamics_nn', color = [:blue :goldenrod2 :magenta :green], linestyle = :dash, 
          label = ["NN(S,E,I,R)" nothing nothing nothing])
    # Compute NRMSE of learned residual dynamics along the true trajectory
    nrmse_train = nrmse(dynamics_true, dynamics_nn)  
    # Do the same for the test data
    dynamics_test = [-β/N * (X_test[1,:] .* X_test[3,:])'; β/N * (X_test[1,:] .* X_test[3,:])' - α * X_test[2,:]';
                     α * X_test[2,:]'; zeros(length(t))']
    dynamics_nn_test = Lux.apply(model, X_test, p_trained.nn, st)[1]
    nrmse_test = nrmse(dynamics_test, dynamics_nn_test)
    # Save NRMSE scores
    nrmses = (train = nrmse_train, test = nrmse_test)    
    save("Results 1/NRMSE_$(reg)_$(ω)_$(γ_init)_$(adam_epochs)_$(bfgs_epochs)_$(array_nr).jld2", "nrmses", nrmses)

    # Combined plot  
    plot(p1, p2, p3, p4, p5, size=(1000,1200), layout=(3,2), margin=16Plots.mm,
         plot_title="Regularization: $(reg), ω: $(ω), NRMSE: $(round(nrmse_train, digits = 4))")
    savefig("Plots 1/Plot_$(reg)_$(ω)_$(γ_init)_$(adam_epochs)_$(bfgs_epochs)_$(array_nr).pdf")
end


##### 6. Define hyperparameter grid ##### 
reg_ω_grid = ((nothing, nothing), ("L2 θ", 10.0), ("L2 θ", 1.0), ("L2 θ", 0.1), ("L2 θ", 0.01), ("L2 θ", 0.005), 
              ("L2 θ", 0.001), ("L2 θ", 0.0005), ("L2 θ", 0.0001), ("L2 θ", 0.00001), ("Integral", 10.0), ("Integral", 1.0), 
              ("Integral", 0.1), ("Integral", 0.01), ("Integral", 0.005), ("Integral", 0.001), ("Integral", 0.0005), 
              ("Integral", 0.0001), ("Integral", 0.00001))
γ_init_grid = round.(p_ode[3] .* (1.0/5.0, 1.0/2.0, 1.0/1.15, 1.0, 1.15, 2.0, 5.0), digits=2)
adam_epochs_grid = (7500, 7000, 6000, 5000, 4000)
bfgs_epochs_grid = (22500, )
experiments = collect(Iterators.product(reg_ω_grid, γ_init_grid, adam_epochs_grid, bfgs_epochs_grid))

# Select hyperparameters via slurm array id
array_nr = parse(Int, ARGS[1])
reg_ω, γ_init, adam_epochs, bfgs_epochs = experiments[array_nr]
println("Experiment 1: $(reg_ω[1])_$(reg_ω[2])_$(γ_init)_$(adam_epochs)_$(bfgs_epochs)_$(array_nr)")


##### 7. Train using selected hyperparameters #####
result = train_ude(reg_ω, γ_init, adam_epochs, bfgs_epochs)
visualize(result)
