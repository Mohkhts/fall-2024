########################################
# Question 1 - Multinomial Logit Model #
########################################


using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using CSV
using FreqTables
using ForwardDiff

# Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Prepare the data
X = Matrix{Float64}([df.age df.white df.collgrad])
Z = Matrix{Float64}(hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8))
y = Vector{Int}(df.occ_code)

# Define the number of choices
const J = 8

# Function to compute choice probabilities
function choice_probs(β::AbstractMatrix, γ::AbstractVector, X::AbstractMatrix, Z::AbstractMatrix)
    N = size(X, 1)
    V = zeros(eltype(β), N, J)
    for j in 1:J
        V[:, j] = X * β[:, j] .+ Z[:, j] .* γ[1]
    end
    eV = exp.(V)
    P = eV ./ sum(eV, dims=2)
    return P
end

# Log-likelihood function
function loglikelihood(θ::AbstractVector, X::AbstractMatrix, Z::AbstractMatrix, y::AbstractVector)
    N, K = size(X)
    β = reshape(θ[1:K*(J-1)], K, J-1)
    γ = θ[K*(J-1)+1:end]
    
    P = choice_probs(hcat(β, zeros(eltype(θ), K, 1)), γ, X, Z)
    
    ll = 0.0
    for i in 1:N
        ll += log(P[i, y[i]])
    end
    
    return -ll  # Negative log-likelihood for minimization
end

# Gradient of the log-likelihood function using ForwardDiff
function loglikelihood_gradient!(G, θ, X, Z, y)
    G[:] = ForwardDiff.gradient(θ -> loglikelihood(θ, X, Z, y), θ)
end

# Set up the optimization problem
N, K = size(X)
initial_θ = vcat(vec(zeros(K, J-1)), zeros(1))  # Initialize parameters
lower = fill(-Inf, length(initial_θ))
upper = fill(Inf, length(initial_θ))

# Optimize using L-BFGS
opt = Optim.optimize(
    θ -> loglikelihood(θ, X, Z, y),
    (G, θ) -> loglikelihood_gradient!(G, θ, X, Z, y),
    lower,
    upper,
    initial_θ,
    Fminbox(LBFGS()),
    Optim.Options(show_trace = true, iterations = 1000)
)

# Extract the results
θ_hat = Optim.minimizer(opt)
β_hat = reshape(θ_hat[1:K*(J-1)], K, J-1)
γ_hat = θ_hat[K*(J-1)+1:end]

# Compute standard errors
H = ForwardDiff.hessian(θ -> loglikelihood(θ, X, Z, y), θ_hat)
se = sqrt.(diag(inv(H)))

# Print results
println("Estimated β:")
display(β_hat)
println("\nEstimated γ:")
display(γ_hat)
println("\nStandard Errors:")
display(se)


#######################################
# Question2
#######################################
# The coefficient gamma is the change in utility with a 1-unit change in log wages
# In Question 1, problem set 3 gamma was negative, but here it is positive which makes more sense because an increase in log wage should increase the utility

#######################################
# Question3
#######################################

using Distributions
using LinearAlgebra  # Add this line to import norm function
using Statistics  # For mean function

include("lgwt.jl")  # Make sure this file is in your working directory

# Define the Normal distribution
d = Normal(0, 1)  # mean=0, standard deviation=1

# (a) Verify that integrating over the density equals 1
nodes, weights = lgwt(7, -4, 4)
integral_density = sum(weights .* pdf.(d, nodes))
println("Integral of density (should be close to 1): ", integral_density)

# Verify that the expectation (mean) is 0
expectation = sum(weights .* nodes .* pdf.(d, nodes))
println("Expectation (should be close to 0): ", expectation)

# (b) Compute integrals for N(0,2) distribution
d_var2 = Normal(0, 2)

# Function to compute integral using quadrature
function quad_integral(f, d, n_points, lower, upper)
    nodes, weights = lgwt(n_points, lower, upper)
    return sum(weights .* f.(nodes) .* pdf.(d, nodes))
end

# Compute integral of x^2 * f(x) with 7 and 10 quadrature points
integral_7 = quad_integral(x -> x^2, d_var2, 7, -5*sqrt(2), 5*sqrt(2))
integral_10 = quad_integral(x -> x^2, d_var2, 10, -5*sqrt(2), 5*sqrt(2))

println("Integral of x^2 * f(x) with 7 points: ", integral_7)
println("Integral of x^2 * f(x) with 10 points: ", integral_10)
println("True variance of N(0,2): ", var(d_var2))

# (c) Monte Carlo integration
function monte_carlo_integral(f, d, n_draws, lower, upper)
    draws = rand(Uniform(lower, upper), n_draws)
    return (upper - lower) * mean(f.(draws) .* pdf.(d, draws))
end

# Compute integrals using Monte Carlo with 1,000,000 draws
mc_integral_x2 = monte_carlo_integral(x -> x^2, d_var2, 1_000_000, -5*sqrt(2), 5*sqrt(2))
mc_integral_x = monte_carlo_integral(x -> x, d_var2, 1_000_000, -5*sqrt(2), 5*sqrt(2))
mc_integral_1 = monte_carlo_integral(x -> 1, d_var2, 1_000_000, -5*sqrt(2), 5*sqrt(2))

println("Monte Carlo integral of x^2 * f(x): ", mc_integral_x2)
println("Monte Carlo integral of x * f(x): ", mc_integral_x)
println("Monte Carlo integral of f(x): ", mc_integral_1)

# Compare Monte Carlo integration with 1,000 vs 1,000,000 draws
mc_integral_1k = monte_carlo_integral(x -> x^2, d_var2, 1_000, -5*sqrt(2), 5*sqrt(2))
println("Monte Carlo integral of x^2 * f(x) with 1,000 draws: ", mc_integral_1k)
println("Monte Carlo integral of x^2 * f(x) with 1,000,000 draws: ", mc_integral_x2)

#######################################
# Question4
#######################################
using Optim
using LinearAlgebra
using Distributions
using Random

# Include the lgwt.jl file for Gauss-Legendre quadrature
include("lgwt.jl")

# Function to compute choice probabilities
function choice_probs(β, γ, X, Z)
    # Implementation of choice probabilities calculation
    # This should be similar to the multinomial logit, but incorporate γ
end

# Likelihood function for mixed logit with quadrature
function log_likelihood(θ, X, Z, y, num_quadrature_points)
    N, T = size(y, 1), 1  # Assuming panel data structure
    J = maximum(y)
    
    # Extract parameters
    β = θ[1:end-2]
    μ_γ, σ_γ = θ[end-1:end]
    
    # Set up quadrature
    nodes, weights = lgwt(num_quadrature_points, -4*σ_γ + μ_γ, 4*σ_γ + μ_γ)
    
    ll = 0.0
    for i in 1:N
        for t in 1:T
            prob_sum = 0.0
            for (node, weight) in zip(nodes, weights)
                γ = node  # The integration variable
                probs = choice_probs(β, γ, X[i,:], Z[i,:,:])
                prob_sum += weight * probs[y[i,t]]
            end
            ll += log(prob_sum)
        end
    end
    
    return -ll  # Return negative log-likelihood for minimization
end

# Optimization
function estimate_mixed_logit(X, Z, y, num_quadrature_points; initial_params=nothing)
    if isnothing(initial_params)
        # Initialize parameters (you might want to use results from multinomial logit)
        initial_params = [randn(size(X,2)*(size(y,2)-1)); 0.0; 1.0]
    end
    
    obj = θ -> log_likelihood(θ, X, Z, y, num_quadrature_points)
    
    result = optimize(obj, initial_params, BFGS(), autodiff=:forward)
    
    return result
end

# Main execution
X = # Your X data
Z = # Your Z data
y = # Your y data
num_quadrature_points = 7  # You can adjust this

result = estimate_mixed_logit(X, Z, y, num_quadrature_points)

# Extract and print results
estimated_params = Optim.minimizer(result)
println("Estimated parameters: ", estimated_params)
println("Log-likelihood: ", -Optim.minimum(result))

#######################################
# Question5
#######################################

using Optim, LinearAlgebra, Random, Distributions

function mixed_logit_monte_carlo(X, Z, y; num_draws=1000, μ_γ=0.0, σ_γ=1.0)
    N, T = size(y, 1), 1  # Assuming each individual has one observation
    J = maximum(y)  # Number of choice alternatives

    function log_likelihood(θ)
        β = reshape(θ[1:end-2], :, J-1)
        μ_γ, σ_γ = θ[end-1], θ[end]

        ll = 0.0
        for i in 1:N
            prob_i = 0.0
            for _ in 1:num_draws
                γ = rand(Normal(μ_γ, σ_γ))
                
                prob_j = zeros(J)
                for j in 1:J-1
                    v = X[i,:] * β[:,j] + γ * (Z[i,j] - Z[i,J])
                    prob_j[j] = exp(v)
                end
                prob_j[J] = 1.0
                prob_j ./= sum(prob_j)
                
                prob_i += prob_j[y[i]]
            end
            ll += log(prob_i / num_draws)
        end
        return -ll  # Return negative log-likelihood for minimization
    end

    # Initial parameter values (you may need to adjust these)
    initial_θ = vcat(vec(zeros(size(X, 2), J-1)), μ_γ, σ_γ)

    # Optimize using BFGS
    result = optimize(log_likelihood, initial_θ, BFGS(), autodiff=:forward)

    return result
end

#######################################
# Question6
#######################################

# Load necessary packages at the top level
using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables, ForwardDiff, Distributions

# Include the lgwt.jl for quadrature
include("lgwt.jl")  # Make sure this file is in the current working directory

function allwrap()
    ##############################
    # Multinomial Logit Estimation
    ##############################
    
    # Load the data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS4-mixture/nlsw88t.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    
    # Prepare the data
    X = Matrix{Float64}([df.age df.white df.collgrad])
    Z = Matrix{Float64}(hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8))
    y = Vector{Int}(df.occ_code)
    
    J = 8  # Number of choices (no need for const here)
    
    # Function to compute choice probabilities
    function choice_probs(β::AbstractMatrix, γ::AbstractVector, X::AbstractMatrix, Z::AbstractMatrix)
        N = size(X, 1)
        V = zeros(eltype(β), N, J)
        for j in 1:J
            V[:, j] = X * β[:, j] .+ Z[:, j] .* γ[1]
        end
        eV = exp.(V)
        P = eV ./ sum(eV, dims=2)
        return P
    end
    
    # Log-likelihood function
    function loglikelihood(θ::AbstractVector, X::AbstractMatrix, Z::AbstractMatrix, y::AbstractVector)
        N, K = size(X)
        β = reshape(θ[1:K*(J-1)], K, J-1)
        γ = θ[K*(J-1)+1:end]
        
        P = choice_probs(hcat(β, zeros(eltype(θ), K, 1)), γ, X, Z)
        
        ll = 0.0
        for i in 1:N
            ll += log(P[i, y[i]])
        end
        
        return -ll  # Negative log-likelihood for minimization
    end
    
    # Gradient of the log-likelihood function
    function loglikelihood_gradient!(G, θ, X, Z, y)
        G[:] = ForwardDiff.gradient(θ -> loglikelihood(θ, X, Z, y), θ)
    end
    
    # Set up optimization
    N, K = size(X)
    initial_θ = vcat(vec(zeros(K, J-1)), zeros(1))  # Initialize parameters
    lower = fill(-Inf, length(initial_θ))
    upper = fill(Inf, length(initial_θ))
    
    opt = Optim.optimize(
        θ -> loglikelihood(θ, X, Z, y),
        (G, θ) -> loglikelihood_gradient!(G, θ, X, Z, y),
        lower,
        upper,
        initial_θ,
        Fminbox(LBFGS()),
        Optim.Options(show_trace = true, iterations = 1000)
    )
    
    θ_hat = Optim.minimizer(opt)
    β_hat = reshape(θ_hat[1:K*(J-1)], K, J-1)
    γ_hat = θ_hat[K*(J-1)+1:end]
    
    # Compute standard errors
    H = ForwardDiff.hessian(θ -> loglikelihood(θ, X, Z, y), θ_hat)
    se = sqrt.(diag(inv(H)))
    
    println("Estimated β:")
    display(β_hat)
    println("Estimated γ:")
    display(γ_hat)
    println("Standard Errors:")
    display(se)

    #########################
    # Quadrature (Question 3)
    #########################
    
    d = Normal(0, 1)
    nodes, weights = lgwt(7, -4, 4)
    
    integral_density = sum(weights .* pdf.(d, nodes))
    expectation = sum(weights .* nodes .* pdf.(d, nodes))
    
    println("Integral of density (should be close to 1): ", integral_density)
    println("Expectation (should be close to 0): ", expectation)

    d_var2 = Normal(0, 2)

    # Function to compute integrals
    function quad_integral(f, d, n_points, lower, upper)
        nodes, weights = lgwt(n_points, lower, upper)
        return sum(weights .* f.(nodes) .* pdf.(d, nodes))
    end
    
    integral_7 = quad_integral(x -> x^2, d_var2, 7, -5*sqrt(2), 5*sqrt(2))
    integral_10 = quad_integral(x -> x^2, d_var2, 10, -5*sqrt(2), 5*sqrt(2))
    
    println("Integral of x^2 * f(x) with 7 points: ", integral_7)
    println("Integral of x^2 * f(x) with 10 points: ", integral_10)
    println("True variance of N(0,2): ", var(d_var2))

    # Monte Carlo integration (Question 3 part c)
    function monte_carlo_integral(f, d, n_draws, lower, upper)
        draws = rand(Uniform(lower, upper), n_draws)
        return (upper - lower) * mean(f.(draws) .* pdf.(d, draws))
    end
    
    mc_integral_x2 = monte_carlo_integral(x -> x^2, d_var2, 1_000_000, -5*sqrt(2), 5*sqrt(2))
    mc_integral_x = monte_carlo_integral(x -> x, d_var2, 1_000_000, -5*sqrt(2), 5*sqrt(2))
    mc_integral_1 = monte_carlo_integral(x -> 1, d_var2, 1_000_000, -5*sqrt(2), 5*sqrt(2))
    
    println("Monte Carlo integral of x^2 * f(x): ", mc_integral_x2)
    println("Monte Carlo integral of x * f(x): ", mc_integral_x)
    println("Monte Carlo integral of f(x): ", mc_integral_1)

    ##############################
    # Mixed Logit Estimation (Question 4 and 5)
    ##############################
    
     # Print final results
    println("All computations finished successfully.")
end



#######################################
# Question7
#######################################
#UNIT TEST FOR QUESTION 1
using Test
using LinearAlgebra

# Define choice_probs function (needs to be defined for the tests)
function choice_probs(β::AbstractMatrix, γ::AbstractVector, X::AbstractMatrix, Z::AbstractMatrix)
    N = size(X, 1)
    V = zeros(eltype(β), N, size(β, 2))
    for j in 1:size(β, 2)
        V[:, j] = X * β[:, j] .+ Z[:, j] .* γ[1]
    end
    eV = exp.(V)
    P = eV ./ sum(eV, dims=2)
    return P
end

# Unit test
@testset "Choice Model Tests" begin
    # Prepare mock data for testing
    X = randn(100, 3)  # 100 samples, 3 features
    Z = randn(100, 8)  # 100 samples, 8 choices
    β_test = randn(3, 8)  # 3 features, 8 choices
    γ_test = randn(1)

    # Test choice_probs function
    P_test = choice_probs(β_test, γ_test, X, Z)
    @test size(P_test) == (size(X, 1), 8)  # Probabilities for 8 choices
    @test all(0 .<= P_test .<= 1)  # Probabilities must be between 0 and 1
    @test all(isapprox.(sum(P_test, dims=2), 1, atol=1e-8))  # Row sums must be 1 (element-wise comparison)
end

#UNIT TEST FOR QUESTION 3

using Test
using Distributions
using LinearAlgebra
include("lgwt.jl")

# Function to compute integral using quadrature (to be tested)
function quad_integral(f, d, n_points, lower, upper)
    nodes, weights = lgwt(n_points, lower, upper)
    return sum(weights .* f.(nodes) .* pdf.(d, nodes))
end

# Increase quadrature points and bounds for more precision
@testset "Quadrature Integration Tests" begin
    # Define a simple test function, e.g., f(x) = x^2
    f(x) = x^2
    d = Normal(0, 1)

    # Test the quadrature integral of x^2 over wider bounds with more points
    integral_result = quad_integral(f, d, 50, -10, 10)
    @test abs(integral_result - 1.0) < 1e-2

    # Now for N(0,2) with 50 quadrature points and wider bounds
    d_var2 = Normal(0, 2)
    integral_var2_50 = quad_integral(f, d_var2, 50, -10, 10)
    @test abs(integral_var2_50 - 4.0) < 1e-2

    # Test with 100 quadrature points
    integral_var2_100 = quad_integral(f, d_var2, 100, -10, 10)
    @test abs(integral_var2_100 - 4.0) < 1e-2
end

#UNIT TEST FOR QUESTION 4

using Test

# Re-defining necessary functions for the unit test
function choice_probs(β::AbstractMatrix, γ::AbstractVector, X::AbstractMatrix, Z::AbstractMatrix)
    N = size(X, 1)
    J = size(β, 2)
    V = zeros(eltype(β), N, J)
    for j in 1:J
        V[:, j] = X * β[:, j] .+ Z[:, j] .* γ[1]
    end
    eV = exp.(V)
    P = eV ./ sum(eV, dims=2)
    return P
end

function loglikelihood(θ::AbstractVector, X::AbstractMatrix, Z::AbstractMatrix, y::AbstractVector)
    N, K = size(X)
    J = size(Z, 2)
    β = reshape(θ[1:K*(J-1)], K, J-1)
    γ = θ[K*(J-1)+1:end]
    
    P = choice_probs(hcat(β, zeros(eltype(θ), K, 1)), γ, X, Z)
    
    ll = 0.0
    for i in 1:N
        ll += log(P[i, y[i]])
    end
    
    return -ll  # Negative log-likelihood for minimization
end

# Unit test code
@testset "Question 4 Tests" begin
    # Mock data for testing
    X = randn(100, 3)  # 100 samples, 3 features
    Z = randn(100, 8)  # 100 samples, 8 choices
    y = rand(1:8, 100)  # Randomly generated choice data
    β_test = randn(3, 8)  # 3 features, 8 choices
    γ_test = randn(1)

    # Test choice_probs function
    P_test = choice_probs(β_test, γ_test, X, Z)
    @test size(P_test) == (size(X, 1), 8)  # Probabilities for 8 choices
    @test all(0 .<= P_test .<= 1)  # Probabilities must be between 0 and 1
    @test all(isapprox.(sum(P_test, dims=2), 1.0, atol=1e-8))  # Row sums must be 1

    # Test loglikelihood function
    θ_test = vcat(vec(β_test), γ_test)
    ll_test = loglikelihood(θ_test, X, Z, y)
    @test isfinite(ll_test)  # Log-likelihood should be finite
end



#UNIT TEST FOR QUESTION 5
using Test

@testset "Question 5 Tests" begin
    # Example input data
    X = [1.0 0.5; 0.5 1.0; 1.0 1.5]  # Design matrix X
    Z = [0.2 0.3; 0.3 0.4; 0.1 0.2]  # Random effect matrix Z
    y = [1, 2, 1]                    # Choice outcomes

    # Parameters for the test
    num_draws = 100
    μ_γ = 0.0
    σ_γ = 1.0

    # Call the function
    result = mixed_logit_monte_carlo(X, Z, y; num_draws=num_draws, μ_γ=μ_γ, σ_γ=σ_γ)

    # Unit tests to validate the results
    @test !isnothing(result)
    @test haskey(result, :minimizer)
    @test haskey(result, :minimum)
    @test haskey(result, :converged)
    @test result[:converged] == true
    @test result[:minimum] < 0.0
end

