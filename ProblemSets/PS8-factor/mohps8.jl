#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# Load required packages
using DataFrames
using Random
using GLM
using Statistics

# Set random seed for reproducibility
Random.seed!(123)

# Generate synthetic data
n = 1000  # number of observations

# Generate predictors
data = DataFrame(
    black = rand([0, 1], n),
    hispanic = rand([0, 1], n),
    female = rand([0, 1], n),
    school = rand(12:18, n),  # years of schooling between 12 and 18
    gradHS = rand([0, 1], n),
    grad4yr = rand([0, 1], n)
)

# Generate log wage with some reasonable coefficients
# Adding some noise to make it realistic
β = [2.5, -0.2, -0.1, -0.15, 0.1, 0.2, 0.3]  # true coefficients
X = hcat(ones(n), Matrix(data))
ε = randn(n) * 0.5  # random noise
data.wage = exp.(X * β + ε)
data.logwage = log.(data.wage)

# Estimate the linear regression model
model1 = @formula(logwage ~ black + hispanic + female + school + gradHS + grad4yr)
reg1 = lm(model1, data)

# Print regression results
println("Regression Results:")
println("==================")
println(reg1)

# Calculate and print R-squared
r2 = r²(reg1)
println("\nR-squared: ", round(r2, digits=4))

# Get coefficient table
coef_table = coeftable(reg1)
println("\nDetailed Coefficient Table:")
println("=========================")
println(coef_table)



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::


# Load required packages
using DataFrames
using Random
using Statistics
using LinearAlgebra
using GLM

# Set random seed for reproducibility
Random.seed!(123)

# Number of observations
n = 1000

# Generate base data
data = DataFrame(
    black = rand([0, 1], n),
    hispanic = rand([0, 1], n),
    female = rand([0, 1], n),
    school = rand(12:18, n),
    gradHS = rand([0, 1], n),
    grad4yr = rand([0, 1], n)
)

# Generate correlated ASVAB scores
# First create a base ability score that will induce correlation
base_ability = randn(n)

# Generate wage first (as in question 1)
β = [2.5, -0.2, -0.1, -0.15, 0.1, 0.2, 0.3]
X = hcat(ones(n), Matrix(data[:, [:black, :hispanic, :female, :school, :gradHS, :grad4yr]]))
ε = randn(n) * 0.5
data.wage = exp.(X * β + ε)
data.logwage = log.(data.wage)

# Now add ASVAB scores with correlation structure
# We'll simulate 6 ASVAB scores that are correlated through base_ability
data.asvab_arithmetic = 0.7 * base_ability + 0.3 * randn(n)
data.asvab_word = 0.8 * base_ability + 0.2 * randn(n)
data.asvab_paragraph = 0.75 * base_ability + 0.25 * randn(n)
data.asvab_math = 0.85 * base_ability + 0.15 * randn(n)
data.asvab_numerical = 0.6 * base_ability + 0.4 * randn(n)
data.asvab_coding = 0.7 * base_ability + 0.3 * randn(n)

# Define ASVAB variable names
asvab_vars = [:asvab_arithmetic, :asvab_word, :asvab_paragraph, 
              :asvab_math, :asvab_numerical, :asvab_coding]

# Compute correlation matrix
asvab_matrix = Matrix(data[:, asvab_vars])
correlation_matrix = cor(asvab_matrix)

# Print correlation matrix with variable names
println("\nCorrelation Matrix of ASVAB Variables:")
println("====================================")

# Print header with shortened names
short_names = ["arith", "word", "para", "math", "num", "code"]
print("      ")
for name in short_names
    print(rpad(name, 8))
end
println()

# Print correlation matrix with rounded values
for (i, row) in enumerate(eachrow(correlation_matrix))
    print(rpad(short_names[i], 6))
    for val in row
        print(rpad(round(val, digits=3), 8))
    end
    println()
end

# Run regression from question 1 again
model1 = @formula(logwage ~ black + hispanic + female + school + gradHS + grad4yr)
reg1 = lm(model1, data)

println("\nRegression Results from Question 1:")
println("=================================")
println(reg1)



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# Load packages
using DataFrames, GLM, Statistics, LinearAlgebra

# Estimate the expanded regression model including ASVAB variables
model2 = @formula(logwage ~ black + hispanic + female + school + gradHS + grad4yr + 
                 asvab_arithmetic + asvab_word + asvab_paragraph + 
                 asvab_math + asvab_numerical + asvab_coding)
reg2 = lm(model2, data)

# Print regression results
println("\nExpanded Regression Results (Including ASVAB):")
println("============================================")
println(reg2)

# Calculate and print R-squared
r2 = r²(reg2)
println("\nR-squared: ", round(r2, digits=4))

# Calculate Variance Inflation Factors (VIF) for ASVAB variables
function calculate_vif(data, variable, other_vars)
    model_formula = Term(variable) ~ sum(Term.(other_vars))
    model = lm(model_formula, data)
    return 1 / (1 - r²(model))
end

# Get all predictor variables
predictors = [:black, :hispanic, :female, :school, :gradHS, :grad4yr,
             :asvab_arithmetic, :asvab_word, :asvab_paragraph, 
             :asvab_math, :asvab_numerical, :asvab_coding]

# Calculate VIF for each variable
println("\nVariance Inflation Factors:")
println("=========================")
for var in predictors
    other_vars = filter(x -> x != var, predictors)
    vif = calculate_vif(data, var, other_vars)
    println(rpad(string(var), 20), round(vif, digits=2))
end



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# Get the ASVAB variables into a matrix format and transpose to get J×N matrix
asvab_vars = [:asvab_arithmetic, :asvab_word, :asvab_paragraph, 
              :asvab_math, :asvab_numerical, :asvab_coding]
asvabMat = Matrix(data[:, asvab_vars])'

# Standardize the ASVAB variables (important for PCA)
asvabMat_std = (asvabMat .- mean(asvabMat, dims=2)) ./ std(asvabMat, dims=2)

# Fit PCA model with the first principal component
using MultivariateStats
M = fit(PCA, asvabMat_std; maxoutdim=1)

# Get the first principal component scores
asvabPCA = MultivariateStats.transform(M, asvabMat_std)

# Convert PCA scores from 1×N to N×1 array
pca_scores = vec(asvabPCA')

# Add PCA scores to the original dataset
data.asvab_pc1 = pca_scores

# Run regression with the first principal component
model_pca = @formula(logwage ~ black + hispanic + female + school + gradHS + grad4yr + asvab_pc1)
reg_pca = lm(model_pca, data)

# Print results
println("\nRegression Results with First Principal Component:")
println("===============================================")
println(reg_pca)

# Print R-squared
r2_pca = r²(reg_pca)
println("\nR-squared: ", round(r2_pca, digits=4))

# Show proportion of variance explained by first PC
explained_var = principalratio(M)
println("\nProportion of variance explained by first PC: ", round(explained_var[1], digits=4))

# Show the loadings (weights) for each ASVAB test in the first PC
println("\nPrincipal Component Loadings:")
println("===========================")
loadings = projection(M)
for (var, loading) in zip(asvab_vars, loadings)
    println(rpad(string(var), 20), round(loading, digits=4))
end



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::


# Load necessary packages
using DataFrames, GLM, Statistics, LinearAlgebra, MultivariateStats

# Get the ASVAB variables into a matrix format and transpose to get J×N matrix
asvab_vars = [:asvab_arithmetic, :asvab_word, :asvab_paragraph, 
              :asvab_math, :asvab_numerical, :asvab_coding]
asvabMat = Matrix(data[:, asvab_vars])'

# Standardize the ASVAB variables (important for Factor Analysis)
asvabMat_std = (asvabMat .- mean(asvabMat, dims=2)) ./ std(asvabMat, dims=2)

# Fit Factor Analysis model with one factor
F = fit(FactorAnalysis, asvabMat_std; maxoutdim=1)

# Get the factor scores
asvabFactor = MultivariateStats.transform(F, asvabMat_std)

# Convert factor scores from 1×N to N×1 array
factor_scores = vec(asvabFactor')

# Add factor scores to the original dataset
data.asvab_factor1 = factor_scores

# Run regression with the first factor
model_factor = @formula(logwage ~ black + hispanic + female + school + gradHS + grad4yr + asvab_factor1)
reg_factor = lm(model_factor, data)

# Print results
println("\nRegression Results with First Factor Analysis Component:")
println("===============================================")
println(reg_factor)

# Print R-squared
r2_factor = r²(reg_factor)
println("\nR-squared: ", round(r2_factor, digits=4))


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 6
#:::::::::::::::::::::::::::::::::::::::::::::::::::

using Optim, Distributions, QuadGK, Random, LinearAlgebra, Statistics

# Define the standard normal PDF
normal_pdf(x) = pdf(Normal(0, 1), x)

# Gauss-Legendre quadrature function
function gauss_legendre_quadrature(func, a, b, n)
    return quadgk(func, a, b, rtol=1e-5)[1]
end

# Define the likelihood function for one observation, given a value of ξ
function individual_likelihood(xi::Float64, params::Vector{Float64}, obs::Dict{Symbol, Union{Vector{Float64}, Float64}})
    # Unpack parameters
    α = params[1:6]           # Adjusted to match covariates length
    γ = params[7:12]           # Factor loadings
    σ_asvab = params[13:18]    # Standard deviations for ASVAB
    β = params[19:24]          # Adjusted to match covariates length
    δ = params[25]             # Loading for latent factor ξ
    σ_wage = params[26]        # Wage model standard deviation
    
    # Measurement model likelihood for ASVAB scores
    asvab_likelihood = 1.0
    for j in 1:length(obs[:asvab_scores])
        mean_asvab = α[j] + γ[j] * xi + dot(obs[:covariates], α) 
        asvab_likelihood *= max(normal_pdf((obs[:asvab_scores][j] - mean_asvab) / σ_asvab[j]) / σ_asvab[j], 1e-10)
    end
    
    # Wage model likelihood
    mean_wage = dot(obs[:covariates], β) + δ * xi
    wage_likelihood = max(normal_pdf((obs[:logwage] - mean_wage) / σ_wage) / σ_wage, 1e-10)

    # Combined likelihood for this observation
    return asvab_likelihood * wage_likelihood
end

# Log-likelihood function integrated over ξ
function log_likelihood(data::Vector{Dict{Symbol, Union{Vector{Float64}, Float64}}}, params::Vector{Float64})::Float64
    total_log_likelihood = 0.0
    for obs in data
        integrand(xi) = individual_likelihood(xi, params, obs) * normal_pdf(xi)
        integrated_likelihood = gauss_legendre_quadrature(integrand, -Inf, Inf, 20)
        
        # Avoid log of non-positive values by using max with a small epsilon value
        total_log_likelihood += log(max(integrated_likelihood, 1e-10))
    end
    return -total_log_likelihood  # Negative log-likelihood for optimization
end

# Define sample data with enforced typing
sample_data = [
    Dict{Symbol, Union{Float64, Vector{Float64}}}(
        :asvab_scores => Vector{Float64}([0.5, -0.3, 0.7, -0.2, 1.2, -0.6]),  # Vector{Float64} for ASVAB scores
        :covariates => Vector{Float64}([1.0, 0.0, 1.0, 16.0, 1.0, 0.0]),     # Vector{Float64} for covariates
        :logwage => 2.3                                                        # Float64 for logwage
    )
]

# Initial guess for parameters, with consistent lengths
initial_params = vcat(
    Float64.(fill(0.1, 6)),   # α (6 elements for covariates)
    Float64.(fill(0.1, 6)),   # γ (factor loadings for 6 ASVAB scores)
    Float64.(fill(1.0, 6)),   # σ_asvab (standard deviations for ASVAB scores)
    Float64.(fill(0.1, 6)),   # β (6 elements to match covariates)
    0.1,                      # δ (loading for latent factor)
    1.0                       # σ_wage (wage model standard deviation)
)

# Ensure `initial_params` is a concrete Vector{Float64}
initial_params = Vector{Float64}(initial_params)

# Run the optimizer
result = optimize(params -> log_likelihood(sample_data, params), initial_params, NelderMead())

# Display results
println("Optimization results:")
println(result)


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 7
#:::::::::::::::::::::::::::::::::::::::::::::::::::


using Test
using Distributions

# Test for `normal_pdf` function
@testset "Test normal_pdf function" begin
    @test normal_pdf(0) ≈ pdf(Normal(0, 1), 0) # Standard normal PDF at 0 should match `Normal` distribution result
    @test normal_pdf(1) ≈ pdf(Normal(0, 1), 1) # Standard normal PDF at 1
    @test normal_pdf(-1) ≈ pdf(Normal(0, 1), -1) # Standard normal PDF at -1
end

# Test for `gauss_legendre_quadrature` function
@testset "Test gauss_legendre_quadrature function" begin
    integrand(x) = exp(-x^2 / 2) / sqrt(2 * π) # Standard normal PDF as integrand
    result = gauss_legendre_quadrature(integrand, -Inf, Inf, 20)
    @test result ≈ 1.0 atol=1e-5 # Should integrate to 1 over entire range

    # Test with a different function
    integrand2(x) = x^2
    result2 = gauss_legendre_quadrature(integrand2, 0, 1, 20)
    @test result2 ≈ 1/3 atol=1e-5 # Integral of x^2 from 0 to 1 should be 1/3
end

# Test for `individual_likelihood` function
@testset "Test individual_likelihood function" begin
    # Define mock parameters and data for testing
    params = vcat(
        Float64.(fill(0.1, 6)),   # α (6 elements for covariates)
        Float64.(fill(0.1, 6)),   # γ (factor loadings for 6 ASVAB scores)
        Float64.(fill(1.0, 6)),   # σ_asvab (standard deviations for ASVAB scores)
        Float64.(fill(0.1, 6)),   # β (6 elements to match covariates)
        0.1,                      # δ (loading for latent factor)
        1.0                       # σ_wage (wage model standard deviation)
    )
    obs = Dict(
        :asvab_scores => Vector{Float64}([0.5, -0.3, 0.7, -0.2, 1.2, -0.6]),
        :covariates => Vector{Float64}([1.0, 0.0, 1.0, 16.0, 1.0, 0.0]),
        :logwage => 2.3
    )
    xi = 0.5

    likelihood = individual_likelihood(xi, params, obs)
    @test likelihood > 0 # Likelihood should be positive
    @test likelihood < 1 # Likelihood for a single observation should be less than 1 in this context
end

# Test for `log_likelihood` function
@testset "Test log_likelihood function" begin
    # Define mock data
    sample_data = [
        Dict(
            :asvab_scores => Vector{Float64}([0.5, -0.3, 0.7, -0.2, 1.2, -0.6]),
            :covariates => Vector{Float64}([1.0, 0.0, 1.0, 16.0, 1.0, 0.0]),
            :logwage => 2.3
        )
    ]

    # Test with initial parameters
    log_likelihood_value = log_likelihood(sample_data, params)
    @test log_likelihood_value < 0 # Log-likelihood should be negative as it's the sum of log probabilities

    # Test for non-positive likelihood handling
    bad_sample_data = [
        Dict(
            :asvab_scores => Vector{Float64}([-100.0, -100.0, -100.0, -100.0, -100.0, -100.0]),
            :covariates => Vector{Float64}([1.0, 0.0, 1.0, 16.0, 1.0, 0.0]),
            :logwage => -100.0
        )
    ]
    bad_log_likelihood_value = log_likelihood(bad_sample_data, params)
    @test bad_log_likelihood_value < 0 # Should handle negative or extreme values gracefully
end
