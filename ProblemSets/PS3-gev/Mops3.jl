#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::

using DataFrames
using CSV
using HTTP
using Optim
using LinearAlgebra

# Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS3-gev/nlsw88w.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Define X and Z matrices
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occupation

# The multinomial logit likelihood function
function multinomial_logit_likelihood(β, X, Z, y)
    n, J = size(Z)
    β_X = X * β[1:3]  # The first 3 elements of β correspond to X coefficients
    γ = β[4]  # The last element corresponds to γ for Z
    log_likelihood = 0.0
    
    for i in 1:n
        utilities = [β_X[i] + γ * (Z[i, j] - Z[i, 8]) for j in 1:J]  # Normalizing β_J = 0
        denom = sum(exp.(utilities))
        log_likelihood += utilities[y[i]] - log(denom)
    end
    return -log_likelihood  # Return negative log-likelihood for minimization
end

# Estimate the multinomial logit model using Optim
function estimate_multinomial_logit(X, Z, y)
    β_init = rand(4)  # Initial guess for β and γ
    result = optimize(β -> multinomial_logit_likelihood(β, X, Z, y), β_init, BFGS())
    return result.minimizer
end



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# After estimating the model
β_estimates = estimate_multinomial_logit(X, Z, y)
γ_estimate = β_estimates[4]
#  γ_estimate Represent the sensitivity of an individual's choice of occupation to the wage differences between alternative occupations.
# Interpretation
println("Estimated γ: $γ_estimate")
if γ_estimate > 0
    println("Wage differences positively influence the choice of occupation.")
elseif γ_estimate < 0
    println("Wage differences negatively influence the choice of occupation.")
else
    println("Wage differences do not influence the choice of occupation.")
end


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::


# Define the same nested logit utility function
function nested_logit_utility(params, data, nest_structure)
    β_WC, β_BC, λ_WC, λ_BC, γ = params  # Parameters
    utilities = zeros(size(data, 1))
    
    for (i, row) in enumerate(eachrow(data))
        if row[:choice] in nest_structure["WC"]
            utilities[i] = β_WC * row[:Professional_Technical] + γ * row[:Sales]
        elseif row[:choice] in nest_structure["BC"]
            utilities[i] = β_BC * row[:Clerical_Unskilled] + γ * row[:Craftsmen]
        else
            utilities[i] = 0  # Other occupations
        end
    end
    
    return utilities
end

# Define the log-likelihood function
function log_likelihood(params, data, nest_structure)
    utilities = nested_logit_utility(params, data, nest_structure)
    exp_utilities = exp.(utilities)
    
    logsum_WC = log(sum(exp_utilities[nest_structure["WC"]]))
    logsum_BC = log(sum(exp_utilities[nest_structure["BC"]]))
    logsum_other = log(sum(exp_utilities[nest_structure["Other"]]))
    
    total_logsum = log(exp(logsum_WC) + exp(logsum_BC) + exp(logsum_other))
    likelihood = sum(utilities) - total_logsum
    
    return -likelihood  # Negative because we are minimizing
end

# Optimization function using the Nelder-Mead algorithm
function estimate_nested_logit(data, nest_structure)
    initial_params = [1.0, 1.0, 1.0, 1.0, 1.0]
    result = optimize(params -> log_likelihood(params, data, nest_structure), initial_params, NelderMead())
    return result
end



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Import required packages
using DataFrames, CSV, HTTP, GLM, LinearAlgebra, Random, Statistics, FreqTables, Optim, Test

# Load Data Function
function load_data()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS3-gev/nlsw88w.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    return df
end

# Function to prepare the data for the model
function prepare_data(df)
    X = hcat(df.age, df.white, df.collgrad)
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occupation
    return X, Z, y
end

# Function to calculate choice probabilities
function choice_probs(β, X, Z, J)
    N = size(X, 1)
    XB = X * β[1:size(X,2)]
    ZG = [Z[:, j] * β[end] for j in 1:J-1]
    
    numerators = [exp.(XB + ZG[j]) for j in 1:J-1]
    push!(numerators, ones(N))  # For the base alternative
    
    denominators = sum(numerators)
    
    return [numerators[j] ./ denominators for j in 1:J]
end

# Log-likelihood function
function log_likelihood(β, X, Z, y, J)
    probs = choice_probs(β, X, Z, J)
    ll = 0.0
    for i in 1:length(y)
        ll += log(probs[y[i]][i])
    end
    return -ll  # Negative because we're minimizing
end

# Function to estimate the coefficients using BFGS optimization
function estimate_mnl(X, Z, y)
    initial_β = zeros(size(X, 2) + 1)
    J = 8  # Number of occupation categories
    result = optimize(β -> log_likelihood(β, X, Z, y, J), initial_β, BFGS())
    return Optim.minimizer(result)
end

# Function to wrap everything and print the results
function main()
    df = load_data()
    X, Z, y = prepare_data(df)
    
    # Estimate coefficients
    β_hat = estimate_mnl(X, Z, y)
    
    println("Estimated coefficients:")
    println("Age: ", β_hat[1])
    println("White: ", β_hat[2])
    println("College graduate: ", β_hat[3])
    println("Log wage (alternative-specific): ", β_hat[end])
    
    # Calculate and print choice probabilities
    J = 8
    probs = choice_probs(β_hat, X, Z, J)
    avg_probs = [mean(probs[j]) for j in 1:J]

    println("\nAverage choice probabilities:")
    for j in 1:J
        println("Occupation $j: ", round(avg_probs[j], digits=4))
    end
end

# Call the main function
main()



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# unit test for question 1
using Test 

# Unit tests
@testset "Multinomial Logit Tests" begin
    # Test if the likelihood function works with initial parameters
    @testset "Likelihood Function" begin
        β_init = rand(4)
        loglik = multinomial_logit_likelihood(β_init, X, Z, y)
        @test !isnan(loglik)  # Check if the likelihood is not NaN
    end

    # Test the optimization process
    @testset "Optimization" begin
        estimated_β = estimate_multinomial_logit(X, Z, y)
        @test length(estimated_β) == 4  # Check if the length of estimated β is correct
        @test !any(isnan.(estimated_β))  # Check if no element in the result is NaN
    end
end


# unit test for question 2


using Test, DataFrames, Optim, LinearAlgebra, Statistics

# Unit Test for Nested Logit Model
@testset "Nested Logit Tests" begin
    # Step 1: Define the nesting structure
    nests = Dict(
        "WC" => [1, 2, 3],  # White Collar: Professional/Technical, Managers/Administrators, Sales
        "BC" => [4, 5, 6, 7],  # Blue Collar: Clerical/Unskilled, Craftsmen, Operatives, Transport
        "Other" => [8]  # Other occupations
    )
    
    # Step 2: Create a sample dataset for testing (replace with actual data)
    df = DataFrame(
        choice = [1, 2, 3, 4, 5, 6, 7, 8],
        Professional_Technical = [1, 0, 0, 0, 0, 0, 0, 0],
        Managers_Administrators = [0, 1, 0, 0, 0, 0, 0, 0],
        Sales = [0, 0, 1, 0, 0, 0, 0, 0],
        Clerical_Unskilled = [0, 0, 0, 1, 0, 0, 0, 0],
        Craftsmen = [0, 0, 0, 0, 1, 0, 0, 0],
        Operatives = [0, 0, 0, 0, 0, 1, 0, 0],
        Transport = [0, 0, 0, 0, 0, 0, 1, 0],
        Other = [0, 0, 0, 0, 0, 0, 0, 1]
    )
    
    # Step 3: Test the utility function
    @testset "Utility Function" begin
        params = [1.0, 1.0, 1.0, 1.0, 1.0]
        utilities = nested_logit_utility(params, df, nests)
        @test !any(isnan, utilities)  # Check if none of the utilities are NaN
        @test length(utilities) == size(df, 1)  # Ensure the length matches the number of rows in the data
    end
    
    # Step 4: Test the log-likelihood function
    @testset "Log-Likelihood Function" begin
        params = [1.0, 1.0, 1.0, 1.0, 1.0]
        ll = log_likelihood(params, df, nests)
        @test !isnan(ll)  # Ensure log-likelihood is a number
    end
    
    # Step 5: Test the optimization process
    @testset "Optimization" begin
        result = estimate_nested_logit(df, nests)
        @test length(result.minimizer) == 5  # Check if we have 5 parameters
        @test result.minimum < 0  # The log-likelihood should be negative
    end
end
