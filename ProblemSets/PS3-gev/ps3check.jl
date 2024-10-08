# Import necessary packages
using DataFrames
using CSV
using HTTP
using LinearAlgebra
using Random
using Optim

# Load the data from the URL
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS3-gev/nlsw88w.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Define the variables X and Z
X = [df.age df.white df.collgrad]  # Individual characteristics (age, white, college grad)
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)  # Wage levels across occupations
y = df.occupation  # Choice of occupation

# Define the number of alternatives and parameters
J = 8  # Number of alternatives (occupations)
n = size(X, 1)  # Number of observations
k = size(X, 2)  # Number of covariates (age, white, collgrad)

# Define the log-likelihood function for the multinomial logit model
function log_likelihood(βγ::Vector)
    β = reshape(βγ[1:k*(J-1)], k, J-1)
    γ = βγ[k*(J-1)+1:end]
    ll = 0.0  # Log-likelihood
    for i in 1:n
        # Compute the utility for each alternative j (dot product between X[i, :] and β[:, j])
        utils = [dot(X[i, :], β[:, j]) + γ[j] * (Z[i, j] - Z[i, J]) for j in 1:(J-1)]
        utils = vcat(utils, 0.0)  # Normalize β_J = 0 for the Jth alternative
        # Compute the log-likelihood contribution for observation i
        denom = 1 + sum(exp.(utils[1:(J-1)]))
        ll += utils[y[i]] - log(denom)
    end
    return -ll  # Minimize the negative log-likelihood
end

# Initialize parameter estimates
β0 = 0.1 * randn(k*(J-1))  # Initial values for β
γ0 = 0.1 * randn(J-1)  # Initial values for γ
βγ0 = vcat(β0, γ0)

# Estimate the model using the Optim package
result = optimize(log_likelihood, βγ0, BFGS())

# Extract the estimated parameters
βγ_hat = Optim.minimizer(result)
β_hat = reshape(βγ_hat[1:k*(J-1)], k, J-1)
γ_hat = βγ_hat[k*(J-1)+1:end]

# Print the results
println("Estimated β coefficients (for X):")
println(β_hat)
println("Estimated γ coefficients (for Z):")
println(γ_hat)

# Utility function for predictions (if needed)
function predict(X_new, Z_new, β_hat, γ_hat)
    utilities = [dot(X_new, β_hat[:, j]) + γ_hat[j] * (Z_new[j] - Z_new[J]) for j in 1:(J-1)]
    utilities = vcat(utilities, 0.0)  # Normalizing for the last alternative
    probs = exp.(utilities) ./ (1 + sum(exp.(utilities[1:(J-1)])))
    return probs
end
