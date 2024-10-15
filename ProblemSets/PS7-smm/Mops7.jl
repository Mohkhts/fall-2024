#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::

using Optim, LinearAlgebra, HTTP, CSV, DataFrames, GLM

# Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [ones(size(df,1)) df.age df.race.==1 df.collgrad.==1]
y = Float64.(df.married.==1)

# GMM estimation function
function gmm_linear_regression(y, X)
    N, K = size(X)
    
    # Moment function
    function g(β)
        ε = y - X * β
        return X' * ε / N
    end
    
    # Objective function (using Identity matrix as weighting matrix)
    function Q(β)
        moment = g(β)
        return moment' * moment
    end
    
    # Optimization
    result = optimize(Q, zeros(K), BFGS(), Optim.Options(g_tol=1e-6, iterations=100_000))
    
    return Optim.minimizer(result)
end

# Run GMM estimation
β_gmm = gmm_linear_regression(y, X)
println("GMM estimates:")
println(β_gmm)

# Compare with OLS estimates
β_ols = inv(X'X) * X'y
println("\nOLS estimates:")
println(β_ols)

# Compare with GLM
df.white = df.race.==1
model_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
println("\nGLM estimates:")
println(coef(model_glm))


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# PART A:
using CSV, HTTP, DataFrames, LinearAlgebra, Optim, Random, FreqTables

# Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Data preparation
df = dropmissing(df, :occupation)
for i in 8:13
    df[df.occupation .== i, :occupation] .= 7
end

# Check occupation distribution
println(freqtable(df, :occupation))

# Prepare X and y
X = [ones(size(df,1)) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

# Multinomial logit function
function mlogit(alpha, X, y)
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N, J)
    for j in 1:J
        bigY[:, j] = y .== j
    end
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
    
    num = zeros(N, J)
    dem = zeros(N)
    for j in 1:J
        num[:, j] = exp.(X * bigAlpha[:, j])
        dem .+= num[:, j]
    end
    
    P = num ./ repeat(dem, 1, J)
    
    loglike = -sum(bigY .* log.(P))
    
    return loglike
end

# Optimization
K = size(X, 2)
J = length(unique(y))
alpha_init = zeros((J-1) * K)
result = optimize(a -> mlogit(a, X, y), alpha_init, BFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))

# Extract MLE estimates
alpha_hat_mle = result.minimizer

# Reshape and print results
beta_hat_mle = reshape(alpha_hat_mle, K, J-1)
println("MLE estimates:")
println(beta_hat_mle)

# Calculate standard errors
function mlogit_for_h(alpha, X, y)
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N, J)
    for j in 1:J
        bigY[:, j] = y .== j
    end
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
    
    T = promote_type(eltype(X), eltype(alpha))
    num = zeros(T, N, J)
    dem = zeros(T, N)
    for j in 1:J
        num[:, j] = exp.(X * bigAlpha[:, j])
        dem .+= num[:, j]
    end
    
    P = num ./ repeat(dem, 1, J)
    
    loglike = -sum(bigY .* log.(P))
    
    return loglike
end

td = TwiceDifferentiable(b -> mlogit_for_h(b, X, y), alpha_init; autodiff = :forward)
H = Optim.hessian!(td, alpha_hat_mle)
se = sqrt.(diag(inv(H)))

println("\nMLE estimates with standard errors:")
println([reshape(alpha_hat_mle, K, J-1) reshape(se, K, J-1)])


# PART B:

using CSV, HTTP, DataFrames, LinearAlgebra, Optim, Random, FreqTables

# Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Data preparation
df = dropmissing(df, :occupation)
for i in 8:13
    df[df.occupation .== i, :occupation] .= 7
end

# Prepare X and y
X = [ones(size(df,1)) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

# Multinomial logit function for MLE
function mlogit(alpha, X, y)
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N, J)
    for j in 1:J
        bigY[:, j] = y .== j
    end
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
    
    num = zeros(N, J)
    dem = zeros(N)
    for j in 1:J
        num[:, j] = exp.(X * bigAlpha[:, j])
        dem .+= num[:, j]
    end
    
    P = num ./ repeat(dem, 1, J)
    
    loglike = -sum(bigY .* log.(P))
    
    return loglike
end

# MLE Optimization
K = size(X, 2)
J = length(unique(y))
alpha_init = zeros((J-1) * K)
mle_result = optimize(a -> mlogit(a, X, y), alpha_init, BFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000))

# Extract MLE estimates
alpha_hat_mle = mle_result.minimizer

# GMM functions
function calculate_P(alpha, X)
    K = size(X, 2)
    J = Int(length(alpha) / K) + 1
    N = size(X, 1)
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
    
    num = zeros(N, J)
    dem = zeros(N)
    for j in 1:J
        num[:, j] = exp.(X * bigAlpha[:, j])
        dem .+= num[:, j]
    end
    
    return num ./ repeat(dem, 1, J)
end

function g_function(alpha, X, y)
    N = size(X, 1)
    J = length(unique(y))
    
    d = zeros(N, J)
    for j in 1:J
        d[:, j] = y .== j
    end
    
    P = calculate_P(alpha, X)
    
    return vec(d - P)
end

function gmm_objective(alpha, X, y, W)
    moment = g_function(alpha, X, y)
    return moment' * W * moment
end

# GMM Estimation
W = I  # Identity matrix as initial weighting matrix
gmm_result = optimize(a -> gmm_objective(a, X, y, W), alpha_hat_mle, BFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000))

# Extract GMM estimates
alpha_hat_gmm = gmm_result.minimizer

# Reshape and print results
beta_hat_gmm = reshape(alpha_hat_gmm, K, J-1)
println("GMM estimates:")
println(beta_hat_gmm)

# Compare with MLE estimates
beta_hat_mle = reshape(alpha_hat_mle, K, J-1)
println("\nMLE estimates:")
println(beta_hat_mle)


# PART c:

using CSV, HTTP, DataFrames, LinearAlgebra, Optim, Random, FreqTables

# Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Data preparation
df = dropmissing(df, :occupation)
for i in 8:13
    df[df.occupation .== i, :occupation] .= 7
end

# Prepare X and y
X = [ones(size(df,1)) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

# GMM functions
function calculate_P(alpha, X)
    K = size(X, 2)
    J = Int(length(alpha) / K) + 1
    N = size(X, 1)
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
    
    num = zeros(N, J)
    dem = zeros(N)
    for j in 1:J
        num[:, j] = exp.(X * bigAlpha[:, j])
        dem .+= num[:, j]
    end
    
    return num ./ repeat(dem, 1, J)
end

function g_function(alpha, X, y)
    N = size(X, 1)
    J = length(unique(y))
    
    d = zeros(N, J)
    for j in 1:J
        d[:, j] = y .== j
    end
    
    P = calculate_P(alpha, X)
    
    return vec(d - P)
end

function gmm_objective(alpha, X, y, W)
    moment = g_function(alpha, X, y)
    return moment' * W * moment
end

# GMM Estimation with random starting values
K = size(X, 2)
J = length(unique(y))
W = I  # Identity matrix as weighting matrix

# Set random seed for reproducibility
Random.seed!(123)

# Generate random starting values
alpha_init_random = randn((J-1) * K)

# Perform GMM estimation
gmm_result_random = optimize(a -> gmm_objective(a, X, y, W), alpha_init_random, BFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000))

# Extract GMM estimates
alpha_hat_gmm_random = gmm_result_random.minimizer

# Reshape and print results
beta_hat_gmm_random = reshape(alpha_hat_gmm_random, K, J-1)
println("GMM estimates with random starting values:")
println(beta_hat_gmm_random)

# Compare convergence
println("\nConvergence with random starting values:")
println("Converged: ", Optim.converged(gmm_result_random))
println("Iterations: ", Optim.iterations(gmm_result_random))
println("Minimum function value: ", Optim.minimum(gmm_result_random))

# Optional: Run multiple times with different random starting values
n_runs = 5
results = []

for i in 1:n_runs
    alpha_init = randn((J-1) * K)
    result = optimize(a -> gmm_objective(a, X, y, W), alpha_init, BFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000))
    push!(results, (minimizer=result.minimizer, minimum=Optim.minimum(result), converged=Optim.converged(result)))
end

# Print results of multiple runs
println("\nResults from $n_runs runs with different random starting values:")
for (i, result) in enumerate(results)
    println("Run $i:")
    println("  Converged: ", result.converged)
    println("  Minimum function value: ", result.minimum)
end

# Find the best result
best_result = argmin(result -> result.minimum, results)
println("\nBest result (lowest minimum):")
println("Minimum function value: ", best_result.minimum)
beta_hat_best = reshape(best_result.minimizer, K, J-1)
println("Corresponding estimates:")
println(beta_hat_best)


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# Part a:

using Random, Statistics, LinearAlgebra

# Set random seed for reproducibility
Random.seed!(123)

# Function to generate X matrix
function generate_X(N, K)
    return randn(N, K)
end

# Set parameters
N = 1000  # Number of observations
J = 4     # Number of choices (J > 2)
K = 3     # Number of covariates (K > 1)

# Generate X matrix
X = generate_X(N, K)

# Display first few rows of X
println("First 5 rows of X:")
display(X[1:5, :])

# Summary statistics
println("\nSummary statistics of X:")
for k in 1:K
    println("Variable $k:")
    println("  Mean: ", mean(X[:, k]))
    println("  Std Dev: ", std(X[:, k]))
    println("  Min: ", minimum(X[:, k]))
    println("  Max: ", maximum(X[:, k]))
end

# Function to check correlation between variables
function check_correlation(X)
    K = size(X, 2)
    cor_matrix = cor(X)
    println("\nCorrelation matrix:")
    display(cor_matrix)
end

check_correlation(X)


# part b:

using Random, LinearAlgebra

# Set random seed for reproducibility
Random.seed!(123)

# Function to generate X matrix (from part a)
function generate_X(N, K)
    return randn(N, K)
end

# Set parameters
N = 1000  # Number of observations
J = 4     # Number of choices (J > 2)
K = 3     # Number of covariates (K > 1)

# Generate X matrix
X = generate_X(N, K)

# Function to generate beta parameters
function generate_beta(K, J)
    # We generate J-1 sets of parameters, as one choice is the base category
    return randn(K, J-1)
end

# Generate beta
β = generate_beta(K, J)

println("Generated β parameters:")
display(β)

# Function to check if β satisfies conformability with X and J
function check_conformability(β, X, J)
    K_X, K_β = size(X, 2), size(β, 1)
    J_β = size(β, 2) + 1  # Add 1 because one choice is the base category
    
    if K_X == K_β && J == J_β
        println("\nConformability check passed:")
        println("  Number of covariates in X: $K_X")
        println("  Number of rows in β: $K_β")
        println("  Number of choices J: $J")
        println("  Number of columns in β (+1): $J_β")
    else
        println("\nConformability check failed:")
        println("  Number of covariates in X: $K_X")
        println("  Number of rows in β: $K_β")
        println("  Number of choices J: $J")
        println("  Number of columns in β (+1): $J_β")
    end
end

check_conformability(β, X, J)

# Optional: Calculate and display the condition number of β
cond_num = cond(β)
println("\nCondition number of β: $cond_num")
if cond_num > 1000
    println("Warning: High condition number. β might be ill-conditioned.")
else
    println("β appears to be well-conditioned.")
end


# part c:
using Random, LinearAlgebra, Statistics

# Set random seed for reproducibility
Random.seed!(123)

# Function to generate X matrix (from part a)
function generate_X(N, K)
    return randn(N, K)
end

# Function to generate beta parameters (from part b)
function generate_beta(K, J)
    return randn(K, J-1)
end

# Set parameters
N = 1000  # Number of observations
J = 4     # Number of choices (J > 2)
K = 3     # Number of covariates (K > 1)

# Generate X matrix and beta
X = generate_X(N, K)
β = generate_beta(K, J)

# Function to calculate choice probabilities
function calculate_probabilities(X, β)
    N, K = size(X)
    J = size(β, 2) + 1  # Add 1 for the base category

    # Calculate utilities
    U = zeros(N, J)
    for j in 1:J-1
        U[:, j] = X * β[:, j]
    end
    # U[:, J] remains 0 as it's the base category

    # Calculate probabilities
    P = exp.(U) ./ sum(exp.(U), dims=2)
    
    return P
end

# Calculate probabilities
P = calculate_probabilities(X, β)

println("First 5 rows of choice probabilities P:")
display(P[1:5, :])

# Check that probabilities sum to 1 for each observation
sum_check = all(isapprox.(sum(P, dims=2), 1, atol=1e-6))
println("\nAll rows sum to 1 (within numerical precision): $sum_check")

# Calculate and display summary statistics of probabilities
println("\nSummary statistics of probabilities:")
for j in 1:J
    p_j = P[:, j]
    println("Choice $j:")
    println("  Mean: ", mean(p_j))
    println("  Min: ", minimum(p_j))
    println("  Max: ", maximum(p_j))
end

# Check for any very small probabilities
min_prob = minimum(P)
if min_prob < 1e-6
    println("\nWarning: Some very small probabilities detected. Minimum probability: $min_prob")
else
    println("\nAll probabilities are reasonably large. Minimum probability: $min_prob")
end

# part d:

using Random, LinearAlgebra, Statistics, Distributions

# Set random seed for reproducibility
Random.seed!(123)

# Function to generate X matrix (from part a)
function generate_X(N, K)
    return randn(N, K)
end

# Function to generate beta parameters (from part b)
function generate_beta(K, J)
    return randn(K, J-1)
end

# Function to calculate choice probabilities (from part c)
function calculate_probabilities(X, β)
    N, K = size(X)
    J = size(β, 2) + 1  # Add 1 for the base category

    # Calculate utilities
    U = zeros(N, J)
    for j in 1:J-1
        U[:, j] = X * β[:, j]
    end
    # U[:, J] remains 0 as it's the base category

    # Calculate probabilities
    P = exp.(U) ./ sum(exp.(U), dims=2)
    
    return P
end

# Set parameters
N = 1000  # Number of observations
J = 4     # Number of choices (J > 2)
K = 3     # Number of covariates (K > 1)

# Generate X matrix and beta
X = generate_X(N, K)
β = generate_beta(K, J)

# Calculate probabilities
P = calculate_probabilities(X, β)

# Function to generate choices
function generate_choices(P)
    N, J = size(P)
    Y = zeros(Int, N)
    for i in 1:N
        Y[i] = rand(Categorical(P[i, :]))
    end
    return Y
end

# Generate choices
Y = generate_choices(P)

println("First 20 generated choices:")
println(Y[1:20])

# Calculate and display choice frequencies
choice_freq = [sum(Y .== j) for j in 1:J]
choice_prop = choice_freq ./ N

println("\nChoice frequencies:")
for j in 1:J
    println("Choice $j: $(choice_freq[j]) ($(round(choice_prop[j]*100, digits=2))%)")
end

# Compare observed proportions with average predicted probabilities
println("\nComparison of observed proportions vs average predicted probabilities:")
avg_probs = mean(P, dims=1)[1, :]
for j in 1:J
    println("Choice $j: Observed $(round(choice_prop[j], digits=4)), Predicted $(round(avg_probs[j], digits=4))")
end

# Chi-square goodness of fit test
expected_freq = N .* avg_probs
chi_square_stat = sum((choice_freq .- expected_freq).^2 ./ expected_freq)
dof = J - 1
p_value = 1 - cdf(Chisq(dof), chi_square_stat)

println("\nChi-square goodness of fit test:")
println("Chi-square statistic: $(round(chi_square_stat, digits=4))")
println("Degrees of freedom: $dof")
println("p-value: $(round(p_value, digits=4))")


# part e:

using Random, LinearAlgebra, Statistics, Distributions

function simulate_multinomial_logit(N::Int, J::Int, K::Int)
    # Generate X matrix
    X = randn(N, K)
    
    # Generate beta parameters
    β = randn(K, J-1)
    
    # Calculate probabilities
    function calculate_probabilities(X, β)
        N, K = size(X)
        J = size(β, 2) + 1
        U = zeros(N, J)
        for j in 1:J-1
            U[:, j] = X * β[:, j]
        end
        P = exp.(U) ./ sum(exp.(U), dims=2)
        return P
    end
    
    P = calculate_probabilities(X, β)
    
    # Generate choices
    Y = [rand(Categorical(P[i, :])) for i in 1:N]
    
    # Summary statistics
    choice_freq = [count(==(j), Y) for j in 1:J]
    choice_prop = choice_freq ./ N
    avg_probs = mean(P, dims=1)[1, :]
    
    # Chi-square goodness of fit test
    expected_freq = N .* avg_probs
    chi_square_stat = sum((choice_freq .- expected_freq).^2 ./ expected_freq)
    dof = J - 1
    p_value = 1 - cdf(Chisq(dof), chi_square_stat)
    
    return (
        X = X,
        Y = Y,
        β = β,
        P = P,
        choice_proportions = choice_prop,
        avg_predicted_probs = avg_probs,
        chi_square_stat = chi_square_stat,
        p_value = p_value
    )
end

# Run the simulation
N, J, K = 1000, 4, 3
Random.seed!(123)
result = simulate_multinomial_logit(N, J, K)

# Display results
println("True β:")
display(result.β)

println("\nFirst 20 generated choices:")
println(result.Y[1:20])

println("\nChoice proportions:")
for j in 1:J
    println("Choice $j: $(round(result.choice_proportions[j], digits=4))")
end

println("\nAverage predicted probabilities:")
for j in 1:J
    println("Choice $j: $(round(result.avg_predicted_probs[j], digits=4))")
end

println("\nChi-square goodness of fit test:")
println("Chi-square statistic: $(round(result.chi_square_stat, digits=4))")
println("p-value: $(round(result.p_value, digits=4))")

# part f:

using Random, LinearAlgebra, Statistics, Distributions, Optim, ForwardDiff

function simulate_multinomial_logit(N::Int, J::Int, K::Int)
    X = randn(N, K)
    β = randn(K, J-1)
    
    function calculate_probabilities(X, β)
        N, K = size(X)
        J = size(β, 2) + 1
        U = X * β
        U = [U zeros(N)]
        P = exp.(U) ./ sum(exp.(U), dims=2)
        return P
    end
    
    P = calculate_probabilities(X, β)
    Y = [rand(Categorical(P[i, :])) for i in 1:N]
    
    return X, Y, β
end

function estimate_multinomial_logit(X, Y)
    N, K = size(X)
    J = length(unique(Y))
    
    function calculate_probabilities(X, β)
        N, K = size(X)
        J = size(β, 2) + 1
        U = X * β
        U = [U zeros(eltype(U), N)]
        P = exp.(U .- maximum(U, dims=2)) ./ sum(exp.(U .- maximum(U, dims=2)), dims=2)
        return P
    end
    
    function loglikelihood(β_flat)
        β = reshape(β_flat, K, J-1)
        P = calculate_probabilities(X, β)
        ll = sum(log.(P[i, Y[i]]) for i in 1:N)
        return -ll  # Negative log-likelihood for minimization
    end
    
    β_init = zeros(K * (J-1))
    result = optimize(loglikelihood, β_init, LBFGS(); autodiff=:forward)
    β_est = reshape(Optim.minimizer(result), K, J-1)
    
    return β_est
end

# Set parameters and simulate data
N, J, K = 1000, 4, 3
Random.seed!(123)
X, Y, β_true = simulate_multinomial_logit(N, J, K)

# Estimate the model
β_est = estimate_multinomial_logit(X, Y)

# Compare true and estimated parameters
println("True β:")
display(β_true)
println("\nEstimated β:")
display(β_est)

# Calculate and display percentage errors
percent_errors = 100 .* abs.(β_est .- β_true) ./ abs.(β_true)
println("\nPercentage errors:")
display(percent_errors)

# Calculate average absolute percentage error
mape = mean(abs.(percent_errors))
println("\nMean Absolute Percentage Error: $(round(mape, digits=2))%")




#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::

using CSV, HTTP, DataFrames, Random, LinearAlgebra, Statistics, Distributions, Optim

# Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Data preparation
df = dropmissing(df, :occupation)
for i in 8:13
    df[df.occupation .== i, :occupation] .= 7
end

# Prepare X and y
X = [ones(size(df,1)) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

# Function to simulate data
function simulate_multinomial_logit(X, β)
    N, K = size(X)
    J = size(β, 2) + 1
    
    U = X * β
    U = [U zeros(N)]
    P = exp.(U .- maximum(U, dims=2)) ./ sum(exp.(U .- maximum(U, dims=2)), dims=2)
    
    Y = [rand(Categorical(P[i, :])) for i in 1:N]
    
    return Y
end

# Function to calculate moments
function calculate_moments(Y, X, J)
    N, K = size(X)
    
    moments = zeros(K * (J-1))
    base_mean = mean(X[Y .== J, :], dims=1)
    
    for j in 1:J-1
        if any(Y .== j)
            moments[(j-1)*K+1:j*K] = mean(X[Y .== j, :], dims=1) - base_mean
        else
            moments[(j-1)*K+1:j*K] .= 0  # Set to zero if category is not present
        end
    end
    
    return moments
end

# SMM objective function
function smm_objective(β_flat, X, y_obs, S, J)
    N, K = size(X)
    β = reshape(β_flat, K, J-1)
    
    m_obs = calculate_moments(y_obs, X, J)
    m_sim = zeros(length(m_obs))
    
    for s in 1:S
        y_sim = simulate_multinomial_logit(X, β)
        m_sim += calculate_moments(y_sim, X, J)
    end
    m_sim /= S
    
    diff = m_obs - m_sim
    return diff' * diff
end

# Set up SMM estimation
N, K = size(X)
J = maximum(y)
S = 10  # Number of simulations

# Initial values (you might want to use MLE estimates as starting values)
β_init = zeros(K * (J-1))

# Debug prints
println("Dimensions:")
println("X: ", size(X))
println("y: ", size(y))
println("β_init: ", size(β_init))
println("J: ", J)

# Test objective function
test_obj = smm_objective(β_init, X, y, S, J)
println("Test objective value: ", test_obj)

# Perform SMM estimation
result = optimize(β -> smm_objective(β, X, y, S, J), β_init, LBFGS(), Optim.Options(show_trace=true, iterations=100))

# Extract SMM estimates
β_smm = reshape(Optim.minimizer(result), K, J-1)

println("\nSMM estimates:")
display(β_smm)

println("\nOptimization result:")
display(result)

# Attempt to calculate standard errors
try
    function numerical_gradient(f, x, eps=1e-8)
        n = length(x)
        grad = zeros(n)
        for i in 1:n
            x_plus = copy(x)
            x_plus[i] += eps
            x_minus = copy(x)
            x_minus[i] -= eps
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
        end
        return grad
    end

    G = numerical_gradient(β -> smm_objective(β, X, y, S, J), Optim.minimizer(result))
    println("\nGradient at optimum:")
    display(G)

    J_hat = G * G'
    println("\nJ_hat matrix:")
    display(J_hat)

    if size(J_hat, 1) > 1
        Σ = inv(J_hat) / N
        se = sqrt.(diag(Σ))
        se_matrix = reshape(se, K, J-1)

        println("\nStandard Errors:")
        display(se_matrix)
    else
        println("\nWarning: J_hat is a scalar. Cannot compute standard errors.")
    end
catch e
    println("\nError in standard error calculation:")
    println(e)
end

# Print additional diagnostics
println("\nFinal objective value: ", Optim.minimum(result))
println("Convergence: ", Optim.converged(result))
println("Iteration count: ", Optim.iterations(result))



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 6
#:::::::::::::::::::::::::::::::::::::::::::::::::::

using Test, LinearAlgebra, Statistics, Distributions, Optim

function wrapall(f::Function)
    return function(args...; kwargs...)
        return f(args...; kwargs...)
    end
end

function run_all_unit_tests()
    @testset "Problem Set 7 Unit Tests" begin
        # Question 1: GMM Linear Regression
        @test begin
            X = [ones(100) randn(100, 2)]
            y = X * [1, 2, 3] + randn(100)
            β_init = zeros(3)
            gmm_obj(β) = (y - X * β)' * X * inv(X' * X) * X' * (y - X * β)
            result = optimize(gmm_obj, β_init, BFGS())
            Optim.converged(result) && length(Optim.minimizer(result)) == 3
        end

        # Question 2: GMM Multinomial Logit
        @test begin
            X = [ones(100) randn(100, 2)]
            y = rand(1:3, 100)
            β = randn(3, 2)
            P = exp.(X * β) ./ sum(exp.(X * β), dims=2)
            g = [y .== j for j in 1:3] - P
            size(g) == (100, 3)
        end

        # Question 3: Multinomial Logit Simulation
        @test begin
            N, K, J = 1000, 3, 4
            X = [ones(N) randn(N, K-1)]
            β = randn(K, J-1)
            U = X * β
            P = exp.(U) ./ sum(exp.(U), dims=2)
            Y = [rand(Categorical(P[i,:])) for i in 1:N]
            length(Y) == N && all(1 .<= Y .<= J)
        end

        # Question 5: SMM Multinomial Logit
        @test begin
            N, K, J = 1000, 3, 4
            X = [ones(N) randn(N, K-1)]
            β = randn(K, J-1)
            sim_data(β) = [rand(Categorical(r)) for r in eachrow(exp.(X*β) ./ sum(exp.(X*β), dims=2))]
            y_obs = sim_data(β)
            moments(y) = [mean(X[y .== j, :], dims=1) - mean(X[y .== J, :], dims=1) for j in 1:J-1]
            obj(β) = sum(abs2, moments(y_obs) - moments(sim_data(reshape(β, K, J-1))))
            result = optimize(obj, vec(β), BFGS(), Optim.Options(iterations=10))
            Optim.converged(result) && Optim.minimum(result) >= 0
        end
    end
end

# Wrap the function using wrapall
wrapped_run_tests = wrapall(run_all_unit_tests)

# Run the wrapped function with tests
wrapped_run_tests()

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 7
#:::::::::::::::::::::::::::::::::::::::::::::::::::


using Test, LinearAlgebra, Statistics, Distributions, Optim

@testset "Problem Set 7 Unit Tests" begin

    @testset "Question 1: GMM Linear Regression" begin
        function gmm_objective(β, X, y)
            ε = y - X * β
            g = X' * ε / size(X, 1)
            return g' * g
        end

        X = [ones(100) randn(100, 2)]
        y = X * [1, 2, 3] + randn(100)
        β_init = zeros(3)

        result = optimize(β -> gmm_objective(β, X, y), β_init, BFGS())
        β_gmm = Optim.minimizer(result)

        @test length(β_gmm) == 3
        @test Optim.converged(result)
        @test gmm_objective(β_gmm, X, y) < gmm_objective(β_init, X, y)
    end

    @testset "Question 2: GMM Multinomial Logit" begin
        function calculate_probabilities(X, β)
            U = X * β
            P = exp.(U) ./ sum(exp.(U), dims=2)
            return P
        end

        function g_function(β, X, y, J)
            N = size(X, 1)
            P = calculate_probabilities(X, β)
            g = zeros(N * J)
            for i in 1:N, j in 1:J
                g[(i-1)*J + j] = (y[i] == j ? 1 : 0) - P[i, j]
            end
            return g
        end

        X = [ones(100) randn(100, 2)]
        y = rand(1:3, 100)
        β = randn(3, 2)
        J = 3

        g = g_function(β, X, y, J)
        @test size(g) == (300,)
        @test all(abs.(g) .<= 1)
    end

    @testset "Question 3: Multinomial Logit Simulation" begin
        function simulate_multinomial_logit(X, β)
            U = X * β
            P = exp.(U) ./ sum(exp.(U), dims=2)
            Y = [rand(Categorical(P[i, :])) for i in 1:size(X, 1)]
            return Y
        end

        N, K, J = 1000, 3, 4
        X = [ones(N) randn(N, K-1)]
        β = randn(K, J-1)

        Y = simulate_multinomial_logit(X, β)
        @test length(Y) == N
        @test all(1 .<= Y .<= J)
        @test length(unique(Y)) == J-1  # J-1 because one category is reference
    end

    @testset "Question 5: SMM Multinomial Logit" begin
        function calculate_moments(Y, X)
            J = maximum(Y)
            K = size(X, 2)
            moments = zeros(K * (J-1))
            for j in 1:J-1
                moments[(j-1)*K+1:j*K] = mean(X[Y .== j, :], dims=1) - mean(X[Y .== J, :], dims=1)
            end
            return moments
        end

        function smm_objective(β, X, y_obs, S)
            N, K = size(X)
            J = maximum(y_obs)
            m_obs = calculate_moments(y_obs, X)
            m_sim = zeros(length(m_obs))
            for s in 1:S
                y_sim = simulate_multinomial_logit(X, reshape(β, K, J-1))
                m_sim += calculate_moments(y_sim, X)
            end
            m_sim /= S
            return (m_obs - m_sim)' * (m_obs - m_sim)
        end

        N, K, J = 1000, 3, 4
        X = [ones(N) randn(N, K-1)]
        β_true = randn(K, J-1)
        y_obs = simulate_multinomial_logit(X, β_true)
        S = 5

        obj_value = smm_objective(vec(β_true), X, y_obs, S)
        @test obj_value >= 0
        @test !isnan(obj_value)
        @test !isinf(obj_value)
    end

end
