#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Load the Optim package
using Optim

# Define the function to maximize
f(x) = -x[1]^4 - 10x[1]^3 - 2x[1]^2 - 3x[1] - 2

# Define the negated version of f(x) for minimization (since Optim minimizes)
negf(x) = x[1]^4 + 10x[1]^3 + 2x[1]^2 + 3x[1] + 2

# Start with a random initial guess
startval = rand(1)  # You can use a different starting value if needed

# Use the L-BFGS optimization algorithm to minimize the negative of the function
result = optimize(negf, startval, LBFGS())

# Print the result
println("Minimizer: ", result.minimizer)
println("Maximum value of f(x): ", -result.minimum)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Load necessary packages
using DataFrames
using CSV
using HTTP
using Optim

# Load the data from the given URL
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Prepare the data for regression
X = [ones(size(df,1)) df.age df.race.==1 df.collgrad.==1]  # Matrix of independent variables
y = df.married .== 1  # Dependent variable (binary)

# Define the OLS objective function
function ols(beta, X, y)
    ssr = (y .- X*beta)'*(y .- X*beta)  # Sum of squared residuals
    return ssr
end

# Perform the optimization to estimate the OLS coefficients
beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(),
                        Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))

# Print the estimated coefficients (minimizer)
println("OLS estimates using Optim package: ", beta_hat_ols.minimizer)

# For comparison, using matrix inversion to solve OLS
using LinearAlgebra
b_ols = inv(X'X)*X'y
println("OLS estimates using matrix inversion: ", b_ols)

# Additionally, using GLM to check the same model
using GLM
df.white = df.race .== 1  # Add white as a variable (binary)
bols_lm = lm(@formula(married ~ age + white + collgrad), df)
println("OLS estimates using GLM package: ", coef(bols_lm))

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::
using DataFrames
using CSV
using HTTP
using Optim
using LinearAlgebra  # for matrix operations like dot product

# Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Prepare the data (X matrix and y vector)
X = [ones(size(df,1)) df.age df.race .== 1 df.collgrad .== 1]  # Intercept + features
y = df.married .== 1  # Dependent variable (married indicator)

# Define the logit likelihood function (negative, because Optim minimizes)
function logit_loglikelihood_neg(beta, X, y)
    linear_comb = X * beta
    p = 1.0 ./ (1.0 .+ exp.(-linear_comb))  # Logistic function (probabilities)
    loglikelihood = sum(y .* log.(p) .+ (1 .- y) .* log.(1 .- p))  # Log-likelihood
    return -loglikelihood  # Return negative because Optim minimizes
end

# Initial guess for beta
beta_init = zeros(size(X, 2))

# Perform the optimization using LBFGS method
result = optimize(b -> logit_loglikelihood_neg(b, X, y), beta_init, LBFGS())

# Print the estimated coefficients (minimizer)
println("Estimated coefficients (logit model): ", result.minimizer)


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::
using DataFrames
using CSV
using HTTP
using GLM

# Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Prepare the data for GLM
df.white = df.race .== 1  # Convert race to a binary variable for white/non-white

# Fit the logistic regression model using GLM
logit_model = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())

# Display the model summary
println("Logit Model Summary using GLM package:")
display(logit_model)
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::
using DataFrames
using CSV
using HTTP
using FreqTables
using Optim

# Load the data from the URL
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Check the frequency table of occupation to see small categories
freqtable(df, :occupation)

# Remove rows where occupation is missing
df = dropmissing(df, :occupation)

# Aggregate small categories (8, 9, 10, 11, 12, 13 into category 7)
df[df.occupation .== 8, :occupation] .= 7
df[df.occupation .== 9, :occupation] .= 7
df[df.occupation .== 10, :occupation] .= 7
df[df.occupation .== 11, :occupation] .= 7
df[df.occupation .== 12, :occupation] .= 7
df[df.occupation .== 13, :occupation] .= 7

# Verify the problem is solved
freqtable(df, :occupation)

# Re-define X and y after modifying the number of rows
X = [ones(size(df,1)) df.age df.race .== 1 df.collgrad .== 1]  # Independent variables
y = df.occupation  # Dependent variable (occupation)

# Define the softmax (multinomial logit) probability function
function softmax(X, beta)
    exps = exp.(X * beta)  # Exponentiate X * beta for each category
    return exps ./ sum(exps, dims=2)  # Normalize to get probabilities
end

# Define the negative log-likelihood for multinomial logit
function multinomial_loglikelihood_neg(beta, X, y, K)
    N = size(X, 1)  # Number of observations
    beta = reshape(beta, (size(X, 2), K))  # Reshape beta into a matrix with K columns
    probs = softmax(X, beta)  # Compute the probabilities
    ll = 0.0
    for i in 1:N
        ll += log(probs[i, y[i]])  # Log-likelihood for each observation
    end
    return -ll  # Return negative log-likelihood
end

# Number of categories (K) after aggregating
K = 7

# Starting values for beta (randomized)
beta_init = randn(size(X, 2) * K)

# Perform optimization using L-BFGS algorithm
result = optimize(b -> multinomial_loglikelihood_neg(b, X, y, K), beta_init, LBFGS(), Optim.Options(g_tol=1e-5))

# Print the estimated coefficients (minimizer)
beta_hat = reshape(result.minimizer, (size(X, 2), K))
println("Estimated coefficients (multinomial logit model): ")
display(beta_hat)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 6
#:::::::::::::::::::::::::::::::::::::::::::::::::::
using DataFrames
using CSV
using HTTP
using FreqTables
using Optim

# Function to run the multinomial logit model
function run_multinomial_logit()
    # Load the data from the URL
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Check the frequency table of occupation to see small categories
    println("Initial Frequency Table of Occupation:")
    display(freqtable(df, :occupation))

    # Remove rows where occupation is missing
    df = dropmissing(df, :occupation)

    # Aggregate small categories (8, 9, 10, 11, 12, 13 into category 7)
    df[df.occupation .== 8, :occupation] .= 7
    df[df.occupation .== 9, :occupation] .= 7
    df[df.occupation .== 10, :occupation] .= 7
    df[df.occupation .== 11, :occupation] .= 7
    df[df.occupation .== 12, :occupation] .= 7
    df[df.occupation .== 13, :occupation] .= 7

    # Verify the problem is solved
    println("Updated Frequency Table of Occupation (after aggregation):")
    display(freqtable(df, :occupation))

    # Re-define X and y after modifying the number of rows
    X = [ones(size(df,1)) df.age df.race .== 1 df.collgrad .== 1]  # Independent variables
    y = df.occupation  # Dependent variable (occupation)

    # Define the softmax (multinomial logit) probability function
    function softmax(X, beta)
        exps = exp.(X * beta)  # Exponentiate X * beta for each category
        return exps ./ sum(exps, dims=2)  # Normalize to get probabilities
    end

    # Define the negative log-likelihood for multinomial logit
    function multinomial_loglikelihood_neg(beta, X, y, K)
        N = size(X, 1)  # Number of observations
        beta = reshape(beta, (size(X, 2), K))  # Reshape beta into a matrix with K columns
        probs = softmax(X, beta)  # Compute the probabilities
        ll = 0.0
        for i in 1:N
            ll += log(probs[i, y[i]])  # Log-likelihood for each observation
        end
        return -ll  # Return negative log-likelihood
    end

    # Number of categories (K) after aggregating
    K = 7

    # Starting values for beta (randomized)
    beta_init = randn(size(X, 2) * K)

    # Perform optimization using L-BFGS algorithm
    println("Running optimization...")
    result = optimize(b -> multinomial_loglikelihood_neg(b, X, y, K), beta_init, LBFGS(), Optim.Options(g_tol=1e-5))

    # Print the estimated coefficients (minimizer)
    beta_hat = reshape(result.minimizer, (size(X, 2), K))
    println("Estimated coefficients (multinomial logit model): ")
    display(beta_hat)
end

# Call the function to run the model
run_multinomial_logit()

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 7
#:::::::::::::::::::::::::::::::::::::::::::::::::::

using DataFrames
using CSV
using HTTP
using FreqTables
using Optim

# Softmax function for multinomial logit
function softmax(X, beta)
    exps = exp.(X * beta)  # Matrix of exponentiated values (N x K)
    sum_exps = sum(exps, dims=2)  # Row-wise sum (N x 1)
    return exps ./ sum_exps  # Normalize each row
end

# Negative log-likelihood function for multinomial logit
function multinomial_loglikelihood_neg(beta, X, y, K)
    N = size(X, 1)  # Number of observations (rows of X)
    P = size(X, 2)  # Number of features (columns of X)
    
    # Reshape beta into a P x K matrix
    beta = reshape(beta, (P, K))  
    
    # Calculate probabilities using softmax
    probs = softmax(X, beta)  # Result will be N x K
    
    # Initialize log-likelihood
    ll = 0.0
    for i in 1:N
        ll += log(probs[i, y[i]])  # Log-likelihood for each observation
    end
    
    return -ll  # Return the negative log-likelihood
end

# Function to run the multinomial logit model
function run_multinomial_logit()
    # Load the data from the URL
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    
    # Show the frequency table of occupation
    println("Initial Frequency Table of Occupation:")
    display(freqtable(df, :occupation))
    
    # Remove missing data in the occupation column
    df = dropmissing(df, :occupation)

    # Group smaller categories into category 7
    df[df.occupation .== 8, :occupation] .= 7
    df[df.occupation .== 9, :occupation] .= 7
    df[df.occupation .== 10, :occupation] .= 7
    df[df.occupation .== 11, :occupation] .= 7
    df[df.occupation .== 12, :occupation] .= 7
    df[df.occupation .== 13, :occupation] .= 7

    # Show the updated frequency table
    println("Updated Frequency Table of Occupation:")
    display(freqtable(df, :occupation))

    # Define independent variables (X) and dependent variable (y)
    X = [ones(size(df,1)) df.age df.race .== 1 df.collgrad .== 1]  # Add intercept and features
    y = df.occupation  # Dependent variable (occupation)
    
    # Number of categories
    K = 7

    # Initialize beta with random values
    beta_init = randn(size(X, 2) * K)  # P*K elements in beta

    # Run optimization
    println("Running optimization...")
    result = optimize(b -> multinomial_loglikelihood_neg(b, X, y, K), beta_init, LBFGS(), Optim.Options(g_tol=1e-5))
    
    # Reshape the result into a P x K matrix for the estimated coefficients
    beta_hat = reshape(result.minimizer, (size(X, 2), K))
    println("Estimated coefficients (multinomial logit model):")
    display(beta_hat)
end

# Call the function to run the multinomial logit model
run_multinomial_logit()
