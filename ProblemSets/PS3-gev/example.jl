using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using CSV

# Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS3-gev/nlsw88w.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
#:::::::::::::::::::::::::::::::::::::::::::::::::::::#
                     #question 1#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::#                     
# Prepare the data
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occupation

# Define the multinomial logit likelihood function
function mnl_loglikelihood(β, X, Z, y)
    n, p = size(X)
    J = size(Z, 2)
    
    # Reshape β into b (coefficients for X) and g (coefficient for Z)
    b = reshape(β[1:p*(J-1)], p, J-1)
    g = β[end]
    
    ll = 0.0
    for i in 1:n
        denom = 1.0
        for j in 1:(J-1)
            denom += exp(dot(X[i,:], b[:,j]) + g * (Z[i,j] - Z[i,J]))
        end
        
        if y[i] == J
            ll += -log(denom)
        else
            ll += dot(X[i,:], b[:,y[i]]) + g * (Z[i,y[i]] - Z[i,J]) - log(denom)
        end
    end
    
    return -ll  # Return negative log-likelihood for minimization
end

# Optimization
n, p = size(X)
J = size(Z, 2)
initial_β = vcat(vec(zeros(p, J-1)), 0.0)  # Initial guess for parameters

result = optimize(β -> mnl_loglikelihood(β, X, Z, y), initial_β, BFGS())

# Extract and reshape the estimated parameters
β_hat = Optim.minimizer(result)
b_hat = reshape(β_hat[1:p*(J-1)], p, J-1)
g_hat = β_hat[end]

# Print results
println("Estimated coefficients for X variables:")
println(b_hat)
println("\nEstimated coefficient for Z (wage differential):")
println(g_hat)

#:::::::::::::::::::::::::::::::::::::::::::::::::::::#
                        #Question 2#
#:::::::::::::::::::::::::::::::::::::::::::::::::::::#

# ... (previous code remains the same)

# Print results
println("Estimated coefficients for X variables:")
println(b_hat)
println("\nEstimated coefficient for Z (wage differential):")
println(g_hat)

# Add this line to make sure we see the value of g_hat
println("\nValue of g_hat: ", g_hat)

#Interpretation of ĝ = -0.0941937975123376:


#Sign:
#The negative sign is unexpected and counterintuitive. 
#It suggests that as the wage in a particular occupation increases relative to the wage in the base occupation (Other),
# the probability of choosing that occupation decreases.
#Magnitude:
#The coefficient's absolute value is relatively small (close to zero),
# indicating that the effect of wage differentials on occupational choice is weak in this model.
#Specific interpretation:
#For a one-unit increase in the log wage differential (Z_ij - Z_iJ), 
#the log-odds of choosing occupation j over the base occupation (Other) decreases by 0.0941937975123376, holding all else constant.
#In terms of actual wages, if the wage ratio between occupation j and the base occupation increases by 1% (e.g., from 1.00 to 1.01), 
# the log-odds of choosing occupation j over the base occupation decreases by approximately 0.000942 (-0.0941937975123376 * log(1.01)).


#::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                      #Question 3#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::#
using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using CSV

# Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS3-gev/nlsw88w.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Prepare the data
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occupation

# Define nesting structure
WC = [1, 2, 3]  # White collar
BC = [4, 5, 6, 7]  # Blue collar
Other = [8]  # Other
function nested_logit_likelihood(β, X, Z, y)
    n = size(X, 1)
    b_WC, b_BC, λ_WC, λ_BC, g = β[1:3], β[4:6], β[7], β[8], β[9]
    
    ll = 0.0
    for i in 1:n
        V_WC = sum(exp((X[i,:]'b_WC + g*(Z[i,j] - Z[i,8]))/λ_WC) for j in WC)
        V_BC = sum(exp((X[i,:]'b_BC + g*(Z[i,j] - Z[i,8]))/λ_BC) for j in BC)
        
        denom = 1 + V_WC^λ_WC + V_BC^λ_BC
        
        if y[i] in WC
            ll += (X[i,:]'b_WC + g*(Z[i,y[i]] - Z[i,8]))/λ_WC + log(V_WC^(λ_WC-1)) - log(denom)
        elseif y[i] in BC
            ll += (X[i,:]'b_BC + g*(Z[i,y[i]] - Z[i,8]))/λ_BC + log(V_BC^(λ_BC-1)) - log(denom)
        else  # Other
            ll += -log(denom)
        end
    end
    
    return -ll  # Return negative log-likelihood for minimization
end

# Optimization
initial_β = ones(9)  # Initial guess for parameters
result = optimize(β -> nested_logit_likelihood(β, X, Z, y), initial_β, BFGS())

# Extract the estimated parameters
β_hat = Optim.minimizer(result)
b_WC_hat, b_BC_hat = β_hat[1:3], β_hat[4:6]
λ_WC_hat, λ_BC_hat = β_hat[7], β_hat[8]
g_hat = β_hat[9]

# Print results
println("Estimated coefficients for White Collar (WC):")
println(b_WC_hat)
println("\nEstimated coefficients for Blue Collar (BC):")
println(b_BC_hat)
println("\nEstimated λ for White Collar (WC):")
println(λ_WC_hat)
println("\nEstimated λ for Blue Collar (BC):")
println(λ_BC_hat)
println("\nEstimated coefficient for Z (wage differential):")
println(g_hat)  

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                        #Question 4#
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::# 

using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV

function estimate_models()
    # Load the data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS3-gev/nlsw88w.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Prepare the data
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occupation

    # Multinomial Logit
    function mnl_loglikelihood(β, X, Z, y)
        n, p = size(X)
        J = size(Z, 2)
        b = reshape(β[1:p*(J-1)], p, J-1)
        g = β[end]
        ll = 0.0
        for i in 1:n
            denom = 1.0 + sum(exp(dot(X[i,:], b[:,j]) + g * (Z[i,j] - Z[i,J])) for j in 1:(J-1))
            ll += (y[i] == J ? -log(denom) : dot(X[i,:], b[:,y[i]]) + g * (Z[i,y[i]] - Z[i,J]) - log(denom))
        end
        return -ll
    end

    n, p = size(X)
    J = size(Z, 2)
    initial_β_mnl = vcat(vec(zeros(p, J-1)), 0.0)
    result_mnl = optimize(β -> mnl_loglikelihood(β, X, Z, y), initial_β_mnl, BFGS())

    β_hat_mnl = Optim.minimizer(result_mnl)
    b_hat_mnl = reshape(β_hat_mnl[1:p*(J-1)], p, J-1)
    g_hat_mnl = β_hat_mnl[end]

    println("Multinomial Logit Results:")
    println("Estimated coefficients for X variables:")
    println(b_hat_mnl)
    println("Estimated coefficient for Z (wage differential):")
    println(g_hat_mnl)

    # Nested Logit
    WC = [1, 2, 3]  # White collar
    BC = [4, 5, 6, 7]  # Blue collar
    Other = [8]  # Other

    function nested_logit_likelihood(β, X, Z, y)
        n = size(X, 1)
        b_WC, b_BC, λ_WC, λ_BC, g = β[1:3], β[4:6], β[7], β[8], β[9]
        ll = 0.0
        for i in 1:n
            V_WC = sum(exp((X[i,:]'b_WC + g*(Z[i,j] - Z[i,8]))/λ_WC) for j in WC)
            V_BC = sum(exp((X[i,:]'b_BC + g*(Z[i,j] - Z[i,8]))/λ_BC) for j in BC)
            denom = 1 + V_WC^λ_WC + V_BC^λ_BC
            if y[i] in WC
                ll += (X[i,:]'b_WC + g*(Z[i,y[i]] - Z[i,8]))/λ_WC + log(V_WC^(λ_WC-1)) - log(denom)
            elseif y[i] in BC
                ll += (X[i,:]'b_BC + g*(Z[i,y[i]] - Z[i,8]))/λ_BC + log(V_BC^(λ_BC-1)) - log(denom)
            else  # Other
                ll += -log(denom)
            end
        end
        return -ll
    end

    initial_β_nl = ones(9)
    result_nl = optimize(β -> nested_logit_likelihood(β, X, Z, y), initial_β_nl, BFGS())
    β_hat_nl = Optim.minimizer(result_nl)
    b_WC_hat, b_BC_hat = β_hat_nl[1:3], β_hat_nl[4:6]
    λ_WC_hat, λ_BC_hat = β_hat_nl[7], β_hat_nl[8]
    g_hat_nl = β_hat_nl[9]

    println("\nNested Logit Results:")
    println("Estimated coefficients for White Collar (WC):")
    println(b_WC_hat)
    println("Estimated coefficients for Blue Collar (BC):")
    println(b_BC_hat)
    println("Estimated λ for White Collar (WC):")
    println(λ_WC_hat)
    println("Estimated λ for Blue Collar (BC):")
    println(λ_BC_hat)
    println("Estimated coefficient for Z (wage differential):")
    println(g_hat_nl)
end

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                         #Question 5#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
using Test
using LinearAlgebra
using Random

# Function definitions (simplified versions for testing)
function mnl_loglikelihood(β, X, Z, y)
    n, p = size(X)
    J = size(Z, 2)
    if length(y) != n
        return NaN  # Return NaN for mismatched dimensions
    end
    b = reshape(β[1:p*(J-1)], p, J-1)
    g = β[end]
    ll = 0.0
    for i in 1:n
        denom = 1.0 + sum(exp(dot(X[i,:], b[:,j]) + g * (Z[i,j] - Z[i,J])) for j in 1:(J-1))
        ll += (y[i] == J ? -log(denom) : dot(X[i,:], b[:,y[i]]) + g * (Z[i,y[i]] - Z[i,J]) - log(denom))
    end
    return -ll
end

function nested_logit_likelihood(β, X, Z, y)
    n = size(X, 1)
    if length(y) != n
        return NaN  # Return NaN for mismatched dimensions
    end
    b_WC, b_BC, λ_WC, λ_BC, g = β[1:3], β[4:6], β[7], β[8], β[9]
    WC = [1, 2, 3]
    BC = [4, 5, 6, 7]
    ll = 0.0
    for i in 1:n
        V_WC = sum(exp((X[i,:]'b_WC + g*(Z[i,j] - Z[i,8]))/λ_WC) for j in WC)
        V_BC = sum(exp((X[i,:]'b_BC + g*(Z[i,j] - Z[i,8]))/λ_BC) for j in BC)
        denom = 1 + V_WC^λ_WC + V_BC^λ_BC
        if y[i] in WC
            ll += (X[i,:]'b_WC + g*(Z[i,y[i]] - Z[i,8]))/λ_WC + log(V_WC^(λ_WC-1)) - log(denom)
        elseif y[i] in BC
            ll += (X[i,:]'b_BC + g*(Z[i,y[i]] - Z[i,8]))/λ_BC + log(V_BC^(λ_BC-1)) - log(denom)
        else  # Other
            ll += -log(denom)
        end
    end
    return -ll
end

# Separated unit tests
@testset "Econometrics Problem Set 3 Tests" begin
    @testset "Question 1: Multinomial Logit Tests" begin
        X_test = [1.0 2.0; 3.0 4.0]
        Z_test = [0.1 0.2 0.3; 0.4 0.5 0.6]
        y_test = [1, 3]
        β_test = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        @testset "Return Type" begin
            @test typeof(mnl_loglikelihood(β_test, X_test, Z_test, y_test)) == Float64
        end
        
        @testset "Correct Output" begin
            @test mnl_loglikelihood(β_test, X_test, Z_test, y_test) ≈ 4.005284550723061 atol=1e-6
        end
        
        @testset "Zero Inputs" begin
            @test mnl_loglikelihood(zeros(5), zeros(2,2), zeros(2,3), [1,2]) ≈ 2.1972245773362196 atol=1e-6
        end
        
        @testset "Mismatched Dimensions" begin
            @test isnan(mnl_loglikelihood(β_test, X_test, Z_test, [1,2,3]))
        end
    end

    @testset "Question 3: Nested Logit Tests" begin
        X_test = [1.0 2.0 3.0; 4.0 5.0 6.0]
        Z_test = rand(2, 😍
        y_test = [1, 5]
        β_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        @testset "Return Type" begin
            @test typeof(nested_logit_likelihood(β_test, X_test, Z_test, y_test)) == Float64
        end
        
        @testset "Correct Output" begin
            @test nested_logit_likelihood(ones(9), zeros(2,3), zeros(2,8), [1,8]) ≈ 4.1588830833596715 atol=1e-6
        end
        
        @testset "Mismatched Dimensions" begin
            @test isnan(nested_logit_likelihood(β_test, X_test, Z_test, [1,2,3]))
        end
    end
end