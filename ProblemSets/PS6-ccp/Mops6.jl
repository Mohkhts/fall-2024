#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1: 
#:::::::::::::::::::::::::::::::::::::::::::::::::::

using DataFrames, CSV, HTTP, DataFramesMeta

# Read in the data (the second CSV file from PS5)
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Identify the columns to reshape
id_vars = [:Branded]
value_vars = [Symbol("Y$t") for t in 1:20]
odo_vars = [Symbol("Odo$t") for t in 1:20]

# Function to process a chunk of data
function process_chunk(chunk)
    # Reshape the Y variables
    df_long_y = stack(chunk, value_vars, id_vars, variable_name=:period, value_name=:decision)
    
    # Reshape the Odo variables
    df_long_odo = stack(chunk, odo_vars, id_vars, variable_name=:period, value_name=:mileage)
    
    # Clean up the period column
    df_long_y.period = parse.(Int, replace.(string.(df_long_y.period), r"[^0-9]" => ""))
    df_long_odo.period = parse.(Int, replace.(string.(df_long_odo.period), r"[^0-9]" => ""))
    
    # Combine the reshaped dataframes
    df_long = innerjoin(df_long_y, df_long_odo, on=[:Branded, :period])
    
    # If there's a Zst column in the original dataframe, add it to df_long
    if "Zst" in names(chunk)
        df_long = leftjoin(df_long, select(chunk, :Branded, :Zst), on=:Branded)
    end
    
    return df_long
end

# Process data in chunks
chunk_size = 100  # Adjust this value based on your available memory
num_chunks = ceil(Int, nrow(df) / chunk_size)
df_long = DataFrame()

for i in 1:num_chunks
    start_idx = (i-1) * chunk_size + 1
    end_idx = min(i * chunk_size, nrow(df))
    chunk = df[start_idx:end_idx, :]
    chunk_long = process_chunk(chunk)
    append!(df_long, chunk_long)
end

# Sort the dataframe
sort!(df_long, [:Branded, :period])

# Display the first few rows of the reshaped data
println("First few rows of the reshaped dataframe:")
println(first(df_long, 5))

# Display information about the reshaped dataframe
println("\nInformation about the reshaped dataframe:")
println(describe(df_long))

# Save the reshaped data to a CSV file (optional)
CSV.write("df_long.csv", df_long)
println("\nReshaped data saved to 'df_long.csv'")




#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 2: 
#:::::::::::::::::::::::::::::::::::::::::::::::::::

using DataFrames, CSV, GLM

# Read the reshaped data
df_long = CSV.read("df_long.csv", DataFrame)

# Create squared terms
df_long.mileage_sq = df_long.mileage .^ 2
df_long.period_sq = df_long.period .^ 2

# Ensure route_usage is present
if !hasproperty(df_long, :route_usage)
    if hasproperty(df_long, :Zst)
        rename!(df_long, :Zst => :route_usage)
    else
        error("Route usage (Zst) column not found in the dataframe")
    end
end

# Create route_usage squared
df_long.route_usage_sq = df_long.route_usage .^ 2

# Create main interaction terms (up to 2nd order)
interaction_cols = [:mileage, :mileage_sq, :route_usage, :route_usage_sq, :Branded, :period, :period_sq]
for i in 1:length(interaction_cols)
    for j in i:length(interaction_cols)
        new_col = Symbol(interaction_cols[i], "_", interaction_cols[j])
        df_long[!, new_col] = df_long[!, interaction_cols[i]] .* df_long[!, interaction_cols[j]]
    end
end

# Create the formula string (up to 2nd order interactions)
formula_str = "decision ~ " * join(names(df_long)[names(df_long) .!= "decision"], " + ")

# Convert string to formula
formula = eval(Meta.parse("@formula($formula_str)"))

# Use a subset of the data if necessary (adjust the fraction as needed)
sample_fraction = 0.1  # Use 10% of the data
sample_size = floor(Int, nrow(df_long) * sample_fraction)
df_sample = df_long[1:sample_size, :]

# Estimate the flexible logit model
flexible_logit_model = glm(formula, df_sample, Binomial(), LogitLink())

# Display the summary of the model
println("\nModel Summary:")
println(flexible_logit_model)

# Save the results to a file
results_df = DataFrame(
    coefficient = coef(flexible_logit_model),
    std_error = stderror(flexible_logit_model)
)
CSV.write("flexible_logit_results.csv", results_df)

println("\nModel estimation complete. Results saved to 'flexible_logit_results.csv'")

# Display first few rows of the results
println("\nFirst few rows of the results:")
println(first(results_df, 5))






#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3: 
#:::::::::::::::::::::::::::::::::::::::::::::::::::

#PARTa
using LinearAlgebra

function create_grids()
    zval = 0.25:0.01:1.25
    xval = 0:0.125:4.0
    zbin = length(zval)
    xbin = length(xval)
    
    xtran = zeros(zbin * xbin, xbin)
    
    for z in 1:zbin
        for x in 1:xbin
            row = x + (z-1)*xbin
            for x_next in x:xbin
                xtran[row, x_next] = exp(-zval[z] * (xval[x_next] - xval[x])) - 
                                     exp(-zval[z] * (xval[x_next] + 0.125 - xval[x]))
            end
            xtran[row, :] ./= sum(xtran[row, :])
        end
    end
    
    return zval, zbin, xval, xbin, xtran
end

# Call the function to create the grids and transition matrix
zval, zbin, xval, xbin, xtran = create_grids()

# Print some information to verify the results
println("Dimensions of xtran: ", size(xtran))
println("First few values of zval: ", zval[1:5])
println("First few values of xval: ", xval[1:5])
println("Sample of xtran:")
println(xtran[1:5, 1:5])





#PARTb
using DataFrames, GLM, LinearAlgebra, Random

# Simplified flexible logit model (for demonstration)
function simple_logit(X)
    β = [1.0, -0.5, 0.3, -0.1, 0.2]  # Simplified coefficients
    return 1 ./ (1 .+ exp.(-X * β))
end

function compute_future_value(zval, zbin, xval, xbin, xtran, T, β)
    # Create a data frame for all possible states
    state_df = DataFrame(
        Odometer = repeat(xval, outer=zbin),
        RouteUsage = repeat(zval, inner=xbin),
        Branded = zeros(zbin * xbin),
        time = zeros(zbin * xbin)
    )

    # Add squared terms
    state_df.Odometer_sq = state_df.Odometer .^ 2
    state_df.RouteUsage_sq = state_df.RouteUsage .^ 2

    # Initialize future value array
    FV = zeros(size(xtran, 1), 2, T + 1)

    # Compute future values
    for t in T:-1:2
        for b in 0:1
            # Update state data frame for current time and brand
            state_df.time .= t
            state_df.Branded .= b
            
            # Prepare input for simple logit model
            X = hcat(ones(nrow(state_df)), 
                     state_df.Odometer, 
                     state_df.RouteUsage, 
                     state_df.Odometer_sq, 
                     state_df.RouteUsage_sq)
            
            # Predict probabilities using the simple logit model
            p0 = 1 .- simple_logit(X)
            
            # Compute and store future value
            FV[:, b+1, t] = -β * log.(p0)
        end
    end

    return FV
end

# Function to map future values to the original data frame
function map_future_values(df_long, FV, zval, xval, zbin, xbin, xtran, T)
    FVT1 = zeros(nrow(df_long))
    
    for i in 1:nrow(df_long)
        t = df_long.period[i]
        if t < T
            z_index = clamp(findfirst(z -> z >= df_long.RouteUsage[i], zval), 1, zbin)
            x_index = clamp(findfirst(x -> x >= df_long.Odometer[i], xval), 1, xbin)
            row = min(x_index + (z_index - 1) * xbin, size(xtran, 1))
            
            next_row = min(row + 1, size(xtran, 1))
            diff_trans = xtran[row, :] - xtran[next_row, :]
            future_vals = FV[1:length(diff_trans), df_long.Branded[i]+1, t+1]
            FVT1[i] = dot(diff_trans, future_vals)
        end
    end
    
    return FVT1
end

# Usage
β = 0.9  # Discount factor
T = 20   # Number of time periods

# Compute future values
FV = compute_future_value(zval, zbin, xval, xbin, xtran, T, β)

# Create a sample df_long that matches the dimensions of our state space
n_states = size(xtran, 1)
df_long = DataFrame(
    period = repeat(1:T, outer=n_states),
    Odometer = repeat(xval, outer=zbin*T),
    RouteUsage = repeat(repeat(zval, inner=xbin), T),
    Branded = rand(0:1, n_states*T)
)

# Map future values to df_long
FVT1 = map_future_values(df_long, FV, zval, xval, zbin, xbin, xtran, T)

# Add future value term to the original data frame
df_long.fv = FVT1

# Print some information to verify the results
println("Dimensions of xtran: ", size(xtran))
println("Dimensions of FV: ", size(FV))
println("First few values of FVT1: ", FVT1[1:5])
println("Sample of df_long with fv:")
println(first(df_long, 5))


#PARTc

using DataFrames, GLM, Statistics, Distributions

# Assuming df_long and FVT1 are already defined from previous steps

# Add the future value term to the original "long panel" data frame
df_long.fv = FVT1

# Print column names to verify
println("Column names in df_long:")
for (i, name) in enumerate(names(df_long))
    println("$i. $name")
end

# Function to get user input for the decision variable name
function get_decision_var(df)
    println("\nPlease enter the number corresponding to the decision variable:")
    choice = parse(Int, readline())
    return names(df)[choice]
end

# Get the decision variable name from user input
decision_var = get_decision_var(df_long)
println("Using '$decision_var' as the decision variable.")

# Estimate the structural model using GLM
formula_str = "$decision_var ~ Odometer + Branded + fv"
println("Formula being used: $formula_str")

theta_hat_ccp_glm = glm(Meta.parse(formula_str), 
                        df_long, 
                        Binomial(), 
                        LogitLink();
                        offset = :fv)

# Display the summary of the estimated model
println(theta_hat_ccp_glm)

# Extract and display the estimated coefficients
coef_estimates = coef(theta_hat_ccp_glm)
println("\nEstimated Coefficients:")
for (i, name) in enumerate(coefnames(theta_hat_ccp_glm))
    println("$name: ", coef_estimates[i])
end

# Calculate and display standard errors
std_errors = stderror(theta_hat_ccp_glm)
println("\nStandard Errors:")
for (i, name) in enumerate(coefnames(theta_hat_ccp_glm))
    println("$name: ", std_errors[i])
end

# Calculate and display t-statistics
t_stats = coef_estimates ./ std_errors
println("\nt-statistics:")
for (i, name) in enumerate(coefnames(theta_hat_ccp_glm))
    println("$name: ", t_stats[i])
end

# Calculate and display p-values
p_values = 2 .* (1 .- cdf.(TDist(dof_residual(theta_hat_ccp_glm)), abs.(t_stats)))
println("\np-values:")
for (i, name) in enumerate(coefnames(theta_hat_ccp_glm))
    println("$name: ", p_values[i])
end

# Display model fit statistics
println("\nModel Fit Statistics:")
println("AIC: ", aic(theta_hat_ccp_glm))
println("BIC: ", bic(theta_hat_ccp_glm))
println("Log Likelihood: ", loglikelihood(theta_hat_ccp_glm))


#PART D

using LinearAlgebra, Optim, Statistics

function custom_binary_logit(X, y, offset; initial_params=nothing, max_iterations=1000)
    # Prepare data
    n, k = size(X)
    
    if isnothing(initial_params)
        initial_params = zeros(k)
    end
    
    # Define the negative log-likelihood function
    function neg_log_likelihood(params)
        linear_pred = X * params + offset
        prob = 1 ./ (1 .+ exp.(-linear_pred))
        return -sum(y .* log.(prob) + (1 .- y) .* log.(1 .- prob))
    end
    
    # Define the gradient of the negative log-likelihood
    function gradient!(G, params)
        linear_pred = X * params + offset
        prob = 1 ./ (1 .+ exp.(-linear_pred))
        G .= X' * (prob - y)
    end
    
    # Optimize using BFGS
    opt_result = optimize(neg_log_likelihood, gradient!, initial_params, BFGS(), 
                          Optim.Options(iterations=max_iterations))
    
    # Extract results
    coefficients = Optim.minimizer(opt_result)
    std_errors = sqrt.(diag(inv(Optim.hessian!(opt_result))))
    
    return coefficients, std_errors
end



#part e

# Define the wrapall function
function wrapall()
    # PART A: Create grids and transition matrices
    function create_grids()
        zval = 0.25:0.01:1.25
        xval = 0:0.125:4.0
        zbin = length(zval)
        xbin = length(xval)
        
        xtran = zeros(zbin * xbin, xbin)
        
        for z in 1:zbin
            for x in 1:xbin
                row = x + (z-1)*xbin
                for x_next in x:xbin
                    xtran[row, x_next] = exp(-zval[z] * (xval[x_next] - xval[x])) - 
                                         exp(-zval[z] * (xval[x_next] + 0.125 - xval[x]))
                end
                xtran[row, :] ./= sum(xtran[row, :])
            end
        end
        
        return zval, zbin, xval, xbin, xtran
    end

    # Call the function to create grids and the transition matrix
    zval, zbin, xval, xbin, xtran = create_grids()

    # PART B: Compute future value
    function simple_logit(X)
        β = [1.0, -0.5, 0.3, -0.1, 0.2]  # Simplified coefficients
        return 1 ./ (1 .+ exp.(-X * β))
    end

    function compute_future_value(zval, zbin, xval, xbin, xtran, T, β)
        state_df = DataFrame(
            Odometer = repeat(xval, outer=zbin),
            RouteUsage = repeat(zval, inner=xbin),
            Branded = zeros(zbin * xbin),
            time = zeros(zbin * xbin)
        )
        state_df.Odometer_sq = state_df.Odometer .^ 2
        state_df.RouteUsage_sq = state_df.RouteUsage .^ 2

        FV = zeros(size(xtran, 1), 2, T + 1)

        for t in T:-1:2
            for b in 0:1
                state_df.time .= t
                state_df.Branded .= b

                X = hcat(ones(nrow(state_df)), 
                         state_df.Odometer, 
                         state_df.RouteUsage, 
                         state_df.Odometer_sq, 
                         state_df.RouteUsage_sq)
                
                p0 = 1 .- simple_logit(X)
                FV[:, b+1, t] = -β * log.(p0)
            end
        end

        return FV
    end

    β = 0.9
    T = 20
    FV = compute_future_value(zval, zbin, xval, xbin, xtran, T, β)

    # PART C: Map future values to df_long
    df_long = DataFrame(
        period = repeat(1:T, outer=size(xtran, 1)),
        Odometer = repeat(xval, outer=zbin*T),
        RouteUsage = repeat(repeat(zval, inner=xbin), T),
        Branded = rand(0:1, size(xtran, 1)*T)
    )

    function map_future_values(df_long, FV, zval, xval, zbin, xbin, xtran, T)
        FVT1 = zeros(nrow(df_long))
        
        for i in 1:nrow(df_long)
            t = df_long.period[i]
            if t < T
                z_index = clamp(findfirst(z -> z >= df_long.RouteUsage[i], zval), 1, zbin)
                x_index = clamp(findfirst(x -> x >= df_long.Odometer[i], xval), 1, xbin)
                row = min(x_index + (z_index - 1) * xbin, size(xtran, 1))
                
                next_row = min(row + 1, size(xtran, 1))
                diff_trans = xtran[row, :] - xtran[next_row, :]
                future_vals = FV[1:length(diff_trans), df_long.Branded[i]+1, t+1]
                FVT1[i] = dot(diff_trans, future_vals)
            end
        end
        
        return FVT1
    end

    FVT1 = map_future_values(df_long, FV, zval, xval, zbin, xbin, xtran, T)

    # PART D: Estimate the structural parameters
    df_long.fv = FVT1

    # Correct the formula definition using @formula
    formula = @formula(Branded ~ Odometer + fv)

    # Use `fit` function with the correct formula type
    theta_hat_ccp_glm = fit(GeneralizedLinearModel, formula, df_long, Binomial(), LogitLink())

    println("Estimated Coefficients: ", coef(theta_hat_ccp_glm))
end








#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 4: 
#:::::::::::::::::::::::::::::::::::::::::::::::::::



# question 1

using Test, DataFrames

function run_tests()
    @testset "Question 1 Tests" begin
        # Test process_chunk function
        @testset "process_chunk" begin
            # Create a sample chunk DataFrame with the correct structure for testing
            sample_chunk = DataFrame(
                Branded = [1, 0, 1, 0],
                Y1 = [1, 0, 1, 0], Y2 = [0, 1, 0, 1], Y3 = [1, 0, 1, 0],
                Odo1 = [1000, 2000, 3000, 4000], Odo2 = [1100, 2100, 3100, 4100], Odo3 = [1200, 2200, 3200, 4200],
                Zst = [0.5, 1.0, 0.7, 0.9]
            )

            # Assume `process_chunk` is defined and accessible
            result_df = process_chunk(sample_chunk)
            
            # Check if the expected columns are present
            @test all(["Branded", "period", "decision", "mileage", "Zst"] .∈ names(result_df))
            
            # Check the number of rows
            @test nrow(result_df) == 12  # 3 periods × 4 rows in original data
            
            # Check if the 'period' column is correctly converted to integers
            @test eltype(result_df.period) == Int
            
            # Check if the 'decision' column has the correct values for reshaped data
            @test result_df.decision == [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
            
            # Check if the 'mileage' column has the correct values for reshaped data
            @test result_df.mileage == [1000, 2000, 3000, 4000, 1100, 2100, 3100, 4100, 1200, 2200, 3200, 4200]
        end

        # Test overall reshaping and combining functionality
        @testset "Full Data Processing" begin
            # Create a sample DataFrame with minimal structure
            test_df = DataFrame(
                Branded = [1, 0],
                Y1 = [1, 0], Y2 = [0, 1], Y3 = [1, 0],
                Odo1 = [1000, 2000], Odo2 = [1100, 2100], Odo3 = [1200, 2200],
                Zst = [0.5, 1.0]
            )
            
            # Use `process_chunk` on the test data
            reshaped_df = process_chunk(test_df)

            # Test the dimensions of the reshaped dataframe
            @test size(reshaped_df, 1) == 6  # 3 periods × 2 rows

            # Ensure all necessary columns are in the reshaped DataFrame
            @test all([:Branded, :period, :decision, :mileage, :Zst] .∈ names(reshaped_df))

            # Test the values in the reshaped DataFrame
            @test reshaped_df[1, :decision] == 1  # First entry for Y1
            @test reshaped_df[4, :mileage] == 1100  # Second entry for Odo2
        end
    end
end

# Run the unit tests
run_tests()






#question 2

using Test, DataFrames, CSV, GLM

function run_tests()
    @testset "Question 2 Tests" begin
        # Test Data Preparation for the Flexible Logit Model
        @testset "Data Preparation" begin
            # Create a small sample DataFrame for testing
            sample_data = DataFrame(
                Branded = [1, 0, 1, 0],
                period = [1, 2, 3, 4],
                decision = [1, 0, 1, 0],
                mileage = [1000, 2000, 3000, 4000],
                RouteUsage = [0.5, 1.0, 0.7, 0.9]
            )

            # Create squared terms and interaction terms
            sample_data.mileage_sq = sample_data.mileage .^ 2
            sample_data.period_sq = sample_data.period .^ 2
            sample_data.RouteUsage_sq = sample_data.RouteUsage .^ 2

            # Check if the squared terms are correctly computed
            @test sample_data.mileage_sq == [1000000, 4000000, 9000000, 16000000]
            @test sample_data.period_sq == [1, 4, 9, 16]
            @test sample_data.RouteUsage_sq == [0.25, 1.0, 0.49, 0.81]

            # Create interaction terms and check a few of them
            sample_data.Branded_mileage = sample_data.Branded .* sample_data.mileage
            sample_data.Branded_period = sample_data.Branded .* sample_data.period
            @test sample_data.Branded_mileage == [1000, 0, 3000, 0]
            @test sample_data.Branded_period == [1, 0, 3, 0]
        end

        # Test Flexible Logit Model Estimation
        @testset "Flexible Logit Model Estimation" begin
            # Create a subset sample for testing the GLM model
            test_df = DataFrame(
                decision = [1, 0, 1, 0, 1, 0],
                mileage = [100, 150, 200, 250, 300, 350],
                mileage_sq = [10000, 22500, 40000, 62500, 90000, 122500],
                RouteUsage = [0.5, 0.4, 0.6, 0.8, 0.3, 0.7],
                RouteUsage_sq = [0.25, 0.16, 0.36, 0.64, 0.09, 0.49],
                period = [1, 2, 3, 4, 5, 6],
                period_sq = [1, 4, 9, 16, 25, 36],
                Branded = [1, 0, 1, 0, 1, 0]
            )

            # Define the formula for the model
            formula_str = "decision ~ mileage + mileage_sq + RouteUsage + RouteUsage_sq + period + period_sq + Branded"
            formula = eval(Meta.parse("@formula($formula_str)"))

            # Fit the flexible logit model using GLM
            flexible_logit_model = glm(formula, test_df, Binomial(), LogitLink())

            # Check if the model is of the expected type
            @test isa(flexible_logit_model, GLM.GeneralizedLinearModel)

            # Verify the number of coefficients estimated
            @test length(coef(flexible_logit_model)) == 8  # 7 variables + intercept

            # Verify the coefficients are not all zeros
            @test all(abs.(coef(flexible_logit_model)) .> 0.0)
        end

        # Test Results Saving to CSV
        @testset "Save Results" begin
            # Create a mock results DataFrame
            results_df = DataFrame(
                coefficient = [0.5, -0.2, 0.1, 0.3, -0.1, 0.05, -0.02, 0.4],
                std_error = [0.1, 0.05, 0.02, 0.03, 0.01, 0.02, 0.01, 0.06]
            )

            # Save results to a CSV file
            output_file = "test_flexible_logit_results.csv"
            CSV.write(output_file, results_df)

            # Read back and validate the saved file
            read_df = CSV.read(output_file, DataFrame)
            @test read_df == results_df  # Check if saved and read dataframes match
            @test size(read_df, 1) == 8  # Check number of rows
            @test all(["coefficient", "std_error"] .∈ names(read_df))  # Validate column names
        end
    end
end

# Run the unit tests
run_tests()


# question3


using Test, DataFrames, LinearAlgebra, Random, GLM

# Define the necessary functions to test

# Part (A): Grid creation function
function create_grids()
    zval = 0.25:0.01:1.25
    xval = 0:0.125:4.0
    zbin = length(zval)
    xbin = length(xval)
    
    xtran = zeros(zbin * xbin, xbin)
    
    for z in 1:zbin
        for x in 1:xbin
            row = x + (z-1)*xbin
            for x_next in x:xbin
                xtran[row, x_next] = exp(-zval[z] * (xval[x_next] - xval[x])) - 
                                     exp(-zval[z] * (xval[x_next] + 0.125 - xval[x]))
            end
            xtran[row, :] ./= sum(xtran[row, :])
        end
    end
    return zval, zbin, xval, xbin, xtran
end

# Part (B): Compute future value function
function compute_future_value(zval, zbin, xval, xbin, xtran, T, β)
    state_df = DataFrame(
        Odometer = repeat(xval, outer=zbin),
        RouteUsage = repeat(zval, inner=xbin),
        Branded = zeros(zbin * xbin),
        time = zeros(zbin * xbin)
    )
    state_df.Odometer_sq = state_df.Odometer .^ 2
    state_df.RouteUsage_sq = state_df.RouteUsage .^ 2

    FV = zeros(size(xtran, 1), 2, T + 1)

    for t in T:-1:2
        for b in 0:1
            state_df.time .= t
            state_df.Branded .= b

            X = hcat(ones(nrow(state_df)), 
                     state_df.Odometer, 
                     state_df.RouteUsage, 
                     state_df.Odometer_sq, 
                     state_df.RouteUsage_sq)
            
            p0 = 1 .- (1 ./ (1 .+ exp.(-X * [1.0, -0.5, 0.3, -0.1, 0.2])))
            FV[:, b+1, t] = -β * log.(p0)
        end
    end
    return FV
end

# Part (C): Map future values function
function map_future_values(df_long, FV, zval, xval, zbin, xbin, xtran, T)
    FVT1 = zeros(nrow(df_long))
    
    for i in 1:nrow(df_long)
        t = df_long.period[i]
        if t < T
            z_index = clamp(findfirst(z -> z >= df_long.RouteUsage[i], zval), 1, zbin)
            x_index = clamp(findfirst(x -> x >= df_long.Odometer[i], xval), 1, xbin)
            row = min(x_index + (z_index - 1) * xbin, size(xtran, 1))
            
            next_row = min(row + 1, size(xtran, 1))
            diff_trans = xtran[row, :] - xtran[next_row, :]
            future_vals = FV[1:length(diff_trans), df_long.Branded[i]+1, t+1]
            FVT1[i] = dot(diff_trans, future_vals)
        end
    end
    return FVT1
end

# Short Unit Tests for Each Part of Question 3
function run_tests()
    @testset "Question 3 Unit Tests" begin
        
        # Part (A) Test: Create Grids
        @testset "Part A - create_grids" begin
            zval, zbin, xval, xbin, xtran = create_grids()
            @test length(zval) == zbin
            @test length(xval) == xbin
            @test size(xtran) == (zbin * xbin, xbin)
        end

        # Part (B) Test: Compute Future Value
        @testset "Part B - compute_future_value" begin
            zval, zbin, xval, xbin, xtran = create_grids()
            T = 5   # Number of time periods for testing
            β = 0.9 # Discount factor
            FV = compute_future_value(zval, zbin, xval, xbin, xtran, T, β)
            @test size(FV) == (zbin * xbin, 2, T + 1)
        end

        # Part (C) Test: Map Future Values
        @testset "Part C - map_future_values" begin
            zval, zbin, xval, xbin, xtran = create_grids()
            T = 5
            sample_df_long = DataFrame(
                period = [1, 2, 3, 4, 1, 2, 3, 4],
                Odometer = [0.25, 0.75, 1.25, 1.75, 0.25, 0.75, 1.25, 1.75],
                RouteUsage = [0.5, 0.75, 1.0, 0.5, 0.75, 1.0, 1.25, 0.5],
                Branded = [0, 0, 1, 1, 0, 1, 0, 1]
            )
            β = 0.9
            FV = compute_future_value(zval, zbin, xval, xbin, xtran, T, β)
            FVT1 = map_future_values(sample_df_long, FV, zval, xval, zbin, xbin, xtran, T)
            @test length(FVT1) == nrow(sample_df_long)
        end

        # Part (D) Test: Structural Model Estimation
        @testset "Part D - Structural Model Estimation" begin
            df_long = DataFrame(
                period = repeat(1:4, outer=5),
                Odometer = repeat([0.25, 0.5, 0.75, 1.0, 1.25], inner=4),
                RouteUsage = repeat([0.5, 1.0, 1.5, 0.75, 1.25], inner=4),
                Branded = repeat([0, 1], inner=10),
                fv = rand(20)
            )
            formula = @formula(Branded ~ Odometer + fv)
            structural_model = glm(formula, df_long, Binomial(), LogitLink())
            @test isa(structural_model, GLM.GeneralizedLinearModel)
            @test length(coef(structural_model)) == 3  # 2 variables + intercept
        end

    end
end

# Run the unit tests
run_tests()
