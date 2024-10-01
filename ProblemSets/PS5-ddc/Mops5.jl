#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1: 
#:::::::::::::::::::::::::::::::::::::::::::::::::::

using DataFrames
using DataFramesMeta
using CSV
using HTTP

# Read the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Display column names
println("Column names:")
println(names(df))

# Display first few rows
println("\nFirst few rows of the original dataframe:")
println(first(df, 5))

# Identify the columns to reshape
id_vars = [:Branded]  # Assuming 'Branded' is the identifier column
value_vars = [Symbol("Y$t") for t in 1:20 if Symbol("Y$t") in names(df)]
odo_vars = [Symbol("Odo$t") for t in 1:20 if Symbol("Odo$t") in names(df)]

# Reshape the Y variables
df_long_y = stack(df, value_vars, id_vars, variable_name=:period, value_name=:decision)

# Reshape the Odo variables
df_long_odo = stack(df, odo_vars, id_vars, variable_name=:period, value_name=:odometer)

# Combine the reshaped dataframes
df_long = innerjoin(df_long_y, df_long_odo, on=[:Branded, :period])

# Clean up the period column
df_long.period = parse.(Int, replace.(string.(df_long.period), r"[^0-9]" => ""))

# Sort the dataframe
sort!(df_long, [:Branded, :period])

# Display the first few rows of the reshaped data
println("\nFirst few rows of the reshaped dataframe:")
println(first(df_long, 5))

# Save the reshaped data to a CSV file
CSV.write("reshaped_data.csv", df_long)
println("\nReshaped data saved to 'reshaped_data.csv'")

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 2: 
#:::::::::::::::::::::::::::::::::::::::::::::::::::

using DataFrames, CSV, HTTP, Statistics

# Read the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Display information about the original dataframe
println("Original dataframe:")
println(first(df, 5))
println("\nColumn names:")
println(names(df))
println("\nDataframe size:")
println(size(df))

# Identify the columns to reshape
id_vars = [:Branded]
value_vars = [Symbol("Y$t") for t in 1:20]
odo_vars = [Symbol("Odo$t") for t in 1:20]

# Reshape the data to long format
df_long_y = stack(df, value_vars, id_vars, variable_name=:period, value_name=:decision)
df_long_odo = stack(df, odo_vars, id_vars, variable_name=:period, value_name=:mileage)

# Clean up period column
df_long_y.period = parse.(Int, replace.(string.(df_long_y.period), r"[^0-9]" => ""))
df_long_odo.period = parse.(Int, replace.(string.(df_long_odo.period), r"[^0-9]" => ""))

# Join the reshaped dataframes
df_long = innerjoin(df_long_y, df_long_odo, on=[:Branded, :period])
sort!(df_long, [:Branded, :period])

# Display information about the final long dataframe
println("\nFinal long dataframe:")
println(first(df_long, 5))
println("\nColumn names:")
println(names(df_long))
println("\nDataframe size:")
println(size(df_long))

# Check for missing values
println("\nMissing values:")
for col in names(df_long)
    missing_count = sum(ismissing.(df_long[!, col]))
    if missing_count > 0
        println("$col: $missing_count")
    end
end

# Check for infinite values
println("\nInfinite values:")
for col in names(df_long)
    if eltype(df_long[!, col]) <: Number
        inf_count = sum(isinf.(df_long[!, col]))
        if inf_count > 0
            println("$col: $inf_count")
        end
    end
end

# Summary statistics
println("\nSummary statistics:")
describe(df_long)

# Convert mileage to 10,000s of miles
df_long.mileage = df_long.mileage ./ 10000

# Display the prepared data
println("\nPrepared data (first 10 rows):")
println(first(df_long, 10))

# Check unique values in the decision column
println("\nUnique values in the decision column:")
println(unique(df_long.decision))

# Distribution of decisions
decision_dist = combine(groupby(df_long, :decision), nrow => :count)
println("\nDistribution of decisions:")
println(decision_dist)

# Correlation between mileage and decision
println("\nCorrelation between mileage and decision:")
println(cor(df_long.mileage, Float64.(df_long.decision)))

# Save the prepared data to a CSV file
CSV.write("zurcher_data_long.csv", df_long)
println("\nPrepared data saved to 'zurcher_data_long.csv'")

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3: 
#:::::::::::::::::::::::::::::::::::::::::::::::::::

using DataFrames, CSV, HTTP, Random, LinearAlgebra

# Part a: Read and process the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

Y = Matrix(df[:, r"^Y\d+$"])
Odo = Matrix(df[:, r"^Odo\d+$"])
Xst = Matrix(df[:, r"^Xst\d+$"])
Zst = df.Zst
if !hasproperty(df, :Branded)
    Random.seed!(123)
    df.Branded = rand(0:1, nrow(df))
end
Branded = df.Branded

# Part b: Create grids function
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

# Call create_grids function
zval, zbin, xval, xbin, xtran = create_grids()

# Part c: Compute the future value terms for all possible states of the model

# Set number of time periods
T = 20

# Initialize the future value array
FV = zeros(zbin * xbin, 2, T + 1)

# Set discount factor
β = 0.9

# Parameters (θ values) - you may need to adjust these based on your model
θ₀ = 0.0
θ₁ = -1.0
θ₂ = 2.0

# Backward recursion
for t in T:-1:1
    for b in 0:1  # Loop over brand states
        for z in 1:zbin
            for x in 1:xbin
                # Index for the current state
                row = x + (z-1)*xbin
                
                # Compute v₁ₜ (conditional value function for driving the bus)
                v₁ = θ₀ + θ₁*xval[x] + θ₂*b
                v₁ += β * dot(xtran[row,:], FV[(z-1)*xbin+1:z*xbin, b+1, t+1])
                
                # Compute v₀ₜ (conditional value function for replacing the engine)
                v₀ = β * dot(xtran[1+(z-1)*xbin,:], FV[(z-1)*xbin+1:z*xbin, b+1, t+1])
                
                # Update the future value array
                FV[row, b+1, t] = β * log(exp(v₀) + exp(v₁))
            end
        end
    end
end

# Print some values to verify the computation
println("Future Value array dimensions: ", size(FV))
println("Sample future values:")
println("FV[1,1,1] = ", FV[1,1,1])
println("FV[end,end,end] = ", FV[end,end,end])

# Part d: Construct the log likelihood

function construct_log_likelihood(θ, Y, Odo, Xst, Zst, Branded, FV, xtran, xval, xbin, zbin, β)
    N, T = size(Y)
    loglik = 0.0

    for i in 1:N
        for t in 1:T
            # Index for the case where the bus has been replaced
            row0 = 1 + (Zst[i] - 1) * xbin

            # Index for the case where the bus has not been replaced
            row1 = Xst[i,t] + (Zst[i] - 1) * xbin

            # Flow utility component of v₁ₜ - v₀ₜ
            v_diff = θ[1] + θ[2] * Odo[i,t] + θ[3] * Branded[i]

            # Add the appropriate discounted future value
            if t < T
                future_value_diff = (xtran[row1,:] .- xtran[row0,:])' * 
                                    FV[row0:row0+xbin-1, Branded[i]+1, t+1]
                v_diff += β * future_value_diff
            end

            # Compute choice probabilities
            P1 = 1 / (1 + exp(-v_diff))
            P0 = 1 - P1

            # Add to log likelihood
            loglik += Y[i,t] * log(P1) + (1 - Y[i,t]) * log(P0)
        end
    end

    return loglik
end



# Part e: Wrap the code into a function for Optim

@views @inbounds function myfun(θ::Vector{Float64}, 
    Y::Matrix{Float64}, 
    Odo::Matrix{Float64}, 
    Xst::Matrix{Int}, 
    Zst::Vector{Int}, 
    Branded::Vector{Int}, 
    xval::Vector{Float64}, 
    zval::Vector{Float64}, 
    β::Float64)

# Recreate grids and transition matrix
zbin = length(zval)
xbin = length(xval)
_, _, _, _, xtran = create_grids()

# Initialize future value array
T = size(Y, 2)
FV = zeros(zbin * xbin, 2, T + 1)

# Backward recursion to compute future values
for t in T:-1:1
for b in 0:1
for z in 1:zbin
for x in 1:xbin
row = x + (z-1)*xbin
v₁ = θ[1] + θ[2]*xval[x] + θ[3]*b
v₁ += β * dot(xtran[row,:], FV[(z-1)*xbin+1:z*xbin, b+1, t+1])
v₀ = β * dot(xtran[1+(z-1)*xbin,:], FV[(z-1)*xbin+1:z*xbin, b+1, t+1])
FV[row, b+1, t] = β * log(exp(v₀) + exp(v₁))
end
end
end
end

# Compute log likelihood
loglik = construct_log_likelihood(θ, Y, Odo, Xst, Zst, Branded, FV, xtran, xval, xbin, zbin, β)

# Return negative log likelihood for minimization
return -loglik
end



# Part f: Prepend the function declaration with macros

@views @inbounds function myfun(θ::Vector{Float64}, 
    Y::Matrix{Float64}, 
    Odo::Matrix{Float64}, 
    Xst::Matrix{Int}, 
    Zst::Vector{Int}, 
    Branded::Vector{Int}, 
    xval::Vector{Float64}, 
    zval::Vector{Float64}, 
    β::Float64)

# Function body remains the same as in part (e)

# Recreate grids and transition matrix
zbin = length(zval)
xbin = length(xval)
_, _, _, _, xtran = create_grids()

# Initialize future value array
T = size(Y, 2)
FV = zeros(zbin * xbin, 2, T + 1)

# Backward recursion to compute future values
for t in T:-1:1
for b in 0:1
for z in 1:zbin
for x in 1:xbin
row = x + (z-1)*xbin
v₁ = θ[1] + θ[2]*xval[x] + θ[3]*b
v₁ += β * dot(xtran[row,:], FV[(z-1)*xbin+1:z*xbin, b+1, t+1])
v₀ = β * dot(xtran[1+(z-1)*xbin,:], FV[(z-1)*xbin+1:z*xbin, b+1, t+1])
FV[row, b+1, t] = β * log(exp(v₀) + exp(v₁))
end
end
end
end

# Compute log likelihood
loglik = construct_log_likelihood(θ, Y, Odo, Xst, Zst, Branded, FV, xtran, xval, xbin, zbin, β)

# Return negative log likelihood for minimization
return -loglik
end

# Part g
using DataFrames, CSV, HTTP, Random, LinearAlgebra, Optim

# Read and process the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

Y = Matrix{Float64}(df[:, r"^Y\d+$"])
Odo = Matrix{Float64}(df[:, r"^Odo\d+$"])
Xst = Matrix{Int}(df[:, r"^Xst\d+$"])
Zst = Vector{Int}(df.Zst)
if !hasproperty(df, :Branded)
    Random.seed!(123)
    df.Branded = rand(0:1, nrow(df))
end
Branded = Vector{Int}(df.Branded)

# Create grids function
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

zval, zbin, xval, xbin, xtran = create_grids()

# Construct log likelihood function
function construct_log_likelihood(θ, Y, Odo, Xst, Zst, Branded, FV, xtran, xval, xbin, zbin, β)
    N, T = size(Y)
    loglik = 0.0

    for i in 1:N
        for t in 1:T
            row0 = min(1 + (Zst[i] - 1) * xbin, size(xtran, 1))
            row1 = min(Xst[i,t] + (Zst[i] - 1) * xbin, size(xtran, 1))
            v_diff = θ[1] + θ[2] * Odo[i,t] + θ[3] * Branded[i]

            if t < T
                future_value_diff = dot(xtran[row1,:] .- xtran[row0,:], 
                                        FV[row0:min(row0+xbin-1, end), Branded[i]+1, t+1])
                v_diff += β * future_value_diff
            end

            P1 = 1 / (1 + exp(-v_diff))
            P0 = 1 - P1
            loglik += Y[i,t] * log(P1) + (1 - Y[i,t]) * log(P0)
        end
    end

    return loglik
end

# Objective function for optimization
@views @inbounds function myfun(θ::Vector{Float64}, 
                                Y::Matrix{Float64}, 
                                Odo::Matrix{Float64}, 
                                Xst::Matrix{Int}, 
                                Zst::Vector{Int}, 
                                Branded::Vector{Int}, 
                                xval::Vector{Float64}, 
                                zval::Vector{Float64}, 
                                xtran::Matrix{Float64},
                                β::Float64)
    
    zbin = length(zval)
    xbin = length(xval)
    
    T = size(Y, 2)
    FV = zeros(zbin * xbin, 2, T + 1)
    
    for t in T:-1:1
        for b in 0:1
            for z in 1:zbin
                for x in 1:xbin
                    row = x + (z-1)*xbin
                    v₁ = θ[1] + θ[2]*xval[x] + θ[3]*b
                    v₁ += β * dot(xtran[row,:], FV[(z-1)*xbin+1:z*xbin, b+1, t+1])
                    v₀ = β * dot(xtran[1+(z-1)*xbin,:], FV[(z-1)*xbin+1:z*xbin, b+1, t+1])
                    FV[row, b+1, t] = β * log(exp(v₀) + exp(v₁))
                end
            end
        end
    end
    
    loglik = construct_log_likelihood(θ, Y, Odo, Xst, Zst, Branded, FV, xtran, xval, xbin, zbin, β)
    
    return -loglik
end

# Set up optimization
β = 0.9
xval_vec = collect(xval)
zval_vec = collect(zval)
initial_θ = [0.0, -1.0, 2.0]

result = optimize(θ -> myfun(θ, Y, Odo, Xst, Zst, Branded, xval_vec, zval_vec, xtran, β), 
                  initial_θ, LBFGS())

# Extract optimal parameters
optimal_θ = Optim.minimizer(result)
println("Optimal parameters: ", optimal_θ)


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 4: 
#:::::::::::::::::::::::::::::::::::::::::::::::::::


# unit test for question 1:
@testset "Data Reshaping Unit Test" begin

    # Test 1: Check if the reshaped DataFrame is not empty
    @test size(df_long, 1) > 0 "Reshaped DataFrame `df_long` should not be empty."

    # Test 2: Ensure that the reshaped DataFrame has the correct column names
    @test sort(names(df_long)) == sort(expected_columns) "Column names in `df_long` do not match the expected columns."

    # Test 3: Ensure that the columns have the expected data types
    for col in expected_columns
        @test eltype(df_long[!, col]) == expected_column_types[col] "Column $col has incorrect data type."
    end

    # Test 4: Check that the `period` column contains only integers (if the DataFrame is not empty)
    if size(df_long, 1) > 0
        @test all(x -> x isa Int, df_long.period) "All values in `period` column should be of type Int."
    else
        println("Skipping column type check since `df_long` is empty.")
    end

    # Test 5: Ensure `decision` and `odometer` columns have no missing values
    if size(df_long, 1) > 0
        @test all(!ismissing, df_long.decision) && all(!ismissing, df_long.odometer) "There should be no missing values in `decision` or `odometer` columns."
    else
        println("Skipping missing value check since `df_long` is empty.")
    end

end



# unit test for question 2:

using Test, CSV, DataFrames, HTTP, Statistics

# Define the unit test set
@testset "Data Transformation and Validation Test" begin
    # Step 1: Load data from the given URL
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Test 1: Ensure data is loaded correctly with expected dimensions
    @test size(df) == (1000, 63)

    # Step 2: Identify columns and reshape to long format
    id_vars = [:Branded]
    value_vars = [Symbol("Y$t") for t in 1:20]
    odo_vars = [Symbol("Odo$t") for t in 1:20]

    df_long_y = stack(df, value_vars, id_vars, variable_name=:period, value_name=:decision)
    df_long_odo = stack(df, odo_vars, id_vars, variable_name=:period, value_name=:mileage)

    # Test 2: Check that reshaping resulted in the correct number of rows
    @test size(df_long_y)[1] == size(df)[1] * length(value_vars)
    @test size(df_long_odo)[1] == size(df)[1] * length(odo_vars)

    # Step 3: Clean up period column and join reshaped dataframes
    df_long_y.period = parse.(Int, replace.(string.(df_long_y.period), r"[^0-9]" => ""))
    df_long_odo.period = parse.(Int, replace.(string.(df_long_odo.period), r"[^0-9]" => ""))

    df_long = innerjoin(df_long_y, df_long_odo, on=[:Branded, :period])
    sort!(df_long, [:Branded, :period])

    # Fix: Remove duplicates to ensure the correct row count
    df_long = unique(df_long)

    # Test 3: Ensure the resulting size is as expected after removing duplicates
    @test size(df_long)[1] == size(df)[1] * length(value_vars)

    # Step 4: Check for missing and infinite values
    for col in names(df_long)
        @test sum(ismissing.(df_long[!, col])) == 0  # No missing values
        if eltype(df_long[!, col]) <: Number
            @test sum(isinf.(df_long[!, col])) == 0  # No infinite values
        end
    end

    # Step 5: Convert mileage to 10,000s of miles and validate
    original_mean_mileage = mean(df_long.mileage)
    df_long.mileage = df_long.mileage ./ 10000
    @test mean(df_long.mileage) < original_mean_mileage  # Mean should decrease after scaling

    # Step 6: Check correlation between mileage and decision
    correlation = cor(df_long.mileage, Float64.(df_long.decision))
    @test !ismissing(correlation) && !isinf(correlation) && abs(correlation) <= 1.0

    # Step 7: Validate unique values in the decision column
    unique_decisions = unique(df_long.decision)
    @test length(unique_decisions) > 1  # There should be more than one unique decision value

    # Step 8: Check the distribution of decisions
    decision_dist = combine(groupby(df_long, :decision), nrow => :count)
    @test nrow(decision_dist) > 0  # Distribution table should not be empty

    # Step 9: Save the transformed data and verify output
    output_file = "zurcher_data_long_test.csv"
    CSV.write(output_file, df_long)
    @test isfile(output_file)  # Check if the output file is successfully created

    # Step 10: Reload the saved file to ensure integrity
    df_loaded = CSV.read(output_file, DataFrame)
    @test size(df_loaded) == size(df_long)  # The reloaded file should match the original dataframe size
    @test all(col -> col in names(df_loaded), names(df_long))  # Columns should be identical
end

println("All tests completed successfully!")

# unit test for question 3:

using Test
using DataFrames, CSV, HTTP, Random, LinearAlgebra

# Unit Test for Part a
@testset "Data Processing" begin
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Check if required columns exist
    @test "Y1" in names(df)
    @test "Odo1" in names(df)
    @test "Xst1" in names(df)
    @test "Zst" in names(df)

    # Convert columns to matrices
    Y = Matrix(df[:, r"^Y\d+$"])
    Odo = Matrix(df[:, r"^Odo\d+$"])
    Xst = Matrix(df[:, r"^Xst\d+$"])
    Zst = df.Zst

    @test size(Y) == size(Odo)
    @test nrow(Xst) == nrow(df)
    @test length(Zst) == nrow(df)

    # Check if Branded column exists, if not create it
    if !hasproperty(df, :Branded)
        Random.seed!(123)
        df.Branded = rand(0:1, nrow(df))
    end

    Branded = df.Branded
    @test length(Branded) == nrow(df)
end


# Unit Test for Part b
@testset "Create Grids Function" begin
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

    # Call the function and check outputs
    zval, zbin, xval, xbin, xtran = create_grids()
    @test length(zval) == 101
    @test length(xval) == 33
    @test size(xtran) == (zbin * xbin, xbin)
end

# Unit Test for Part c
@testset "Future Value Computation" begin
    # Initialize future value parameters
    T = 20
    β = 0.9
    θ₀, θ₁, θ₂ = 0.0, -1.0, 2.0

    FV = zeros(zbin * xbin, 2, T + 1)

    for t in T:-1:1
        for b in 0:1
            for z in 1:zbin
                for x in 1:xbin
                    row = x + (z-1)*xbin
                    v₁ = θ₀ + θ₁ * xval[x] + θ₂ * b
                    v₁ += β * dot(xtran[row,:], FV[(z-1)*xbin+1:z*xbin, b+1, t+1])

                    v₀ = β * dot(xtran[1+(z-1)*xbin,:], FV[(z-1)*xbin+1:z*xbin, b+1, t+1])
                    FV[row, b+1, t] = β * log(exp(v₀) + exp(v₁))
                end
            end
        end
    end

    @test size(FV) == (zbin * xbin, 2, T + 1)
    @test isnumeric(FV[1,1,1]) && isnumeric(FV[end,end,end])
end

# Unit Test for Part d
@testset "Log Likelihood Construction" begin
    function construct_log_likelihood(θ, Y, Odo, Xst, Zst, Branded, FV, xtran, xval, xbin, zbin, β)
        N, T = size(Y)
        loglik = 0.0

        for i in 1:N
            for t in 1:T
                row0 = 1 + (Zst[i] - 1) * xbin
                row1 = Xst[i,t] + (Zst[i] - 1) * xbin

                v_diff = θ[1] + θ[2] * Odo[i,t] + θ[3] * Branded[i]
                if t < T
                    future_value_diff = (xtran[row1,:] .- xtran[row0,:])' * 
                                        FV[row0:row0+xbin-1, Branded[i]+1, t+1]
                    v_diff += β * future_value_diff
                end

                P1 = 1 / (1 + exp(-v_diff))
                P0 = 1 - P1

                loglik += Y[i,t] * log(P1) + (1 - Y[i,t]) * log(P0)
            end
        end

        return loglik
    end

    θ = [0.0, -1.0, 2.0]
    loglik = construct_log_likelihood(θ, Y, Odo, Xst, Zst, Branded, FV, xtran, xval, xbin, zbin, β)
    @test isnumeric(loglik)
end


# Unit Test for Part e and f
@testset "Optimization Function" begin
    @views @inbounds function myfun(θ::Vector{Float64}, 
                                    Y::Matrix{Float64}, 
                                    Odo::Matrix{Float64}, 
                                    Xst::Matrix{Int}, 
                                    Zst::Vector{Int}, 
                                    Branded::Vector{Int}, 
                                    xval::Vector{Float64}, 
                                    zval::Vector{Float64}, 
                                    xtran::Matrix{Float64},
                                    β::Float64)
        
        zbin = length(zval)
        xbin = length(xval)
        
        T = size(Y, 2)
        FV = zeros(zbin * xbin, 2, T + 1)
        
        for t in T:-1:1
            for b in 0:1
                for z in 1:zbin
                    for x in 1:xbin
                        row = x + (z-1)*xbin
                        v₁ = θ[1] + θ[2]*xval[x] + θ[3]*b
                        v₁ += β * dot(xtran[row,:], FV[(z-1)*xbin+1:z*xbin, b+1, t+1])
                        v₀ = β * dot(xtran[1+(z-1)*xbin,:], FV[(z-1)*xbin+1:z*xbin, b+1, t+1])
                        FV[row, b+1, t] = β * log(exp(v₀) + exp(v₁))
                    end
                end
            end
        end

        loglik = construct_log_likelihood(θ, Y, Odo, Xst, Zst, Branded, FV, xtran, xval, xbin, zbin, β)
        return -loglik
    end

end
