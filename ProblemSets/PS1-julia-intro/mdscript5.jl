using Random
using LinearAlgebra
using Statistics
using JLD
using Test
using CSV
using DataFrames
using Distributions  # Import the Distributions package for Uniform and Normal distributions

# Define the q1() function
function q1()
    # Set the seed for reproducibility
    Random.seed!(1234)

    # Part (a) Create the matrices
    A = rand(Uniform(-5, 10), 10, 7)  # A10×7 - random numbers distributed U[−5,10]
    B = rand(Normal(-2, 15), 10, 7)   # B10×7 - random numbers distributed N(−2,15)

    # C5×7 - first 5 rows, first 5 columns of A, and last 2 columns of B
    C = [A[1:5, 1:5] B[1:5, 6:7]]

    # D10×7 - where Di,j = Ai,j if Ai,j ≤ 0, or 0 otherwise
    D = A .* (A .<= 0)

    # Part (d) Create matrix E (vec operation on B)
    E = reshape(B, :, 1)

    # Part (e) Create a 3D array F containing A and B
    F = cat(A, B, dims=3)

    # Part (f) Use permutedims to twist F from 10x7x2 to 2x10x7
    F = permutedims(F, (3, 1, 2))

    # Part (g) Create matrix G which is B⊗C (Kronecker product)
    G = kron(B, C)

    # Part (h) Save matrices A, B, C, D, E, F, and G as a .jld file
    save("matrixpractice.jld", Dict("A" => A, "B" => B, "C" => C, "D" => D, "E" => E, "F" => F, "G" => G))

    # Part (i) Save only matrices A, B, C, and D as a .jld file
    save("firstmatrix.jld", Dict("A" => A, "B" => B, "C" => C, "D" => D))

    # Part (j) Export C as a .csv file (using :auto for automatic column names)
    CSV.write("Cmatrix.csv", DataFrame(C, :auto))

    # Part (k) Export D as a tab-delimited .dat file (using :auto for automatic column names)
    CSV.write("Dmatrix.dat", DataFrame(D, :auto), delim='\t')

    # Return A, B, C, and D
    return A, B, C, D
end

# Execute q1() and save results
A, B, C, D = q1()

# Unit Tests for q1() function
@testset "Unit Tests for q1()" begin
    # Test if A is 10x7 and within the expected range
    @test size(A) == (10, 7)
    @test all(A .>= -5) && all(A .<= 10)

    # Test if B is 10x7 and follows the normal distribution properties
    @test size(B) == (10, 7)
    @test mean(B) ≈ -2.0 atol=2.0  # Allowing some tolerance due to randomness
    @test std(B) ≈ 15.0 atol=2.0

    # Test if C is 5x7 and correctly composed from A and B
    @test size(C) == (5, 7)
    @test C[:, 1:5] == A[1:5, 1:5]
    @test C[:, 6:7] == B[1:5, 6:7]

    # Test if D is 10x7 and contains only non-positive elements
    @test size(D) == (10, 7)
    @test all(D .<= 0)

    # Test if the saved files exist
    @test isfile("matrixpractice.jld")
    @test isfile("firstmatrix.jld")
    @test isfile("Cmatrix.csv")
    @test isfile("Dmatrix.dat")
end
using Distributions
using Test

# Assuming A, B, and C are already defined from previous questions

function q2(A, B, C)
    # Part (a) Element-by-element product of A and B
    AB = [A[i, j] * B[i, j] for i in 1:size(A, 1), j in 1:size(A, 2)]

    # Matrix multiplication for element-by-element product without a loop
    AB2 = A .* B

    # Part (b) Create a column vector Cprime with elements of C between -5 and 5 (inclusive)
    Cprime = []
    for i in 1:size(C, 1)
        for j in 1:size(C, 2)
            if -5 <= C[i, j] <= 5
                push!(Cprime, C[i, j])
            end
        end
    end

    # Create Cprime2 without a loop
    Cprime2 = vec(C[(C .>= -5) .& (C .<= 5)])

    # Part (c) Create a 3-dimensional array X of dimension N×K×T
    N = 15169
    K = 6
    T = 5
    X = Array{Float64}(undef, N, K, T)
    dummy_variable = Array{Int64}(undef, N, T)  # Store dummy variable for consistent testing
    
    for t in 1:T
        X[:, 1, t] .= 1.0  # Intercept (all ones)
        dummy_variable[:, t] = rand(Binomial(1, 0.75*(6-t)/5), N)  # Dummy variable
        X[:, 2, t] .= dummy_variable[:, t]
        X[:, 3, t] .= rand(Normal(15 + t - 1, 5*(t - 1)), N)  # Continuous variable
        X[:, 4, t] .= rand(Normal(π*(6-t)/3, 1/exp(1)), N)  # Continuous variable
        X[:, 5, t] .= rand(Binomial(20, 0.6), N)  # Discrete normal variable
        X[:, 6, t] .= rand(Binomial(20, 0.5), N)  # Binomial variable
    end
    
    return AB, AB2, Cprime, Cprime2, X, dummy_variable
end

# Execute q2() and store results
AB, AB2, Cprime, Cprime2, X, dummy_variable = q2(A, B, C)

# Tests for the q2() function
@testset "Unit Tests for q2()" begin
    # Test if AB and AB2 are identical
    @test AB == AB2
    
    # Test if Cprime and Cprime2 are identical (order-insensitive comparison)
    @test sort(Cprime) == sort(Cprime2)
    
    # Test dimensions of X
    @test size(X) == (15169, 6, 5)
    
    # Test if the intercept column in X is all ones
    @test all(X[:, 1, :] .== 1.0)
    
    # Test if the dummy variable is consistent with what was generated in the function
    @test all(X[:, 2, 1] .== dummy_variable[:, 1])
    
    # Test mean and std of continuous variable in X[:, 3, t] and X[:, 4, t]
    for t in 1:5
        @test mean(X[:, 3, t]) ≈ (15 + t - 1) atol=1.0
        @test std(X[:, 3, t]) ≈ (5*(t - 1)) atol=1.0
        @test mean(X[:, 4, t]) ≈ (π*(6-t)/3) atol=1.0
        @test std(X[:, 4, t]) ≈ (1/exp(1)) atol=0.1
    end
    
    # Test if X[:, 5, :] and X[:, 6, :] have values within expected range
    @test all(X[:, 5, :] .>= 0) && all(X[:, 5, :] .<= 20)
    @test all(X[:, 6, :] .>= 0) && all(X[:, 6, :] .<= 20)
end
using JLD
using LinearAlgebra
using Test
using CSV
using DataFrames

# Load matrices from firstmatrix.jld
function load_matrices()
    data = load("firstmatrix.jld")
    return data["A"], data["B"], data["C"], data["D"]
end

# Function to perform matrix operations
function matrixops(A, B)
    # Check if inputs have the same size
    if size(A) != size(B)
        error("inputs must have the same size")
    end

    # Element-by-element product of A and B
    elementwise_product = A .* B
    
    # Matrix product A'B
    matrix_product = A' * B
    
    # Sum of all elements in A + B
    sum_elements = sum(A + B)
    
    return elementwise_product, matrix_product, sum_elements
end

# Function to perform operations and run tests
function q4()
    # Part (a) Load matrices from firstmatrix.jld
    A, B, C, D = load_matrices()

    # Part (b) Evaluate matrixops() using A and B
    elementwise_product, matrix_product, sum_elements = matrixops(A, B)

    # Part (f) Evaluate matrixops() using C and D
    try
        elementwise_product_CD, matrix_product_CD, sum_elements_CD = matrixops(C, D)
    catch e
        println("Error evaluating matrixops with C and D: ", e)
    end

    # Part (g) Evaluate matrixops() using ttl_exp and wage from nlsw88_processed.csv
    nlsw88 = CSV.read("nlsw88_processed.csv", DataFrame)
    
    # Check if required columns exist
    if :ttl_exp in names(nlsw88) && :wage in names(nlsw88)
        ttl_exp = convert(Array, nlsw88.ttl_exp)
        wage = convert(Array, nlsw88.wage)

        try
            elementwise_product_exp_wage, matrix_product_exp_wage, sum_elements_exp_wage = matrixops(ttl_exp, wage)
        catch e
            println("Error evaluating matrixops with ttl_exp and wage: ", e)
        end
    else
        println("Required columns ttl_exp or wage not found in the DataFrame.")
    end
end

# Run the q4 function
q4()

# Unit Tests for q4() function
@testset "Unit Tests for q4()" begin
    # Load matrices for testing
    A, B, C, D = load_matrices()

    # Test matrixops with A and B
    elementwise_product, matrix_product, sum_elements = matrixops(A, B)
    
    # Test the shape of the elementwise product
    @test size(elementwise_product) == size(A)
    
    # Test if the matrix product has the expected shape
    @test size(matrix_product) == (size(A, 2), size(B, 2))
    
    # Test if the sum of elements is a scalar
    @test isa(sum_elements, Number)

    # Test matrixops with C and D - should raise an error because of size mismatch
    @test_throws ErrorException matrixops(C, D)

    # Test matrixops with ttl_exp and wage from nlsw88_processed.csv
    nlsw88 = CSV.read("nlsw88_processed.csv", DataFrame)
    if :ttl_exp in names(nlsw88) && :wage in names(nlsw88)
        ttl_exp = convert(Array, nlsw88.ttl_exp)
        wage = convert(Array, nlsw88.wage)

        # Since ttl_exp and wage might not be the same size, test should handle that
        if size(ttl_exp) == size(wage)
            elementwise_product_exp_wage, matrix_product_exp_wage, sum_elements_exp_wage = matrixops(ttl_exp, wage)
            
            # Test the shape of the elementwise product
            @test size(elementwise_product_exp_wage) == size(ttl_exp)
            
            # Test if the matrix product has the expected shape
            @test size(matrix_product_exp_wage) == (size(ttl_exp, 2), size(wage, 2))
            
            # Test if the sum of elements is a scalar
            @test isa(sum_elements_exp_wage, Number)
        else
            @test_throws ErrorException matrixops(ttl_exp, wage)
        end
    else
        println("Required columns ttl_exp or wage not found in the DataFrame.")
    end
end
