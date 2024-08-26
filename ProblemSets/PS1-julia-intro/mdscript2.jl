# Import necessary packages
using Random
using LinearAlgebra
using JLD2
using DataFrames
using CSV
using DelimitedFiles
using Distributions

# Function q1: Initializes variables and creates matrices
function q1()
    # Set the seed for reproducibility
    Random.seed!(1234)

    # i. A 10x7 matrix with U[-5, 10] distribution
    A = rand(-5:10, 10, 7)

    # ii. B 10x7 matrix with N(-2, 15) distribution
    B = randn(10, 7) * 15 .+ (-2)

    # iii. C 5x7 matrix from specific rows and columns of A and B
    C = [A[1:5, 1:5] B[1:5, 6:7]]

    # iv. D 10x7 matrix where D_ij = A_ij if A_ij <= 0, otherwise 0
    D = map(x -> x <= 0 ? x : 0, A)

    # (b) Number of elements in A
    num_elements_A = length(A)
    println("Number of elements in A: ", num_elements_A)

    # (c) Number of unique elements in D
    num_unique_elements_D = length(unique(D))
    println("Number of unique elements in D: ", num_unique_elements_D)

    # (d) Reshape B into a vector (vec) to create matrix E
    E = reshape(B, :)

    # (e) Create a 3-dimensional array F
    F = cat(A, B, dims=3)

    # (f) Use permutedims to reshape F
    F = permutedims(F, (2, 1, 3))

    # (g) Kronecker product of B and C to create matrix G
    G = kron(B, C)

    # (h) Save matrices A, B, C, D, E, F, G to a .jld file
    @save "matrixpractice.jld2" A B C D E F G

    # (i) Save only A, B, C, D to another .jld file
    @save "firstmatrix.jld2" A B C D

    # (j) Export C as a .csv file
    C_df = DataFrame(C, :auto)
    CSV.write("Cmatrix.csv", C_df)

    # (k) Export D as a tab-delimited .dat file
    D_df = DataFrame(D, :auto)
    CSV.write("Dmatrix.dat", D_df; delim='\t')

    # Return the matrices A, B, C, D
    return A, B, C, D
end

# Function q2: Practice with loops and comprehensions
function q2(A, B, C)
    # (a) Element-by-element product of A and B using a comprehension
    AB = [A[i, j] * B[i, j] for i in 1:size(A, 1), j in 1:size(A, 2)]
    println("AB matrix:")
    println(AB)

    # Without a loop or comprehension
    AB2 = A .* B

    # (b) Create Cprime using a loop that contains elements of C between -5 and 5
    Cprime = []
    for i in 1:size(C, 1), j in 1:size(C, 2)
        if -5 <= C[i, j] <= 5
            push!(Cprime, C[i, j])
        end
    end
    println("Cprime vector:")
    println(Cprime)

    # Without a loop
    Cprime2 = C[(C .>= -5) .& (C .<= 5)]

    # (c) Create a 3D array X of dimensions N x K x T
    N = 15_169
    K = 6
    T = 5
    X = Array{Float64}(undef, N, K, T)

    for t in 1:T
        X[:, 1, t] = ones(N)  # Intercept
        X[:, 2, t] = rand(Binomial(1, 0.75 * (6 - t) / 5), N)  # Dummy variable
        X[:, 3, t] = randn(N) .* (5 * (t - 1)) .+ (15 + t - 1)  # Continuous variable with mean and std dev
        X[:, 4, t] = randn(N) .* (1 / exp(1)) .+ sqrt((6 - t) / 3)  # Another continuous variable
        X[:, 5, t] = rand(Binomial(20, 0.6), N)  # Discrete normal
        X[:, 6, t] = rand(Binomial(20, 0.5), N)  # Discrete binomial
    end
    println("3D Array X:")
    println(X)

    # (d) Create matrix b which is K x T
    b = [i == 1 ? 1 + 0.25 * (t - 1) : 
        i == 2 ? log(t) :
        i == 3 ? -sqrt(t) : 
        i == 4 ? exp(t) - exp(t + 1) :
        i == 5 ? t :
        t / 3 for i in 1:K, t in 1:T]
    println("Matrix b:")
    println(b)

    # (e) Create matrix Y which is N x T defined by Yt = Xt * bt + et
    Y = [X[:, :, t] * b[:, t] .+ randn(N) .* sqrt(0.36) for t in 1:T]
    println("Matrix Y:")
    println(Y)

    # Since Y is a 2D matrix over time, we need to concatenate them horizontally
    Y = hcat(Y...)

    # Function returns nothing, following instructions
    return nothing
end

# Ensure that q2() gets called after q1()
A, B, C, D = q1()  # Call q1 to generate A, B, C, D
q2(A, B, C)  # Call q2 with A, B, C
