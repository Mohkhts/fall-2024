using DataFrames
using CSV

function q1()
    # (a) Create the matrices with random numbers
    Random.seed!(1234) # Set seed
    
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
    
    # Easier way (the vec function)
    E_alternative = vec(B)
    
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

# Execute the function and get the matrices A, B, C, D
A, B, C, D = q1()
