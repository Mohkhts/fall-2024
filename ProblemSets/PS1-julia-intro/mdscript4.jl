using JLD

# Load the .jld file which contains matrices
data = load("firstmatrix.jld")

# Assuming the file contains matrices A and B
A = data["A"]
B = data["B"]
using JLD

# Load the matrices A and B from the JLD file
data = load("firstmatrix.jld")
A = data["A"]
B = data["B"]

# Function definition
function matrixops(A, B)
    # This function takes two matrices A and B and returns three outputs:
    # 1. The element-wise product of A and B.
    # 2. The matrix product of A and B.
    # 3. The sum of all elements of A + B.
    
    # (e) Check if matrices are the same size, otherwise raise an error
    if size(A) != size(B)
        error("inputs must have the same size.")
    end

    # (i) Element-wise product of A and B
    elementwise_product = A .* B
    
    # (ii) Matrix product of A and B
    matrix_product = A * B
    
    # (iii) Sum of all elements of A + B
    sum_elements = sum(A + B)
    
    return elementwise_product, matrix_product, sum_elements
end

# (d) Evaluate matrixops using A and B from question (a)
elementwise_product, matrix_product, sum_elements = matrixops(A, B)

# Output the results
println("Element-wise product of A and B:\n", elementwise_product)
println("\nMatrix product of A and B:\n", matrix_product)
println("\nSum of all elements of A + B:\n", sum_elements)
using JLD

# Load the matrices A and B from the JLD file
data = load("firstmatrix.jld")
A = data["A"]
B = data["B"]

# Function definition
function matrixops(A, B)
    # This function takes two matrices A and B and returns three outputs:
    # 1. The element-wise product of A and B.
    # 2. The matrix product of A and B.
    # 3. The sum of all elements of A + B.

    # Check if matrices are the same size, otherwise raise an error
    if size(A) != size(B)
        error("inputs must have the same size.")
    end

    # (i) Element-wise product of A and B
    elementwise_product = A .* B
    
    # (ii) Matrix product of A and B
    matrix_product = A * B
    
    # (iii) Sum of all elements of A + B
    sum_elements = sum(A + B)
    
    return elementwise_product, matrix_product, sum_elements
end

# Evaluate matrixops using A and B from question (a)
elementwise_product, matrix_product, sum_elements = matrixops(A, B)

# Output the results
println("Element-wise product of A and B:\n", elementwise_product)
println("\nMatrix product of A and B:\n", matrix_product)
println("\nSum of all elements of A + B:\n", sum_elements)
using JLD

# Load the matrices A and B from the JLD file
data = load("firstmatrix.jld")
A = data["A"]
B = data["B"]

# Function definition
function matrixops(A, B)
    # This function takes two matrices A and B and returns three outputs:
    # 1. The element-wise product of A and B.
    # 2. The matrix product of A and B.
    # 3. The sum of all elements of A + B.

    # Check if matrices are the same size, otherwise raise an error
    if size(A) != size(B)
        error("inputs must have the same size.")
    end

    # (i) Element-wise product of A and B
    elementwise_product = A .* B
    
    # (ii) Matrix product of A and B
    matrix_product = A * B
    
    # (iii) Sum of all elements of A + B
    sum_elements = sum(A + B)
    
    return elementwise_product, matrix_product, sum_elements
end

# (d) Evaluate matrixops using A and B from question (a)
elementwise_product, matrix_product, sum_elements = matrixops(A, B)

# Output the results
println("Element-wise product of A and B:\n", elementwise_product)
println("\nMatrix product of A and B:\n", matrix_product)
println("\nSum of all elements of A + B:\n", sum_elements)
using JLD

# Load the matrices A and B from the JLD file
data = load("firstmatrix.jld")
A = data["A"]
B = data["B"]

# Function definition
function matrixops(A, B)
    # This function takes two matrices A and B and returns three outputs:
    # 1. The element-wise product of A and B.
    # 2. The matrix product of A and B.
    # 3. The sum of all elements of A + B.

    # (e) Check if matrices are the same size, otherwise raise an error
    if size(A) != size(B)
        error("inputs must have the same size.")
    end

    # (i) Element-wise product of A and B
    elementwise_product = A .* B
    
    # (ii) Matrix product of A and B
    matrix_product = A * B
    
    # (iii) Sum of all elements of A + B
    sum_elements = sum(A + B)
    
    return elementwise_product, matrix_product, sum_elements
end

# Evaluate matrixops using A and B from question (a)
elementwise_product, matrix_product, sum_elements = matrixops(A, B)

# Output the results
println("Element-wise product of A and B:\n", elementwise_product)
println("\nMatrix product of A and B:\n", matrix_product)
println("\nSum of all elements of A + B:\n", sum_elements)
using JLD

# Load the data from the JLD file
data = load("firstmatrix.jld")

# Print available keys in the data
println("Available keys in the data: ", keys(data))

# Since only A and B are available, we will use them
A = data["A"]
B = data["B"]

# Function definition
function matrixops(A, B)
    # This function takes two matrices A and B and returns three outputs:
    # 1. The element-wise product of A and B.
    # 2. The matrix product of A and B.
    # 3. The sum of all elements of A + B.

    # Check if matrices are the same size, otherwise raise an error
    if size(A) != size(B)
        error("inputs must have the same size.")
    end

    # (i) Element-wise product of A and B
    elementwise_product = A .* B
    
    # (ii) Matrix product of A and B
    matrix_product = A * B
    
    # (iii) Sum of all elements of A + B
    sum_elements = sum(A + B)
    
    return elementwise_product, matrix_product, sum_elements
end

# Evaluate matrixops using A and B
try
    elementwise_product, matrix_product, sum_elements = matrixops(A, B)

    # Output the results
    println("Element-wise product of A and B:\n", elementwise_product)
    println("\nMatrix product of A and B:\n", matrix_product)
    println("\nSum of all elements of A + B:\n", sum_elements)
catch e
    println("An error occurred: ", e)
end
using CSV
using DataFrames

# Load the CSV file into a DataFrame
nlsw88 = CSV.read("nlsw88_processed.csv", DataFrame)

# Convert the relevant columns to Arrays
ttl_exp = convert(Array, nlsw88.ttl_exp)
wage = convert(Array, nlsw88.wage)

# Function definition (as previously defined)
function matrixops(A, B)
    # This function takes two matrices A and B and returns three outputs:
    # 1. The element-wise product of A and B.
    # 2. The matrix product of A and B.
    # 3. The sum of all elements of A + B.

    # Check if matrices are the same size, otherwise raise an error
    if size(A) != size(B)
        error("inputs must have the same size.")
    end

    # (i) Element-wise product of A and B
    elementwise_product = A .* B
    
    # (ii) Matrix product of A and B
    matrix_product = A' * B  # Matrix product (note: ' is used for transpose if necessary)
    
    # (iii) Sum of all elements of A + B
    sum_elements = sum(A + B)
    
    return elementwise_product, matrix_product, sum_elements
end

# Evaluate matrixops using ttl_exp and wage
try
    elementwise_product, matrix_product, sum_elements = matrixops(ttl_exp, wage)

    # Output the results
    println("Element-wise product of ttl_exp and wage:\n", elementwise_product)
    println("\nMatrix product of ttl_exp and wage:\n", matrix_product)
    println("\nSum of all elements of ttl_exp + wage:\n", sum_elements)
catch e
    println("An error occurred: ", e)
end
using JLD
using CSV
using DataFrames

# Define the q4() function
function q4()
    # Part (a): Load the matrices A and B from the JLD file
    data = load("firstmatrix.jld")
    A = data["A"]
    B = data["B"]

    # Part (b)-(c): Define matrixops function
    function matrixops(A, B)
        # This function takes two matrices A and B and returns three outputs:
        # 1. The element-wise product of A and B.
        # 2. The matrix product of A and B.
        # 3. The sum of all elements of A + B.

        # Check if matrices are the same size, otherwise raise an error
        if size(A) != size(B)
            error("inputs must have the same size.")
        end

        # (i) Element-wise product of A and B
        elementwise_product = A .* B

        # (ii) Matrix product of A and B
        matrix_product = A * B

        # (iii) Sum of all elements of A + B
        sum_elements = sum(A + B)

        return elementwise_product, matrix_product, sum_elements
    end

    # Part (d): Evaluate matrixops using A and B
    elementwise_product, matrix_product, sum_elements = matrixops(A, B)

    println("Results for A and B:")
    println("Element-wise product:\n", elementwise_product)
    println("Matrix product:\n", matrix_product)
    println("Sum of elements of A + B:\n", sum_elements)

    # Part (e): Error handling is already in the matrixops function

    # Part (f): Attempt to load matrices C and D, evaluate if available
    if haskey(data, "C") && haskey(data, "D")
        C = data["C"]
        D = data["D"]

        try
            elementwise_product, matrix_product, sum_elements = matrixops(C, D)

            println("\nResults for C and D:")
            println("Element-wise product:\n", elementwise_product)
            println("Matrix product:\n", matrix_product)
            println("Sum of elements of C + D:\n", sum_elements)
        catch e
            println("An error occurred with C and D: ", e)
        end
    else
        println("Matrices 'C' and 'D' are not found in the file.")
    end

    # Part (g): Evaluate matrixops using ttl_exp and wage from CSV
    nlsw88 = CSV.read("nlsw88_processed.csv", DataFrame)
    ttl_exp = convert(Array, nlsw88.ttl_exp)
    wage = convert(Array, nlsw88.wage)

    try
        elementwise_product, matrix_product, sum_elements = matrixops(ttl_exp, wage)

        println("\nResults for ttl_exp and wage:")
        println("Element-wise product:\n", elementwise_product)
        println("Matrix product:\n", matrix_product)
        println("Sum of elements of ttl_exp + wage:\n", sum_elements)
    catch e
        println("An error occurred with ttl_exp and wage: ", e)
    end
end

# Call q4() to execute the code
q4()
