using Pkg
Pkg.add("CSV")
using CSV
using CSV
using DataFrames

# Read the CSV file and appropriately handle missing values
nlsw88 = CSV.read("C:/Users/mdkha/fall-2024/ProblemSets/PS1-julia-intro/nlsw88.csv", DataFrame; missingstring="NA")

# Rest of your code follows...
using CSV
using DataFrames

# Read the CSV file and appropriately handle missing values
nlsw88 = CSV.read("C:/Users/mdkha/fall-2024/ProblemSets/PS1-julia-intro/nlsw88.csv", DataFrame; missingstring="NA")

# Convert variable names to lowercase
rename!(nlsw88, lowercase.(names(nlsw88)))

# Save the processed DataFrame as a new CSV file
CSV.write("nlsw88_processed.csv", nlsw88)

println("Processed DataFrame saved as 'nlsw88_processed.csv'.")
using CSV
using DataFrames

# Read the processed CSV file
nlsw88 = CSV.read("nlsw88_processed.csv", DataFrame)

# Calculate percentage of the sample that has never been married
never_married_percentage = sum(nlsw88.married .== 0) / nrow(nlsw88) * 100

# Calculate percentage of college graduates (assuming college graduates have 16 or more years of education)
college_graduates_percentage = sum(nlsw88.grade .>= 16) / nrow(nlsw88) * 100

println("Percentage never married: ", round(never_married_percentage, digits=2), "%")
println("Percentage of college graduates: ", round(college_graduates_percentage, digits=2), "%")
using CSV
using DataFrames
using FreqTables

# Read the processed CSV file
nlsw88 = CSV.read("nlsw88_processed.csv", DataFrame)

# Create a frequency table for race
race_freq_table = freqtable(nlsw88, :race)

# Convert the frequency table to percentages
race_percentage = 100 * race_freq_table ./ nrow(nlsw88)

println("Race Percentage Table: ")
println(race_percentage)
using FreqTables
using CSV
using DataFrames
using FreqTables  # Load FreqTables after installation

# Read the processed CSV file
nlsw88 = CSV.read("nlsw88_processed.csv", DataFrame)

# Create a frequency table for race
race_freq_table = freqtable(nlsw88, :race)

# Convert the frequency table to percentages
race_percentage = 100 * race_freq_table ./ nrow(nlsw88)

println("Race Percentage Table: ")
println(race_percentage)
using CSV
using DataFrames
using Statistics

# Read the processed CSV file
nlsw88 = CSV.read("nlsw88_processed.csv", DataFrame)

# Generate summary statistics
summarystats = describe(nlsw88, :mean, :median, :std, :min, :max, :nunique)

# Check how many grade observations are missing
missing_grade_count = sum(ismissing.(nlsw88.grade))

println("Summary Statistics: ")
println(summarystats)
println("Number of missing grade observations: ", missing_grade_count)
using CSV
using DataFrames

# Read the processed CSV file
nlsw88 = CSV.read("nlsw88_processed.csv", DataFrame)

# Cross-tabulation of industry and occupation
industry_occupation_crosstab = combine(groupby(nlsw88, [:industry, :occupation]), nrow => :count)

println("Joint Distribution of Industry and Occupation: ")
println(industry_occupation_crosstab)
using CSV
using DataFrames
using Statistics

# Read the processed CSV file
nlsw88 = CSV.read("nlsw88_processed.csv", DataFrame)

# Subset the data frame to only include industry, occupation, and wage
subset_df = select(nlsw88, :industry, :occupation, :wage)

# Calculate the mean wage by industry and occupation
mean_wage_by_category = combine(groupby(subset_df, [:industry, :occupation]), :wage => mean => :mean_wage)

println("Mean Wage by Industry and Occupation: ")
println(mean_wage_by_category)
using CSV
using DataFrames
using Statistics

# Read the processed CSV file
nlsw88 = CSV.read("nlsw88_processed.csv", DataFrame)

# Subset the data frame to only include industry, occupation, and wage
subset_df = select(nlsw88, :industry, :occupation, :wage)

# Calculate the mean wage by industry and occupation
mean_wage_by_category = combine(groupby(subset_df, [:industry, :occupation]), :wage => mean => :mean_wage)

println("Mean Wage by Industry and Occupation: ")
println(mean_wage_by_category)
