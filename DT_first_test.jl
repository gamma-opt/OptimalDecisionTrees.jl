using JuMP
using DataFrames
using CSV
using HiGHS

# Hyper-parameters
D = 2
N_min = 2
alpha = 0.1

# Data
data_df = DataFrame(CSV.File("iris_data.csv"))
println(df)
data_mat = Matrix(data_df) # convert from df to matrix

# Constants
n = size(data_mat, 1)
data_width = size(data_mat, 2)
p = data_width - 1

T = 2^(D + 1) - 1
largest_branch = T ÷ 2 # floor function of T/2

#L_hat = 

# Extract X and y from data
X = data_mat[:, 1:data_width-1]
y = data_mat[:, data_width]

# Function to calculate j'th element of epsilon vector
function epsilon_j(j)
    x_j = sort(x[:, j], rev=true) # sort x_j decreasingly
    x_j_dist = zeros(length(x_j) - 1) # init vector for distances

    for i in eachindex(x_j_dist) # calculate distances between sorted x_j's
        x_j_dist[i] = x_j[i] - x_j[i + 1] # this is different than in the book!!! i and i+1 swapped
                                          # Do you agree that this seems logical?
    end

    nz_x_j_dist = x_j_dist[x_j_dist .> 0] # remove zero distances
    return  findmin(nz_x_j_dist)[1] # find and return minimum dist
end

# epsilon and epsilon_min
epsilon = zeros(p) # init vector
for j in 1:p # for all features
    epsilon[j] = epsilon_j(j)
end
epsilon_min = findmin(epsilon)[1]

# Functions to calculate R_cursive and L_cursive
function find_R_cursive(t)
    temp_set = Set() # init empty set
    while t != 1 # when not in root node
        next_t = t ÷ 2 # calculate parent node
        if t % 2 == 1 # if node is on right branch..
            push!(temp_set, next_t) # ..add its parent node to set
        end
        t = next_t # update next node (parent node)
    end
    return temp_set 
end

function find_L_cursive(t)
    temp_set = Set()
    while t != 1
        next_t = t ÷ 2
        if t % 2 == 0 # if node is on left branch
            push!(temp_set, next_t)
        end
        t = next_t
    end
    return temp_set
end

# R_cursive and L_cursive vectors to hold info for all nodes
R_cursive = Vector{Set{Int}}(undef, T)
L_cursive = Vector{Set{Int}}(undef, T)
for f in 1:T
    R_cursive[f] = find_R_cursive(f)
    L_cursive[f] = find_L_cursive(f)
end







