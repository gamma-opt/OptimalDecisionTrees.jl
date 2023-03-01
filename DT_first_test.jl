using JuMP
using DataFrames
using CSV
using HiGHS
using StatsBase
using LinearAlgebra

# Hyper-parameters
D = 2
N_min = 2
alpha = 0.1

# Data
data_df = CSV.read("iris_data.csv", header=false, DataFrame)
println(data_df)
data_mat = Matrix(data_df) # convert from df to matrix

# Constants
n = size(data_mat, 1)
data_width = size(data_mat, 2)
p = data_width - 1

T = 2^(D + 1) - 1
largest_B = T ÷ 2 # floor function of T/2
# T_B, T_L

# Extract X and y from data
X = data_mat[:, 1:data_width-1]
y = data_mat[:, data_width]

# Dictionary, class labels to frequencies (works only for string names of labels)
dict_names_freqs = countmap(y)
class_names = sort(collect(keys(dict_names_freqs)))
dict_names_labels = Dict([string => index for (index, string) in enumerate(class_names)])
dict_labels_names = Dict(value => key for (key, value) in dict_names_labels)
dict_labels_freqs = Dict(a => dict_names_freqs[b] for (a, b) in dict_labels_names)

# More constants
L_hat = maximum(values(dict_labels_freqs))
K = length(class_names)

# Function to calculate j'th element of epsilon vector
function epsilon_j(j)
    x_j = sort(X[:, j], rev=true) # sort x_j decreasingly
    x_j_dist = zeros(length(x_j) - 1) # init vector for distances

    for i in eachindex(x_j_dist) # calculate distances between sorted x_j's
        x_j_dist[i] = x_j[i] - x_j[i + 1] # this is different than in the book!!! i and i+1 swapped
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

# R_cursive and L_cursive
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
# calculate R_cursive and L_cursive for all nodes
R_cursive = Vector{Set{Int}}(undef, T)
L_cursive = Vector{Set{Int}}(undef, T)
for f in 1:T
    R_cursive[f] = find_R_cursive(f)
    L_cursive[f] = find_L_cursive(f)
end

function formulation(model::Model) 
    @variable(model, d[1:largest_B], Bin)           # d_t

    @variable(model, a[1:p, 1:largest_B], Bin)      # a_jt

    @variable(model, c[1:K, (largest_B+1):T], Bin)  # c_kt

    @variable(model, l[(largest_B+1):T], Bin)       # l_t  

    @variable(model, z[1:n, (largest_B+1):T], Bin)  # z_it

    @variable(model, b[t=1:largest_B] >= 0)         # b_t

    @variable(model, C)                             # C     

    @variable(model, N_t[(largest_B+1):T])          # N_t

    @variable(model, N_kt[1:K, (largest_B+1):T])    # N_kt

    @variable(model, L[(largest_B+1):T] >= 0)       # L_t


    # Constraints
    # d_T <= d_p(t)
    @constraint(model, [t = 2:largest_B], d[t] <= d[t÷2])

    # 0 <= b_t <= d_t
    @constraint(model, [t = 1:largest_B], b[t] <= d[t]) 

    # sum(a_jt) == d_t
    @constraint(model, [t = 1:largest_B], sum(a[j, t] for j in 1:p) == d[t])

    # sum(z_it >= N_min*l_t)
    @constraint(model, [t = (largest_B+1):T], sum(z[i, t] for i in 1:n) >= N_min*l[t])

    # z_it <= l_t
    @constraint(model, [i = 1:n, t = (largest_B+1):T], z[i,t] <= l[t])

    # sum(z_it == 1)
    @constraint(model, [i = 1:n], sum(z[i, t] for t in (largest_B+1):T) == 1)

    # a_m*x_i >= b_m - (1 - z_it)
    #@

    # a_m*(x_i + epsilon - epsilon_min) + epsilon_min <= b_m + (1 + epsilon_max)(1 - z_it)
    #@
    # These will be harder, since size of m varies with different values of t. Could be implemented with if-statements?

    # C == sum(d_t)
    @constraint(model, C == sum(d[t] for t in 1:largest_B))

    # sum(c_kt) == l_t
    @constraint(model, [t = (largest_B+1):T], sum(c[k,t] for k in 1:K) == l[t])

    # N_t == sum(z_it)
    @constraint(model, [t = (largest_B+1):T], N_t[t] == sum(z[i,t] for i in 1:n))

    # N_kt == sum(z_it)
    @constraint(model, [t = (largest_B+1):T, k = 1:K], N_kt[k,t] == sum(z[i,t] for i in 1:K if y[i] == k))

    # L_t <= N_t - N_kt + n*c_kt
    @constraint(model, [t = (largest_B+1):T, k = 1:K], L[t] <= N_t[t] - N_kt[k,t] + n*c[k,t])

    # L_t >= N_t - N_kt - n(1 - c_kt)
    @constraint(model, [t = (largest_B+1):T, k = 1:K], L[t] >= N_t[t] - N_kt[k,t] - n*(1 - c[k,t]))


    # Objective
    @objective(model, Min, (1/L_hat) * sum(L[t] for t in (largest_B+1):T) + alpha*C)
end


# Formulation
# Initialize optimization model
model = Model(HiGHS.Optimizer)

# Initialize variables and constraints
formulation(model)
#init_consts(model)






