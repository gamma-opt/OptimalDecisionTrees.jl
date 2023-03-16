using JuMP
using DataFrames
using CSV
using HiGHS
using StatsBase
using LinearAlgebra
using DecisionTree


# Hyper-parameters
D = 2 # Maximum depth of the tree
N_min = 2 # Minimum number of points in any leaf node
alpha = 0.1 # complexity parameter 

# Data
data_df = CSV.read("iris_data.csv", header=false, DataFrame)
println(data_df)
data_mat = Matrix(data_df) # convert from df to matrix

# Constants
n = size(data_mat, 1) # number of observations
data_width = size(data_mat, 2)
p = data_width - 1 # number of features

T = 2^(D + 1) - 1 # Maximum number of nodes in the tree of depth D 
largest_B = T ÷ 2 # floor function of T/2 - number of branch nodes
# T_B, T_L

# Extract X and y from data
X = data_mat[:, 1:data_width-1]
# Normalize x
max_digits = 3
for i = 1:size(X,1)
    X[i,:] = round.(X[i,:]/sum(X[i,:]), digits = max_digits)
end

y = data_mat[:, data_width]

# Dictionary, class labels to frequencies (works only for string names of labels)
dict_names_freqs = countmap(y)
class_names = sort(collect(keys(dict_names_freqs)))
dict_names_labels = Dict([string => index for (index, string) in enumerate(class_names)])
dict_labels_names = Dict(value => key for (key, value) in dict_names_labels)
dict_labels_freqs = Dict(a => dict_names_freqs[b] for (a, b) in dict_labels_names)
y_labels = [dict_names_labels[y[i]] for i = 1:n]

function Y(i::Int, k::Int, dataset::Vector{Any}) 
    if dataset[i] == k return 1
    else return -1
    end
end


# More constants
L_hat = maximum(values(dict_labels_freqs)) # Number of points in the most popular class 
K = length(class_names) # Total number of classes

# Function to calculate j'th element of epsilon vector
function epsilon_j(j)
    x_j = sort(X[:, j], rev=true) # sort x_j decreasingly
    x_j_dist = zeros(length(x_j) - 1) # init vector for distances

    for i in eachindex(x_j_dist) # calculate distances between sorted x_j's
        x_j_dist[i] = x_j[i] - x_j[i + 1] # this is different than in the book!!! i and i+1 swapped
    end

    nz_x_j_dist = x_j_dist[x_j_dist .> 0] # remove zero distances
    @show nz_x_j_dist
    return  findmin(nz_x_j_dist)[1] # find and return minimum dist
end

# epsilon and epsilon_min
epsilon = zeros(p) # init vector
for j in 1:p # for all features
    epsilon[j] = epsilon_j(j)
end
#epsilon_min = findmin(epsilon)[1]

# # R_cursive and L_cursive
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



# function to generate set of ancestor node of t who followed left (A_l) and right (A_l) branches from the root node to the node t
function ancestors_LR(t::Int)
    A_l = Array{Int}(undef,0)
    A_r = Array{Int}(undef,0)
    current_node = t
    while current_node != 1
        current_parent = current_node÷2
        if current_node % 2 == 0
            push!(A_l, current_parent)
        else 
            push!(A_r, current_parent)
        end
        current_node = current_parent
    end
    return A_l, A_r

end


function formulation(X,y) 

    model = Model(HiGHS.Optimizer)

    @variable(model, d[1:largest_B], Bin)           # d_t - indicator whether the split occured at node t (d_t = 1)

    @variable(model, a[1:p, 1:largest_B], Bin)      # a_jt - left-hands side of splitting condition

    @variable(model, b[t=1:largest_B] >= 0)         # b_t - right-hand side of splitting condition

    @variable(model, c[1:K, (largest_B+1):T], Bin)  # c_kt - predicition of each leaf node, i.e., c_kt = 1 => the node t has more points of class k

    @variable(model, l[(largest_B+1):T], Bin)       # l_t  - indicator whther leaf t contains any points => l_t = 1

    @variable(model, z[1:n, (largest_B+1):T], Bin)  # z_it - the indicator to track points assigned to each leaf node ( point i is at the node t => z_it = 1)


    #@variable(model, C)                             # C    

    @variable(model, N_t[(largest_B+1):T])          # N_t - total number of points at leaf node t

    @variable(model, N_kt[1:K, (largest_B+1):T])    # N_kt - number of points of label k at leaf node t 

    @variable(model, L[(largest_B+1):T] >= 0)       # L_t - miscalssification loss at leaf node t


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
    for t = (largest_B +1):T
        t_A_l, t_A_r = ancestors_LR(t)
        # a_m*x_i >= b_m - (1 - z_it)
        @show t
        @show t_A_l, t_A_r
        if !isempty(t_A_r)
            @constraint(model, [i = 1:n, m in t_A_r], sum(a[:, m].* X[i, :]) >= b[m] - (1  - z[i,t]))
        end
        if !isempty(t_A_l)
            @constraint(model, [i = 1:n, m in t_A_l], sum(a[:, m].* (X[i, :] .+ epsilon)) <= b[m] + (1 + maximum(epsilon))*(1 - z[i,t]))
        end
    end

    # C == sum(d_t)
    #@constraint(model, C == sum(d[t] for t in 1:largest_B))

    # sum(c_kt) == l_t
    @constraint(model, [t = (largest_B+1):T], sum(c[k,t] for k in 1:K) == l[t])

    # N_t == sum(z_it)
    @constraint(model, [t = (largest_B+1):T], N_t[t] == sum(z[i,t] for i in 1:n))

    # N_kt == sum(z_it)
    #@constraint(model, [t = (largest_B+1):T, k = 1:K], N_kt[k,t] == sum(z[i,t] for i in 1:K if y[i] == k))

    # N_kt == sum(z_it)
    @constraint(model, [t = (largest_B+1):T, k = 1:K], N_kt[k,t] == 0.5 * sum((1 + Y(i,k,y))*z[i,t] for i in 1:n))

    # L_t <= N_t - N_kt + n*c_kt
    @constraint(model, [t = (largest_B+1):T, k = 1:K], L[t] <= N_t[t] - N_kt[k,t] + n*c[k,t])

    # L_t >= N_t - N_kt - n(1 - c_kt)
    @constraint(model, [t = (largest_B+1):T, k = 1:K], L[t] >= N_t[t] - N_kt[k,t] - n*(1 - c[k,t]))

    # Objective
    @objective(model, Min, (1/L_hat) * sum(L[t] for t in (largest_B+1):T) + alpha*sum(d[t] for t in 1:largest_B))

    return model
end


# Initialize optimization model
model=formulation(X,y)
optimize!(model)

value.(model[:a])
value.(model[:b])
z_output = Array(value.(model[:z]))

# Trying out CART 
features, labels = load_data("iris")    # also see "adult" and "digits" datasets

# the data loaded are of type Array{Any}
# cast them to concrete types for better performance
features = float.(features)
labels   = string.(labels)
# train full-tree classifier
n_subfeatures=1; max_depth=2; min_samples_leaf=1; min_samples_split=2
min_purity_increase=0.0; pruning_purity = 1.0; seed=3
model    =   build_tree(labels, features,
                        n_subfeatures,
                        max_depth,
                        min_samples_leaf,
                        min_samples_split,
                        min_purity_increase;
                        rng = seed)
print_tree(model, 2)
