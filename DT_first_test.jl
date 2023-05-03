current_dir =  @__DIR__
using Pkg
cd(current_dir)
Pkg.instantiate()

using JuMP
using DataFrames
using CSV
using HiGHS
using StatsBase
using LinearAlgebra
using DecisionTree
using Gurobi

# Mode of optimization
mode = 2
# 1 = Complexity parameter
# 2 = Complexity parameter with CART warm start
# 3 = Fixed number of splits


N_min_perc = 0.1 # Percentage of N_min from all data points
C_perc = 0.8 # Percentage of splits from all possible

# Hyper-parameters
D = 2 # Maximum depth of the tree
alpha = 0.0 # Complexity parameter

# Data
data_df = CSV.read("iris_data.csv", header=false, DataFrame)
println(data_df)
data_mat = Matrix(data_df) # convert from df to matrix
max_digits = 3


# the function to trim the data to only two observations with two different labels if needed (the second parameter then = "reduced")
# or work with full data (the second parameter is then "full")
function data_generation(data, new_size)
    if new_size == "reduced"
        new_data = Array{Any}(undef, 2, size(data, 2))
        new_data[1,:] = data[1,:]
        new_data[2,:] = data[51,:]
    else 
        new_data = data
    end
    # Constants
    n = size(new_data, 1) # number of observations
    data_width = size(new_data, 2)
    p = data_width - 1 # number of features

    T = 2^(D + 1) - 1 # Maximum number of nodes in the tree of depth D 
    largest_B = T ÷ 2 # floor function of T/2 - number of branch nodes
    # T_B, T_L

    # Extract X and y from data
    X = new_data[:, 1:data_width-1]
    # Normalize x
    for i = 1:size(X,2)
        X[:,i] = round.((X[:,i] .- minimum(X[:,i]))./(maximum(X[:,i])-minimum(X[:,i])), digits = max_digits)
    end
    #X= [X[1,:]'; X[51,:]']

    y = new_data[:, data_width]
    #y = [y[1], y[51]]
    return X,y,n,p,T,largest_B 
end 

X,y,n,p,T,largest_B = data_generation(data_mat, "full")
N_min = round(Int, N_min_perc*n) # Minimum number of points in any leaf node


# Dictionary, class labels to frequencies (works only for string names of labels)
dict_names_freqs = countmap(y)
class_names = sort(collect(keys(dict_names_freqs)))
dict_names_labels = Dict([string => index for (index, string) in enumerate(class_names)])
dict_labels_names = Dict(value => key for (key, value) in dict_names_labels)
dict_labels_freqs = Dict(a => dict_names_freqs[b] for (a, b) in dict_labels_names)
y_labels = [dict_names_labels[y[i]] for i = 1:n]

# More constants
L_hat = maximum(values(dict_labels_freqs)) # Number of points in the most popular class 
K = length(class_names) # Total number of classes

# Function to calculate j'th element of epsilon vector
function epsilon_j(j)
    x_j = sort(X[:, j], rev=true) # sort x_j decreasingly
    x_j_dist = zeros(length(x_j) - 1) # init vector for distances

    for i in eachindex(x_j_dist) # calculate distances between sorted x_j's
        x_j_dist[i] = x_j[i] - x_j[i + 1] # this is different than in the book!!! i and i+1 swapped. Rounding here?
    end

    # rounding to prevent errors in comparing
    x_j_dist = round.(x_j_dist, digits = max_digits)

    nz_x_j_dist = x_j_dist[x_j_dist .> 0] # remove zero distances
    #nz_x_j_dist = round.(nz_x_j_dist, digits = max_digits ) #trim data
    @show nz_x_j_dist
    if !isempty(nz_x_j_dist)
        return  findmin(nz_x_j_dist)[1]  # find and return minimum dist
    else return 0
    end
end

# epsilon and eps_min and eps_max
epsilon = zeros(p) # init vector
for j in 1:p # for all features
    epsilon[j] = epsilon_j(j)
end
eps_min = minimum(epsilon)
eps_max = maximum(epsilon)

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


function formulation(X,y, a_s, d_s, z_s, l_s, c_s) 
    model = Model(Gurobi.Optimizer)

    @variable(model, d[1:largest_B], Bin)           # d_t - indicator whether the split occured at node t (d_t = 1)

    @variable(model, a[1:p, 1:largest_B], Bin)      # a_jt - left-hands side of splitting condition

    @variable(model, b[t=1:largest_B] >= 0)         # b_t - right-hand side of splitting condition

    @variable(model, c[1:K, (largest_B+1):T], Bin)  # c_kt - predicition of each leaf node, i.e., c_kt = 1 => the node t has more points of class k

    @variable(model, l[(largest_B+1):T], Bin)       # l_t  - indicator whther leaf t contains any points => l_t = 1

    @variable(model, z[1:n, (largest_B+1):T], Bin)  # z_it - the indicator to track points assigned to each leaf node ( point i is at the node t => z_it = 1)


    @variable(model, C)                             # C - number of splits included in the tree
    if mode == 3 # If fixed num of splits
        fix(C, C_perc*largest_B)
    end

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

    for t = (largest_B +1):T
        t_A_l, t_A_r = ancestors_LR(t)
        
        @show t
        @show t_A_l, t_A_r
        if !isempty(t_A_r)
            # a_m*x_i >= b_m - (1 - z_it)
            @constraint(model, [i = 1:n, m in t_A_r], map_coefficients_inplace!(a -> round(a, digits=3), sum(a[:, m].* X[i, :]) - b[m] + (1  - z[i,t])) >= 0)
        end
        if !isempty(t_A_l)
            @show  t_A_l
            # a_m^T(x_i + epsilon - eps_min) + eps_min >= b_m + (1 + eps_max)(1 - z_it)
            @constraint(model, [i = 1:n, m in t_A_l], map_coefficients_inplace!(a -> round(a, digits=3), sum(a[:, m].* (X[i, :] .+ epsilon .- eps_min)) + eps_min - b[m] - (1 + eps_max)*(1 - z[i,t])) <= 0)
        end
    end

    # C == sum(d_t)
    if mode != 3
        @constraint(model, C == sum(d[t] for t in 1:largest_B)) # Complexity parameter
    else
        @constraint(model, sum(d[t] for t in 1:largest_B) <= C) # Fixed num of splits
    end

    # sum(c_kt) == l_t
    @constraint(model, [t = (largest_B+1):T], sum(c[k,t] for k in 1:K) == l[t])

    # N_t == sum(z_it)
    @constraint(model, [t = (largest_B+1):T], N_t[t] == sum(z[i,t] for i in 1:n))

    # N_kt == sum(z_it)
    for k = 1:K
        ind_y_i_k = findall(x->x==k, y_labels)
        @show k
        @show  ind_y_i_k
        @constraint(model, [t = (largest_B+1):T], N_kt[k,t] == sum(z[i,t] for i in ind_y_i_k ))
    end

    # L_t <= N_t - N_kt + n*c_kt
    @constraint(model, [t = (largest_B+1):T, k = 1:K], L[t] <= N_t[t] - N_kt[k,t])

    # L_t >= N_t - N_kt - n(1 - c_kt)
    @constraint(model, [t = (largest_B+1):T, k = 1:K], L[t] >= N_t[t] - N_kt[k,t] - n*(1 - c[k,t]))

    # Objective
    if mode != 3
        @objective(model, Min, (1/L_hat) * sum(L[t] for t in (largest_B+1):T) + alpha*C) # Complexity parameter
    else
        @objective(model, Min, (1/L_hat) * sum(L[t] for t in (largest_B+1):T)) # Fixed num of splits
    end

    if mode == 2
        set_start_value.(a, a_s)
        set_start_value.(d, d_s)
        set_start_value.(z, z_s)
        set_start_value.(l, l_s)
        set_start_value.(c, c_s)
    end

    return model
end

# CART
println("CART:")
n_subfeatures=1; max_depth=D; min_samples_leaf=N_min; min_samples_split=max(2,N_min)
min_purity_increase=0.0; pruning_purity = 1.0; seed=3
model2    =   build_tree(y_labels, X,
                        n_subfeatures,
                        max_depth,
                        min_samples_leaf,
                        min_samples_split,
                        min_purity_increase;
                        rng = seed)
print_tree(model2, D) # Print CART 
println()


# Extract nodes from CART output
nd = Vector{Any}(undef, T)
nd[1] = model2.node

a2 = zeros(p, largest_B)
b2 = zeros(largest_B)
d2 = zeros(largest_B)

for i in 1:largest_B # go trough every branch node
    if(typeof(nd[i]) == Node{Any, Int64}) # if is branch node in CART
        nd[i*2] = nd[i].left # assign childs
        nd[i*2 + 1] = nd[i].right

        a2[nd[i].featid, i] = 1 # set splitting feature
        b2[i] = nd[i].featval # set splitting value
        d2[i] = 1 # set split bool
    else # else assign 0
        nd[i*2] = 0
        nd[i*2 + 1] = 0
    end
end

preds = apply_tree(model2, X)
z_cart = zeros(150,4)

println("Misclassified data points:")
for i in 1:50
    if(preds[i] == 1) 
        z_cart[i,2] = 1
    else 
        println(i)
    end
end

for i in 51:100
    if(preds[i] == 2) 
        z_cart[i,3] = 1
    else
        println(i)
    end
end

for i in 101:150
    if(preds[i] == 3) 
        z_cart[i,4] = 1
    else
        println(i)
    end
end


# Setting warm start values from CART output manually

# Misclassified data points
# 53 = 3
z_cart[53, 4] = 1
# 71 = 3 
z_cart[71, 4] = 1
# 73 = 3
z_cart[73, 4] = 1
# 77 = 3
z_cart[77, 4] = 1
# 78 = 3
z_cart[78, 4] = 1
# 84 = 3
z_cart[84, 4] = 1
# 107= 2
z_cart[107, 3] = 1

l2 = [0, 1, 1, 1]

c2 = zeros(3, 4) # set zeros
c2[1, 2] = 1 
c2[2, 3] = 1 
c2[3, 4] = 1 

N_t2 = [0, 50, 45, 55]

N_kt2 = zeros(3, 4) # set zeros
N_kt2[1,2] = 50 
N_kt2[2,3] = 44 
N_kt2[3,4] = 49 

N_kt2[2,4] = 6 #
N_kt2[3,3] = 1 # 

L2 = [0, 0, 1, 6] 

C2 = 2


# Initialize optimization model
model=formulation(X,y, a2, d2, z_cart, l2, c2)
#print(model)
optimize!(model)

a = value.(model[:a])
b = value.(model[:b])
d = value.(model[:d])
z_output = Array(value.(model[:z]))
L_output = Array(value.(model[:L]))
c_output = Array(value.(model[:c]))
N_kt_output = Array(value.(model[:N_kt]))

# Result type modification for better interpretability
function res_analysis()
    class_sizes = countmap(y_labels) # dictionary, class label to class size
    label_sums = zeros(Int, K) # sum of correct classifications for a class
    println("Accuracies for every leaf:")
    for t in 1:(largest_B+1)
        print("Leaf "); print(t); print(": ")
        if isapprox(sum(c_output[:, t]), 1, atol = 0.1) # if leaf node has data points
            for k in 1:K
                if isapprox(c_output[k, t], 1, atol = 0.1) # labelled class for leaf t
                    points = round(Int, N_kt_output[k, t]) # number of data points
                    label_sums[k] = label_sums[k] + points # sum up to gather total number for a class
                    print(points)
                end
            end
            print("/"); println(round(Int, sum(z_output[:, t])))
        else
            println("empty")
        end
    end
    println()

    # print result
    println("Accuracies for every label:")
    for k in 1:K
        print("Label "); print(k); print(": "); print(label_sums[k])  
        print("/"); println(class_sizes[k])    
    end
    println()
    println("Overall accuracy:")
    println(sum(label_sums)/n)
    println()
end

res_analysis()
