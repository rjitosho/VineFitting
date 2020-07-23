using LinearAlgebra
using JuMP, Ipopt, COSMO
using Plots
using BSON: @load

# load data
@load "vineQVF.bson" Q V F

# setup for optimization
numPins, numTimesteps = size(Q)
A = [Q' V']
A = kron(A,Matrix(I,numPins,numPins))
y = vec(F)
W = I

# solve optimization
numParams = 2*numPins^2
model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
set_silent(model)
@variable(model, x[1:numParams])
@objective(model, Min,(A*x-y)'*(A*x-y) + .05*x'*W*x)
JuMP.optimize!(model)
KCmatrix = reshape(JuMP.value.(x), numPins, 2*numPins)

# show K
K = KCmatrix[:, 1:numPins]
lim = maximum(abs.(K))
heatmap(reverse(K, dims=1), title="K", clim=(-lim,lim), c=:pu_or)

# show C
C = KCmatrix[:, numPins+1:end]
lim = maximum(abs.(C))
heatmap(reverse(C, dims=1), title="C", clim=(-lim,lim), c=:pu_or)

# compute goodness of fit
SSR = (A*JuMP.value.(param)-y)'*(A*JuMP.value.(param)-y)
SST = norm(y .- sum(y)/length(y))^2
Rsq = 1.0 - SSR/SST
