using LinearAlgebra
using JuMP, COSMO
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
model = Model(COSMO.Optimizer)
@variable(model, x[1:numParams])
@variable(model, K[1:numPins, 1:numPins], PSD)
@variable(model, C[1:numPins, 1:numPins])
@objective(model, Min,(A*[vec(-K);vec(C)]-y)'*(A*[vec(-K);vec(C)]-y) + .05*[vec(-K);vec(C)]'*W*[vec(-K);vec(C)])
JuMP.optimize!(model)

# show K
K = -JuMP.value.(K)
lim = maximum(abs.(K))
heatmap(reverse(K, dims=1), title="K", clim=(-lim,lim), c=:pu_or)

# show C
C = JuMP.value.(C)
lim = maximum(abs.(C))
heatmap(reverse(C, dims=1), title="C", clim=(-lim,lim), c=:pu_or)

# compute goodness of fit
SSR = (A*[vec(K);vec(C)]-y)'*(A*[vec(K);vec(C)]-y)
SST = norm(y .- sum(y)/length(y))^2
Rsq = 1.0 - SSR/SST
