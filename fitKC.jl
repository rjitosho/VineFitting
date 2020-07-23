using LinearAlgebra
using JuMP, Ipopt
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
@variable(model, param[1:numParams])
@objective(model, Min,(A*param-y)'*(A*param-y) + .05*param'*W*param)
JuMP.optimize!(model)

# show results
KCmatrix = reshape(JuMP.value.(param), numPins, 2*numPins)
K = KCmatrix[:, 1:numPins]
C = KCmatrix[:, numPins+1:end]
heatmap(K)
heatmap(C)

# compute goodness of fit
SSR = (A*JuMP.value.(param)-y)'*(A*JuMP.value.(param)-y)
SST = norm(y .- mean(y))^2
Rsq = 1.0 - SSR/SST
