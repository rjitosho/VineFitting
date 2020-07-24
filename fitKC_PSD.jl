using LinearAlgebra, ToeplitzMatrices
using JuMP, COSMO
using Plots
using BSON: @load

# load data
@load "vineQVF.bson" Q V F
numPins, numTimesteps = size(Q)

# setup for optimization
A = [Q' V']
A = kron(A,Matrix(I,numPins,numPins))

y = vec(F)

# set weights matrix
v = .1*[0;0;[i for i = range(.3,1,length = numPins-2)]]
WK = Toeplitz(v,v)

v = .05*[i for i = range(.3,1,length = numPins)]
WC = Toeplitz(v,v)

W = Diagonal([vec(WK);vec(WC)])

# solve optimization
model = JuMP.Model(COSMO.Optimizer)

@variable(model, K[1:numPins, 1:numPins], PSD)
@variable(model, C[1:numPins, 1:numPins], PSD)

x = [vec(-K);vec(-C)]
@objective(model, Min,(A*x-y)'*(A*x-y) + x'*W*x)

JuMP.optimize!(model)

# show K
K = -JuMP.value.(K)
lim = maximum(abs.(K))
heatmap(reverse(K, dims=1), title="K", clim=(-lim,lim), c=:pu_or)
png("K")

# show C
C = -JuMP.value.(C)
lim = maximum(abs.(C))
heatmap(reverse(C, dims=1), title="C", clim=(-lim,lim), c=:pu_or)
png("C")

# compute goodness of fit
SSR = (A*[vec(K);vec(C)]-y)'*(A*[vec(K);vec(C)]-y)
SST = norm(y .- sum(y)/length(y))^2
Rsq = 1.0 - SSR/SST
