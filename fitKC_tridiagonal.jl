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

# extract cols of A corresponding to tridiagonal elements of K and C
upIdx = [i for i = numPins+1:numPins+1:numPins^2]
diagIdx = [i for i = 1:numPins+1:numPins^2]
downIdx = [i for i = 2:numPins+1:numPins^2]
tridiagIdx = [downIdx; diagIdx; upIdx]
KCTridiagIdx = [tridiagIdx; numPins^2 .+ tridiagIdx]
A = A[:, KCTridiagIdx]

# weights matrix
W = .005I

# solve optimization
numParams = length(KCTridiagIdx)
model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
set_silent(model)
@variable(model, x[1:numParams])
@objective(model, Min,(A*x-y)'*(A*x-y) + x'*W*x)

# extract results
JuMP.optimize!(model)
params = JuMP.value.(x)

# show K
Kparams = params[1:Int(numParams/2)]
K = Tridiagonal(Kparams[1:numPins-1], Kparams[numPins:2*numPins-1], Kparams[2*numPins:end])
lim = maximum(abs.(K))
heatmap(reverse(Matrix(K), dims=1), title="K", clim=(-lim,lim), c=:pu_or)

# show C
Cparams = params[1+Int(numParams/2):end]
C = Tridiagonal(Cparams[1:numPins-1], Cparams[numPins:2*numPins-1], Cparams[2*numPins:end])
lim = maximum(abs.(C))
heatmap(reverse(Matrix(C), dims=1), title="C", clim=(-lim,lim), c=:pu_or)

# compute goodness of fit
SSR = (A*params-y)'*(A*params-y)
SST = norm(y .- sum(y)/length(y))^2
Rsq = 1.0 - SSR/SST
