using LinearAlgebra
using Plots
using BSON: @load

# load data
@load "vineQVF.bson" Q V F
numPins, numTimesteps = size(Q)

# setup for optimization
A = [Q' V']
A = kron(A,Matrix(I,numPins,numPins))

y = vec(F)

# extract cols of A corresponding to tridiagonal elements of K and C
upIdx = [i for i = numPins+1:numPins+1:numPins^2]
diagIdx = [i for i = 1:numPins+1:numPins^2]
lowIdx = [i for i = 2:numPins+1:numPins^2]

tridiagIdx = [lowIdx; diagIdx; upIdx]
KCTridiagIdx = [tridiagIdx; numPins^2 .+ tridiagIdx]

A = A[:, KCTridiagIdx]

# weights matrix
W = .01I

# solve optimization
x = (A'*A + W)\(A'*y)

# show K
Kparams = x[1:Int(length(x)/2)]

lower = Kparams[1:numPins-1]
diag = Kparams[numPins:2*numPins-1]
upper = Kparams[2*numPins:end]

K = Tridiagonal(lower, diag, upper)

lim = maximum(abs.(K))
heatmap(reverse(Matrix(K), dims=1), title="K", clim=(-lim,lim), c=:pu_or)
png("K")

# show C
Cparams = x[1+Int(length(x)/2):end]

lower = Cparams[1:numPins-1]
diag = Cparams[numPins:2*numPins-1]
upper = Cparams[2*numPins:end]

C = Tridiagonal(lower, diag, upper)

lim = maximum(abs.(C))
heatmap(reverse(Matrix(C), dims=1), title="C", clim=(-lim,lim), c=:pu_or)
png("C")

# compute goodness of fit
SSR = (A*x-y)'*(A*x-y)
SST = norm(y .- sum(y)/length(y))^2
Rsq = 1.0 - SSR/SST
