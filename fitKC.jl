include("../src/regression.jl")


links, timesteps = size(X_c)
links,N = size(U_c)

A = [X_c' Xd_c']
A = kron(A,Matrix(I,links,links))
y = vec(U_c)

params_atan, RsqValues_atan = fit_w_all_angles(atan.(a*X_c), Xd_c, U_c)
roundR2 = round(mean(RsqValues_atan),digits=2)
Kmatrix = params_atan[:,1:links]
Cmatrix = params_atan[:,links+1:end]

# save results
saveHeatMap(reverse(Kmatrix,dims=1)',"","","K, $links Links, θc = $θ_cutoff, R^2 = $roundR2",string("examples/MatrixRegressionResults/KMatrix$links","links$θ_cutoff.png"))
saveHeatMap(reverse(Cmatrix,dims=1)',"","","C, $links Links, θc = $θ_cutoff, R^2 = $roundR2",string("examples/MatrixRegressionResults/CMatrix$links","links$θ_cutoff.png"))

# adjust diagonal method 1
Kmatrix = params_atan[:,1:links]
KRowMax = maximum(abs.(Kmatrix), dims=2)
scaling = maximum(KRowMax)/links
for i = links:-1:1
    Kmatrix[i,i] = minimum([-KRowMax[i]; Kmatrix[i,i]; [Kmatrix[j,j]-scaling for j = i+1:links]])
end
saveHeatMap(reverse(Kmatrix,dims=1)',"","","K, $links Links, θc = $θ_cutoff, R^2 = $roundR2",string("examples/MatrixRegressionResults/KMatrixADJ$links","links$θ_cutoff.png"))

# # adjust diagonal method 2
# Kmatrix = params_atan[:,1:links] - .06I
# for i = 12:-1:1
#     Kmatrix[i,i] = Kmatrix[i+1,i+1] - .01
# end
# Makie.plot(reverse(Kmatrix,dims=1)')
saveHeatMap(reverse(Kmatrix,dims=1)',"","","K, $links Links, θc = $θ_cutoff, R^2 = $roundR2",string("examples/MatrixRegressionResults/KMatrixADJ$links","links$θ_cutoff.png"))
