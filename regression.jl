using LinearAlgebra
using JuMP, Ipopt
using Makie

function saveHeatMap(matrix,xaxis,yaxis,title,filename,sym=true)
    scene = Scene()
    if sym
        top = Makie.plot!(scene,matrix,colormap=ColorSchemes.bwr,colorrange=(-maximum(abs.(matrix)),maximum(abs.(matrix))))
    else
        top = Makie.plot!(scene,matrix,colormap=ColorSchemes.bwr)
    end
    axis = scene[Makie.Axis]
    axis[:names, :axisnames] = (xaxis, yaxis)
    axis[:names, :title] = (title)
    cm = colorlegend(top[end],width = (30,150))
    top = vbox(top,cm)
    Makie.save(filename,top)
    return
end

"""
linear fit
use all data points
use angles from all joints to fit model for each joint
feature map (theta_all thetadot_all)
"""
function compute_params(A, y, W, PSD = false)


    # solve optimization
    num_cols = 2*links*links
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    set_silent(model)
    @variable(model, param[1:num_cols])
    @constraint(model, damp_min, param[links^2+1:end] .>= -.02)
    @constraint(model, damp_max, param[links^2+1:end] .<= .02)
    @objective(model, Min,(A*param-y)'*(A*param-y) + .05*param'*param)
    JuMP.optimize!(model)
    KCmatrix = reshape(JuMP.value.(param), links, 2*links)

    # compute goodness of fit
    SSR = (A*JuMP.value.(param)-y)'*(A*JuMP.value.(param)-y)
    SST = norm(y .- mean(y))^2
    Rsq = 1.0 - SSR/SST

    return KCmatrix, Rsq
end

"""LINEAR FIT"""
function fit_angle_range(lower, upper, X_c, Xd_c, Xl, Xld, U_c)
    links,N = size(U_c)
    params = zeros(5,links)
    RsqValues = []
    jointIndices = []
    for i = 1:links
        θ = X_c[i,:]
        θd = Xd_c[i,:]
        l = Xl[i,:]
        ld = Xld[i,:]
        τ = U_c[i,:]

        bentIndices = findall(x->((x<upper)&&(x>lower)),θ)
        if length(bentIndices) > 5
            bent_θ = θ[bentIndices]
            bent_θd = θd[bentIndices]
            bent_l = l[bentIndices]
            bent_ld = ld[bentIndices]
            bent_τ = τ[bentIndices]

            A = [bent_θ bent_θd bent_l bent_ld ones(length(bent_θ))]
            param, Rsq = linearRegression(A, bent_τ)
            params[:,i] = param
            push!(RsqValues, Rsq)
            push!(jointIndices, i)
        end
    end
    return params, RsqValues, jointIndices
end

"""LINEAR FIT WITH ATAN"""
function fit_angle_range_nonlinear(lower, upper, θ_cutoff, X_c, Xd_c, Xl, Xld, U_c)
    a = 1.5*180/(pi*θ_cutoff)
    links,N = size(U_c)
    params = zeros(5,links)
    RsqValues = []
    jointIndices = []
    for i = 1:links
        θ = X_c[i,:]
        θd = Xd_c[i,:]
        l = Xl[i,:]
        ld = Xld[i,:]
        τ = U_c[i,:]

        bentIndices = findall(x->((x<upper)&&(x>lower)),θ)
        if length(bentIndices) > 5
            bent_θ = θ[bentIndices]
            bent_θd = θd[bentIndices]
            bent_l = l[bentIndices]
            bent_ld = ld[bentIndices]
            bent_τ = τ[bentIndices]

            A = [bent_θ bent_θd bent_l bent_ld atan.(a*bent_θ)]
            param, Rsq = linearRegression(A, bent_τ)
            params[:,i] = param
            push!(RsqValues, Rsq)
            push!(jointIndices, i)
        end
    end
    return params, RsqValues, jointIndices
end

"""SIMPLE LINEAR FIT WITH ATAN (no length terms)"""
function fit_angle_range_nonlinear_no_length_terms(lower, upper, θ_cutoff, X_c, Xd_c, U_c)
    a = 1.5*180/(pi*θ_cutoff)
    links,N = size(U_c)
    params = zeros(3,links)
    RsqValues = []
    jointIndices = []
    for i = 1:links
        θ = X_c[i,:]
        θd = Xd_c[i,:]
        τ = U_c[i,:]

        bentIndices = findall(x->((x<upper)&&(x>lower)),θ)
        if length(bentIndices) > 5
            bent_θ = θ[bentIndices]
            bent_θd = θd[bentIndices]
            bent_τ = τ[bentIndices]

            A = [bent_θ bent_θd atan.(a*bent_θ)]
            param, Rsq = linearRegression(A, bent_τ)
            params[:,i] = param
            push!(RsqValues, Rsq)
            push!(jointIndices, i)
        end
    end
    return params, RsqValues, jointIndices
end
