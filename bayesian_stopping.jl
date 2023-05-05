using Distributed
using DataFrames
using HypothesisTests
using Gadfly
using Cairo
using Bootstrap
using Fontconfig
using Colors
using ColorSchemes
using RCall
addprocs(...) # set number of cores to be made available for parallel computing

@everywhere begin
    using Distributions
    using HypergeometricFunctions
    using QuadGK
    using SpecialFunctions
    using StatsBase
    using LinearAlgebra
    using KernelDensity
end

@everywhere function hdi(x::Vector{Float64}; α::Float64=0.05)::Vector{Float64}
    n = length(x)
    m = max(1, ceil(Int, α * n))
    y = sort(x)
    a = y[1:m]
    b = y[(n - m + 1):n]
    _, i = findmin(b - a)
    return [a[i], b[i]]
end

@everywhere function rii(x::Vector{Float64}; θ::Float64=.95)::Vector{Float64} # relative importance interval
    d = kde(sample(x, 20_000))
    m = maximum(d.density)
    cutoff = (1 - θ)m
    above = d.density .> cutoff
    s = sum(above) > 0. ? d.x[above] : [0.]
    return [extrema(s)...]
end

@everywhere function bayesfactor(k::Int64, n::Int64, p::Float64=0.5)::Float64
    function f(g, k, n)
        binom_pmf = pdf(Binomial(n, g), k)
        return binom_pmf
    end
    binom_bf = quadgk(x->f(x, k, n), 0, 1)[1] / pdf(Binomial(n, p), k)
    return binom_bf
end

@everywhere data_fnc(bias, numb_trials) = rand(Bernoulli(bias), numb_trials)

@everywhere function upd_hdi(bias; numb_trials=1500)
    v = Matrix{Float64}(undef, numb_trials + 1, 3)
    v[1, :] = vcat(hdi(rand(Beta(1, 1), 20_000)), 0.)
    tosses = data_fnc(bias, numb_trials)
    a = 0.
    b = 0.
    @inbounds for i in 1:numb_trials
        if tosses[i]
            a += 1
        else
            b += 1
        end
        v[i + 1, :] = vcat(hdi(rand(Beta(1 + a, 1 + b), 20_000)), a)
    end
    return v
end

@everywhere function upd_rii(bias; numb_trials=1500)
    v = Matrix{Float64}(undef, numb_trials + 1, 3)
    v[1, :] = vcat(rii(rand(Beta(1, 1), 20_000)), 0.)
    tosses = data_fnc(bias, numb_trials)
    a = 0.
    b = 0.
    @inbounds for i in 1:numb_trials
        if tosses[i]
            a += 1
        else
            b += 1
        end
        v[i + 1, :] = vcat(rii(rand(Beta(1 + a, 1 + b), 20_000)), a)
    end
    return v
end

@everywhere function upd_bf(bias; numb_trials=1500)
    v = Matrix{Float64}(undef, numb_trials + 1, 2)
    v[1, :] = vcat(bayesfactor(0, 0), 0.)
    tosses = data_fnc(bias, numb_trials)
    @inbounds for i in 1:numb_trials
        s = sum(tosses[1:i])
        v[i + 1, :] = vcat(bayesfactor(s, i), s/i)
    end
    return v
end

function accept_reject(res, i; precision=.08)
    ff = findfirst(x->x<precision, mapslices(diff, res[i][:, 1:2], dims=2))
    ff1 = ff[1]
    heads = res[i][ff1, 3]
    if res[i][ff1, 1] > .45 && res[i][ff1, 2] < .55
        val = 1
    elseif res[i][ff1, 1] >= .55 || res[i][ff1, 2] <= .45
        val = 2
    else
        val = 3
    end
    return ff1, heads/ff1, val
end

################
## BIAS = .5 ##
################

hdi_res = pmap(_->upd_hdi(.5), 1:1000)
rii_res = pmap(_->upd_rii(.5), 1:1000)
bf_res = pmap(_->upd_bf(.5), 1:1000)

hdi_final = [ accept_reject(hdi_res, i) for i in 1:length(hdi_res) ]

rii_final = [ accept_reject(rii_res, i) for i in 1:length(rii_res) ]

ff = [ findfirst(x->(x<1/3 || x>3), bf_res[i][:, 1]) for i in 1:length(bf_res) ]
bfr = [ bf_res[i][ff[i], 1] for i in 1:length(bf_res) ]
bf_ar = [ ifelse(bfr[i]>3, 2, 1) for i in 1:length(bf_res) ]
bf_est = [ bf_res[i][ff[i], 2] for i in 1:length(bf_res) ]
bf_final = [ (ff[i], bf_est[i], bf_ar[i]) for i in 1:length(bf_res) ]

mean_and_std(first.(hdi_final))
mean_and_std(first.(rii_final))
mean_and_std(first.(bf_final))

countmap(last.(hdi_final))
countmap(last.(rii_final))
countmap(last.(bf_final))

OneWayANOVATest([first.(hdi_final), first.(rii_final), first.(bf_final)]...)

OneWayANOVATest([(getindex.(hdi_final[last.(hdi_final) .== 1], 2) .- .5).^2, (getindex.(rii_final[last.(rii_final) .== 1], 2) .- .5).^2, (getindex.(bf_final[last.(bf_final) .== 1], 2) .- .5).^2]...)
EqualVarianceTTest((getindex.(hdi_final[last.(hdi_final) .== 1], 2) .- .5).^2, (getindex.(rii_final[last.(rii_final) .== 1], 2) .- .5).^2)
EqualVarianceTTest((getindex.(hdi_final[last.(hdi_final) .== 1], 2) .- .5).^2, (getindex.(bf_final[last.(bf_final) .== 1], 2) .- .5).^2)
EqualVarianceTTest((getindex.(rii_final[last.(rii_final) .== 1], 2) .- .5).^2, (getindex.(bf_final[last.(bf_final) .== 1], 2) .- .5).^2)

mean_and_std((getindex.(hdi_final[last.(hdi_final) .== 1], 2) .- .5).^2)
mean_and_std((getindex.(rii_final[last.(rii_final) .== 1], 2) .- .5).^2)
mean_and_std((getindex.(bf_final[last.(bf_final) .== 1], 2) .- .5).^2)

mean_and_std((getindex.(bf_final[last.(bf_final) .== 2], 2) .- .5).^2)

EqualVarianceTTest((getindex.(hdi_final[last.(hdi_final) .== 3], 2) .- .5).^2, (getindex.(rii_final[last.(rii_final) .== 3], 2) .- .5).^2)

mean_and_std((getindex.(hdi_final[last.(hdi_final) .== 3], 2) .- .5).^2)
mean_and_std((getindex.(rii_final[last.(rii_final) .== 3], 2) .- .5).^2)

s1 = sum(last.(hdi_final) .== 1)
s2 = sum(last.(rii_final) .== 1)
s3 = sum(last.(bf_final) .== 1)

@rput s1
@rput s2
@rput s3

R"""
prop.test(c(s1, s2, s3), c(1000, 1000, 1000))
"""

# to create figure 13.6, top right panel, from kruschke's book

hdi_switch = first.(hdi_final[last.(hdi_final) .== 1])
hdi_switches = ones(Int, 1500, length(hdi_switch))
[ hdi_switches[1:hdi_switch[i], i] .-= 1 for i in 1:length(hdi_switch) ]
hdi_proportion_accepted_per_time_step = mean(hdi_switches, dims=2) .* length(hdi_switch)
hdi_undecided_per_time_step = fill(1000, 1500) .- hdi_proportion_accepted_per_time_step

df_hdi = DataFrame(accepted = vec(hdi_proportion_accepted_per_time_step), 
                   undecided = vec(hdi_undecided_per_time_step), 
                   rejected = fill(0, 1500),
                   N = 1:1500)

df_hdi_stack = stack(df_hdi, [:accepted, :undecided, :rejected])
rename!(df_hdi_stack, [:N, :Null, :Proportion])

hdi_prop_plot = plot(df_hdi_stack, x=:N, y=:Proportion, color=:Null, Geom.line, 
                     Guide.title("HDI"),
                     style(line_width=2pt),
                     Scale.color_discrete_manual(["black", "slategray", "lightgray"]...))

draw(PDF("hdi_prop_plot_gray.pdf"), hdi_prop_plot)

# same for rii

rii_switch = first.(rii_final[last.(rii_final) .== 1])
rii_switches = ones(Int, 1500, length(rii_switch))
[ rii_switches[1:rii_switch[i], i] .-= 1 for i in 1:length(rii_switch) ]
rii_proportion_accepted_per_time_step = mean(rii_switches, dims=2) .* length(rii_switch)
rii_undecided_per_time_step = fill(1000, 1500) .- rii_proportion_accepted_per_time_step

df_rii = DataFrame(accepted = vec(rii_proportion_accepted_per_time_step), 
                   undecided = vec(rii_undecided_per_time_step), 
                   rejected = fill(0, 1500),
                   N = 1:1500)

df_rii_stack = stack(df_rii, [:accepted, :undecided, :rejected])
rename!(df_rii_stack, [:N, :Null, :Proportion])

rii_prop_plot = plot(df_rii_stack, x=:N, y=:Proportion, color=:Null, Geom.line, 
                     Guide.title("RII"),
                     style(line_width=2pt),
                     Scale.color_discrete_manual(["black", "slategray", "lightgray"]...))

draw(PDF("rii_prop_plot_gray.pdf"), rii_prop_plot)

# same for bf

bf_switch = first.(bf_final[last.(bf_final) .== 1])
bf_switches = ones(Int, 1500, length(bf_switch))
[ bf_switches[1:bf_switch[i], i] .-= 1 for i in 1:length(bf_switch) ]
bf_proportion_accepted_per_time_step = mean(bf_switches, dims=2) .* length(bf_switch)
bf_switch_r = first.(bf_final[last.(bf_final) .== 2])
bf_switches_r = ones(Int, 1500, length(bf_switch_r))
[ bf_switches_r[1:bf_switch_r[i], i] .-= 1 for i in 1:length(bf_switch_r) ]
bf_proportion_rejected_per_time_step = mean(bf_switches_r, dims=2) .* length(bf_switch_r)
bf_undecided_per_time_step = fill(1000, 1500) .- (bf_proportion_accepted_per_time_step .+ bf_proportion_rejected_per_time_step)

df_bf = DataFrame(accepted = vec(bf_proportion_accepted_per_time_step), 
                  undecided = vec(bf_undecided_per_time_step), 
                  rejected = vec(bf_proportion_rejected_per_time_step),
                  N = 1:1500)

df_bf_stack = stack(df_bf, [:accepted, :undecided, :rejected])
rename!(df_bf_stack, [:N, :Null, :Proportion])

bf_prop_plot = plot(df_bf_stack, x=:N, y=:Proportion, color=:Null, Geom.line, 
                    Guide.title("BF"),
                    Scale.x_log10(labels=x->"$(floor(Int, 10^x))"),
                    style(line_width=2pt),
                    Coord.cartesian(xmax=log10(1500)),
                    Scale.color_discrete_manual(["black", "slategray", "lightgray"]...))

draw(PDF("bf_prop_plot_gray.pdf"), bf_prop_plot)

# for other panels in the rightmost column of figure 13.6 in kruschke

p_hdi_accept = plot(x=getindex.(hdi_final[last.(hdi_final) .== 1], 2), Geom.histogram(bincount=25), Guide.title("HDI: Accept null"), Guide.xlabel("Sample proportion at decision"), Guide.ylabel("Count"), Coord.cartesian(xmin=.4, xmax=.601), Theme(default_color=colorant"slategray", major_label_font_size=14pt, minor_label_font_size=12pt));
p_rii_accept = plot(x=getindex.(rii_final[last.(rii_final) .== 1], 2), Geom.histogram(bincount=25), Guide.title("RII: Accept null"), Guide.xlabel("Sample proportion at decision"), Guide.ylabel("Count"), Coord.cartesian(xmin=.4, xmax=.601), Theme(default_color=colorant"slategray", major_label_font_size=14pt, minor_label_font_size=12pt));
p_bf_accept = plot(x=getindex.(bf_final[last.(bf_final) .== 1], 2), Geom.histogram(bincount=25), Guide.title("BF: Accept null"), Guide.xlabel("Sample proportion at decision"), Guide.ylabel("Count"), Coord.cartesian(xmin=.4, xmax=.601), Theme(default_color=colorant"slategray", major_label_font_size=14pt, minor_label_font_size=12pt));

draw(PDF("hdi_hist05_acc.pdf"), p_hdi_accept)
draw(PDF("rii_hist05_acc.pdf"), p_rii_accept)
draw(PDF("bf_hist05_acc.pdf"), p_bf_accept)

p_hdi_reject = plot(x=getindex.(hdi_final[last.(hdi_final) .== 2], 2), Geom.histogram(bincount=25), Guide.title("HDI: Reject null"), Guide.xlabel("Sample proportion at decision"), Guide.ylabel("Count"), Coord.cartesian(xmin=0, xmax=1), Theme(default_color=colorant"slategray", major_label_font_size=14pt, minor_label_font_size=12pt));
p_rii_reject = plot(x=getindex.(rii_final[last.(rii_final) .== 2], 2), Geom.histogram(bincount=25), Guide.title("RII: Reject null"), Guide.xlabel("Sample proportion at decision"), Guide.ylabel("Count"), Coord.cartesian(xmin=0, xmax=1), Theme(default_color=colorant"slategray", major_label_font_size=14pt, minor_label_font_size=12pt));
p_bf_reject = plot(x=getindex.(bf_final[last.(bf_final) .== 2], 2), Geom.histogram(bincount=25), Guide.title("BF: Reject null"), Guide.xlabel("Sample proportion at decision"), Guide.ylabel("Count"), Coord.cartesian(xmin=0, xmax=1), Theme(default_color=colorant"slategray", major_label_font_size=14pt, minor_label_font_size=12pt));

draw(PDF("hdi_hist05.pdf"), p_hdi_reject)
draw(PDF("rii_hist05.pdf"), p_rii_reject)
draw(PDF("bf_hist05.pdf"), p_bf_reject)

p_hdi_und = plot(x=getindex.(hdi_final[last.(hdi_final) .== 3], 2), Geom.histogram(bincount=25), Guide.title("HDI: Undecided"), Guide.xlabel("Sample proportion at decision"), Guide.ylabel("Count"), Coord.cartesian(xmin=0, xmax=1), Theme(default_color=colorant"slategray", major_label_font_size=14pt, minor_label_font_size=12pt));
p_rii_und = plot(x=getindex.(rii_final[last.(rii_final) .== 3], 2), Geom.histogram(bincount=25), Guide.title("RII: Undecided"), Guide.xlabel("Sample proportion at decision"), Guide.ylabel("Count"), Coord.cartesian(xmin=0, xmax=1), Theme(default_color=colorant"slategray", major_label_font_size=14pt, minor_label_font_size=12pt));
p_bf_und = plot(x=getindex.(bf_final[last.(bf_final) .== 3], 2), Geom.histogram(bincount=25), Guide.title("BF: Undecided"), Guide.xlabel("Sample proportion at decision"), Guide.ylabel("Count"), Coord.cartesian(xmin=0, xmax=1), Theme(default_color=colorant"slategray", major_label_font_size=14pt, minor_label_font_size=12pt));

draw(PDF("hdi_hist05_und.pdf"), p_hdi_und)
draw(PDF("rii_hist05_und.pdf"), p_rii_und)
draw(PDF("bf_hist05_und.pdf"), p_bf_und)

################
## BIAS = .65 ##
################

hdi_res_f = pmap(_->upd_hdi(.65), 1:1000)
hdi_final_f = [ accept_reject(hdi_res_f, i) for i in 1:length(hdi_res_f) ]
rii_res_f = pmap(_->upd_rii(.65), 1:1000)
rii_final_f = [ accept_reject(rii_res_f, i) for i in 1:length(rii_res_f) ]
bf_res_f = pmap(_->upd_bf(.65), 1:1000)
fff = [ findfirst(x->(x<1/3 || x>3), bf_res_f[i][:, 1]) for i in 1:length(bf_res_f) ]
bfrf = [ bf_res_f[i][fff[i], 1] for i in 1:length(bf_res_f) ]
bf_arf = [ ifelse(bfrf[i]>3, 2, 1) for i in 1:length(bf_res_f) ]
bf_estf = [ bf_res_f[i][fff[i], 2] for i in 1:length(bf_res_f) ]
bf_final_f = [ (fff[i], bf_estf[i], bf_arf[i]) for i in 1:length(bf_res_f) ]

mean_and_std(first.(hdi_final_f))
mean_and_std(first.(rii_final_f))
mean_and_std(first.(bf_final_f))

countmap(last.(hdi_final_f))
countmap(last.(rii_final_f))
countmap(last.(bf_final_f))

OneWayANOVATest([first.(hdi_final_f), first.(rii_final_f), first.(bf_final_f)]...)

OneWayANOVATest([(getindex.(hdi_final_f[last.(hdi_final_f) .== 2], 2) .- .65).^2, (getindex.(rii_final_f[last.(rii_final_f) .== 2], 2) .- .65).^2, (getindex.(bf_final_f[last.(bf_final) .== 2], 2) .- .65).^2]...)
EqualVarianceTTest((getindex.(hdi_final_f[last.(hdi_final_f) .== 2], 2) .- .65).^2, (getindex.(rii_final_f[last.(rii_final_f) .== 2], 2) .- .65).^2)
EqualVarianceTTest((getindex.(hdi_final_f[last.(hdi_final_f) .== 2], 2) .- .65).^2, (getindex.(bf_final_f[last.(rii_final_f) .== 2], 2) .- .65).^2)
EqualVarianceTTest((getindex.(hdi_final_f[last.(rii_final_f) .== 2], 2) .- .65).^2, (getindex.(bf_final_f[last.(rii_final_f) .== 2], 2) .- .65).^2)

mean_and_std((getindex.(hdi_final_f[last.(hdi_final_f) .== 2], 2) .- .65).^2)
mean_and_std((getindex.(rii_final_f[last.(rii_final_f) .== 2], 2) .- .65).^2)
mean_and_std((getindex.(bf_final_f[last.(bf_final_f) .== 2], 2) .- .65).^2)

s1f = sum(last.(hdi_final_f) .== 1)
s2f = sum(last.(rii_final_f) .== 1)
s3f = sum(last.(bf_final_f) .== 1)

@rput s1f
@rput s2f
@rput s3f

R"""
prop.test(c(s1f, s2f, s3f), c(1000, 1000, 1000))
"""

# figure 13.7

hdi_switch_f = first.(hdi_final_f[last.(hdi_final_f) .== 2])
hdi_switches_f = ones(Int, 1500, length(hdi_switch_f))
[ hdi_switches_f[1:hdi_switch_f[i], i] .-= 1 for i in 1:length(hdi_switch_f) ]
hdi_proportion_reject_per_time_step = mean(hdi_switches_f, dims=2) .* length(hdi_switch_f)
hdi_undecided_per_time_step_f = fill(1000, 1500) .- hdi_proportion_reject_per_time_step

df_hdi_f = DataFrame(accepted = fill(0, 1500), 
                   undecided = vec(hdi_undecided_per_time_step_f), 
                   rejected = vec(hdi_proportion_reject_per_time_step),
                   N = 1:1500)

df_hdi_f_stack = stack(df_hdi_f, [:accepted, :undecided, :rejected])
rename!(df_hdi_f_stack, [:N, :Null, :Proportion])

hdi_prop_plot_f = plot(df_hdi_f_stack, x=:N, y=:Proportion, color=:Null, Geom.line,
                       Guide.title("HDI"),
                       style(line_width=2pt),
                       Scale.color_discrete_manual(["black", "slategray", "lightgray"]...))


draw(PDF("hdi_prop_plot_f.pdf"), hdi_prop_plot_f)

# same for rii

rii_switch_f = first.(rii_final_f[last.(rii_final_f) .== 2])
rii_switches_f = ones(Int, 1500, length(rii_switch_f))
[ rii_switches_f[1:rii_switch_f[i], i] .-= 1 for i in 1:length(rii_switch_f) ]
rii_proportion_reject_per_time_step = mean(rii_switches_f, dims=2) .* length(rii_switch_f)
rii_undecided_per_time_step_f = fill(1000, 1500) .- rii_proportion_reject_per_time_step

df_rii_f = DataFrame(accepted = fill(0, 1500), 
                   undecided = vec(rii_undecided_per_time_step_f), 
                   rejected = vec(rii_proportion_reject_per_time_step),
                   N = 1:1500)

df_rii_f_stack = stack(df_rii_f, [:accepted, :undecided, :rejected])
rename!(df_rii_f_stack, [:N, :Null, :Proportion])

rii_prop_plot_f = plot(df_rii_f_stack, x=:N, y=:Proportion, color=:Null, Geom.line, 
                     Guide.title("RII"),
                     style(line_width=2pt),
                     Scale.color_discrete_manual(["black", "slategray", "lightgray"]...))

draw(PDF("rii_prop_plot_f.pdf"), rii_prop_plot_f)

# same for bf

bf_switch_f = first.(bf_final_f[last.(bf_final_f) .== 2])
bf_switches_f = ones(Int, 1500, length(bf_switch_f))
[ bf_switches_f[1:bf_switch_f[i], i] .-= 1 for i in 1:length(bf_switch_f) ]
bf_proportion_rejected_per_time_step_f = mean(bf_switches_f, dims=2) .* length(bf_switch_f)
bf_switch_a = first.(bf_final_f[last.(bf_final_f) .== 1])
bf_switches_a = ones(Int, 1500, length(bf_switch_a))
[ bf_switches_a[1:bf_switch_a[i], i] .-= 1 for i in 1:length(bf_switch_a) ]
bf_proportion_accepted_per_time_step_f = mean(bf_switches_a, dims=2) .* length(bf_switch_a)
bf_undecided_per_time_step_f = fill(1000, 1500) .- (bf_proportion_accepted_per_time_step_f .+ bf_proportion_rejected_per_time_step_f)

df_bf_f = DataFrame(accepted = vec(bf_proportion_accepted_per_time_step_f), 
                   undecided = vec(bf_undecided_per_time_step_f), 
                   rejected = vec(bf_proportion_rejected_per_time_step_f),
                   N = 1:1500)

df_bf_stack_f = stack(df_bf_f, [:accepted, :undecided, :rejected])
rename!(df_bf_stack_f, [:N, :Null, :Proportion])

bf_prop_plot_f = plot(df_bf_stack_f, x=:N, y=:Proportion, color=:Null, Geom.line, 
                      Guide.title("BF"),
                      Scale.x_log10(labels=x->"$(floor(Int, 10^x))"),
                      style(line_width=2pt),
                      Coord.cartesian(xmax=log10(1500)),
                      Scale.color_discrete_manual(["black", "slategray", "lightgray"]...))

draw(PDF("bf_prop_plot_gray_f.pdf"), bf_prop_plot_f)

# for other panels in the rightmost column of figure 13.7 in kruschke

p_hdi_accept_f = plot(x=getindex.(hdi_final_f[last.(hdi_final_f) .== 1], 2), Geom.histogram(bincount=25), Guide.title("HDI: Accept null"), Guide.xlabel("Sample proportion at decision"), Guide.ylabel("Count"), Coord.cartesian(xmin=0, xmax=1), Theme(default_color=colorant"slategray", major_label_font_size=14pt, minor_label_font_size=12pt))
p_rii_accept_f = plot(x=getindex.(rii_final_f[last.(rii_final_f) .== 1], 2), Geom.histogram(bincount=25), Guide.title("RII: Accept null"), Guide.xlabel("Sample proportion at decision"), Guide.ylabel("Count"), Coord.cartesian(xmin=0, xmax=1), Theme(default_color=colorant"slategray", major_label_font_size=14pt, minor_label_font_size=12pt))
p_bf_accept_f = plot(x=getindex.(bf_final_f[last.(bf_final_f) .== 1], 2), Geom.histogram(bincount=25), Guide.title("BF: Accept null"), Guide.xlabel("Sample proportion at decision"), Guide.ylabel("Count"), Coord.cartesian(xmin=0, xmax=1), Theme(default_color=colorant"slategray", major_label_font_size=14pt, minor_label_font_size=12pt))

draw(PDF("hdi_hist065_acc.pdf"), p_hdi_accept_f)
draw(PDF("rii_hist065_acc.pdf"), p_rii_accept_f)
draw(PDF("bf_hist065_acc.pdf"), p_bf_accept_f)

p_hdi_reject_f = plot(x=getindex.(hdi_final_f[last.(hdi_final_f) .== 2], 2), Geom.histogram(bincount=25), Guide.title("HDI: Reject null"), Guide.xlabel("Sample proportion at decision"), Guide.ylabel("Count"), Coord.cartesian(xmin=0, xmax=1), Theme(default_color=colorant"slategray", major_label_font_size=14pt, minor_label_font_size=12pt))
p_rii_reject_f = plot(x=getindex.(rii_final_f[last.(rii_final_f) .== 2], 2), Geom.histogram(bincount=25), Guide.title("RII: Reject null"), Guide.xlabel("Sample proportion at decision"), Guide.ylabel("Count"), Coord.cartesian(xmin=0, xmax=1), Theme(default_color=colorant"slategray", major_label_font_size=14pt, minor_label_font_size=12pt))
p_bf_reject_f = plot(x=getindex.(bf_final_f[last.(bf_final_f) .== 2], 2), Geom.histogram(bincount=25), Guide.title("BF: Reject null"), Guide.xlabel("Sample proportion at decision"), Guide.ylabel("Count"), Coord.cartesian(xmin=0, xmax=1), Theme(default_color=colorant"slategray", major_label_font_size=14pt, minor_label_font_size=12pt))

draw(PDF("hdi_hist065.pdf"), p_hdi_reject_f)
draw(PDF("rii_hist065.pdf"), p_rii_reject_f)
draw(PDF("bf_hist065.pdf"), p_bf_reject_f)

####################
## Generalization ##
####################

@everywhere struct Result
    bias::Float64
    interval::Vector{Float64}
    score::Float64
    stop::Int
end

@everywhere function cont_upd_var(probs, res, numb_samples)
    smps = Bool[ rand(Bernoulli(probs[j])) for j in 1:numb_samples ]
    hits = smps .== res
    post = probs[hits]
    return sample(post, numb_samples)
end

@everywhere function update_till_hdi(precision::Float64, α::Float64; max_updates::Int=1000, numb_samples::Int=2_500_000)
    bias = rand()
    pr = rand(Uniform(0, 1), numb_samples)
    for i in 1:max_updates
        flip = Int(rand(Bernoulli(bias)))
        pr = cont_upd_var(pr, flip, numb_samples)
        interval = hdi(pr; α=α)
        score = (mean(pr) - bias)^2 
        if diff(interval)[1] < precision
            return Result(bias, interval, score, i)
        end
    end
    return Result(bias, hdi(pr; α=α), (mean(pr) - bias)^2, max_updates)
end

@everywhere function update_till_rii(precision::Float64, θ::Float64; max_updates::Int=1000, numb_samples::Int=2_500_000)
    bias = rand()
    pr = rand(Uniform(0, 1), numb_samples)
    for i in 1:max_updates
        flip = Int(rand(Bernoulli(bias)))
        pr = cont_upd_var(pr, flip, numb_samples)
        interval = rii(pr; θ=θ)
        score = (mean(pr) - bias)^2
        if diff(interval)[1] < precision
            return Result(bias, interval, score, i)
        end
    end
    return Result(bias, rii(pr; θ=θ), (mean(pr) - bias)^2, max_updates)
end

@everywhere function update_till_bf(thresh::Int64; max_updates::Int=1000, numb_samples::Int=2_500_000)
    bias = rand()
    pr = rand(Uniform(0, 1), numb_samples)
    flip_count = 0
    for i in 1:max_updates
        flip = Int(rand(Bernoulli(bias)))
        pr = cont_upd_var(pr, flip, numb_samples)
        flip_count += flip
        bf = bayesfactor(flip_count, i)
        score = (mean(pr) - bias)^2
        if (bf < 1/thresh) || (bf > thresh)
            return Result(bias, [bf], score, i)
        end
    end
    return Result(bias, [0], (mean(pr) - bias)^2, max_updates)
end

res_h = pmap(_->update_till_hdi(.1, .05), 1:1000)
res_r = pmap(_->update_till_rii(.1, .95), 1:1000)
res_bf = pmap(_->update_till_bf(3), 1:1000)

bias1 = [ res_h[i].bias for i in 1:length(res_h) ]
lower1 = [ res_h[i].interval[1] for i in 1:length(res_h) ]
upper1 = [ res_h[i].interval[2] for i in 1:length(res_h) ]
scores1 = [ res_h[i].score for i in 1:length(res_h) ]
stop1 = [ res_h[i].stop for i in 1:length(res_h) ]

min.(bias1 .> lower1, bias1 .< upper1) |> mean_and_std
mean_and_std(scores1)
mean_and_std(stop1)

bias2 = [ res_r[i].bias for i in 1:length(res_r) ]
lower2 = [ res_r[i].interval[1] for i in 1:length(res_r) ]
upper2 = [ res_r[i].interval[2] for i in 1:length(res_r) ]
scores2 = [ res_r[i].score for i in 1:length(res_r) ]
stop2 = [ res_r[i].stop for i in 1:length(res_r) ]

min.(bias2 .> lower2, bias2 .< upper2) |> mean_and_std
mean_and_std(scores2)
mean_and_std(stop2)

scores3 = [ res_bf[i].score for i in 1:length(res_bf) ]
stop3 = [ res_bf[i].stop for i in 1:length(res_bf) ]

mean_and_std(scores3)
mean_and_std(stop3)
(1000 - sum([ res_bf[i].interval[1] for i in 1:length(res_bf) ] .< 1/3))/1000

OneWayANOVATest([stop1, stop2, stop3]...)
EqualVarianceTTest(stop1, stop2)
EqualVarianceTTest(stop1, stop3)
EqualVarianceTTest(stop2, stop3)

OneWayANOVATest([scores1, scores2, scores3]...)
EqualVarianceTTest(scores1, scores2)
EqualVarianceTTest(scores1, scores3)
EqualVarianceTTest(scores2, scores3)

sg1 = sum(min.(bias1 .> lower1, bias1 .< upper1))
sg2 = sum(min.(bias2 .> lower2, bias2 .< upper2))
sg3 = sum(1 .- [ res_bf[i].interval[1] for i in 1:length(res_bf) ] .< 1/3)

@rput sg1
@rput sg2
@rput sg3

R"""
prop.test(c(sg1, sg2, sg3), c(1000, 1000, 1000))
"""

R"""
prop.test(c(sg1, sg3), c(1000, 1000))
"""

R"""
prop.test(c(sg2, sg3), c(1000, 1000))
"""

R"""
prop.test(c(sg1, sg2), c(1000, 1000))
"""

# further generalization

function run_simulation_hdi(α::Float64)
    res = pmap(_->update_till_hdi(.1, α), 1:1000)
    bias = [ res[i].bias for i in 1:length(res) ]
    lower = [ res[i].interval[1] for i in 1:length(res) ]
    upper = [ res[i].interval[2] for i in 1:length(res) ]
    scores = [ res[i].score for i in 1:length(res) ]
    stop = [ res[i].stop for i in 1:length(res) ]
    return mean_and_std(min.(bias .> lower, bias .< upper)), mean_and_std(scores), mean_and_std(stop)
end

sim_res_hdi = [ run_simulation_hdi(i) for i in .05:.05:.5 ]

function run_simulation_rii(θ::Float64)
    res = pmap(_->update_till_rii(.1, θ), 1:1000)
    bias = [ res[i].bias for i in 1:length(res) ]
    lower = [ res[i].interval[1] for i in 1:length(res) ]
    upper = [ res[i].interval[2] for i in 1:length(res) ]
    scores = [ res[i].score for i in 1:length(res) ]
    stop = [ res[i].stop for i in 1:length(res) ]
    return mean_and_std(min.(bias .> lower, bias .< upper)), mean_and_std(scores), mean_and_std(stop)
end

sim_res_rii = [ run_simulation_rii(i) for i in .5:.05:.95 ]

function run_simulation_bf(thresh::Int64)
    res = pmap(_->update_till_bf(thresh), 1:1000)
    bf = [ res[i].interval[1] for i in 1:length(res) ]
    bfs = bf .> thresh
    scores = [ res[i].score for i in 1:length(res) ]
    stop = [ res[i].stop for i in 1:length(res) ]
    return mean_and_std(bfs), mean_and_std(scores), mean_and_std(stop)
end

sim_res_bf = [ run_simulation_bf(i) for i in 1:10 ]
