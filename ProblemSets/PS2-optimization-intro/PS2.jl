## Group Members : Ashish Kumar, Pallavi Wats and Firat Melih Yilmaz

## Problem Set 2

using Optim   
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using CSV
using FreqTables
using Distributions
using LaTeXStrings

## INTODUCTION TO NON-LINEAR OPTIMISATION IN JULIA

### Q1 ###
#optim is a function minimizer -- to find max of f(x) we need to minimize -f(x)
#optim packages gives us optimize() -- it requires three inputs : objective function, a string value and an optimizn algorithm

f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
negf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
startval = rand(1)   # random starting value
result = optimize(negf, startval, LBFGS())

## optimal value of x is -7.38 and the maximum is[-(-9.643*10^2) = 964.3]
m = -9.643134e+02

### Q2 ###
# use optim to compute OLS estimates of a simple linear regression using actual data 
nlsw = CSV.read("/Users/prachi/Desktop/Econometrics6343/PS1/nlsw88.csv")
df = convert(DataFrame, nlsw)
save("/Users/prachi/Desktop/Econometrics6343/PS1/nlsw88.jld", "df", df)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end

beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(beta_hat_ols.minimizer)
#coefficients in order of intercept, age, white, collgrad

#alternatively,
using GLM
estimate = inv(X'*X)*X'*y
df.white = df.race.==1
estimate_lm = lm(@formula(married ~ age + white + collgrad), df)

#### Q3 ###
#use optim to estimate logit likelihood -- for this, I will need to pass Optim the negative of the likelihood function

function logit(alpha, X, d)
    a = log.(1 .+ exp.(X*alpha))
    b = (d.*(X*alpha))
    loglike =  sum(a .- b)
    return loglike
end
beta_hat_logit = optimize(b-> logit(b,X,y),rand(size(X,2)),LBFGS(),Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(beta_hat_logit.minimizer)

## minimum : 1.416951e+03 (1416.951)

### Q4 ###
# use glm(formula, data, family-Binomial, link-LogitLink) to check answer
α = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
-loglikelihood(α) # answer = 1416.951

## Q5 ###
freqtable(df, :occupation) # note small number of obs in some occupations
df = dropmissing(df, :occupation)

#we also need to aggregate some of the occupation categories (category 10,11,12,13 -- 9)
df[df.occupation.==10,:occupation] .= 9
df[df.occupation.==11,:occupation] .= 9
df[df.occupation.==12,:occupation] .= 9
df[df.occupation.==13,:occupation] .= 9
freqtable(df, :occupation) # problem solved


#we changed the number of rows of df, we will need to re-define our X and y objects
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

#function mlogit(alpha, X, d)


    # your turn

    return loglike
#end
