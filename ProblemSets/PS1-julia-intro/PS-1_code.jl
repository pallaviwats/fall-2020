### GROUP MEMBERS : Ashish Kumar, Pallavi Wats and Firat Melih Yilmaz

using Random
using LinearAlgebra
using BenchmarkTools
using Distributions
using JLD
using DataFrames
using CSV
using DelimitedFiles
using TypedTables
using Latexify
using LaTeXStrings
using FreqTables


############## Q1 ################
#1.(a)
Random.seed!(1234)
A = rand(Uniform(-1,2), 10,7)
B = rand(Normal(-2,15),10,7)
C = hcat(A[1:5,1:5],B[1:5,6:7])
D = similar(A)
for i in 1:7, j in 1:7
    if A[i,j] <= 0
        D[i,j]=A[i,j]
    else D[i,j]=0
    end
end   

#1.(b)
#list the number of elements of the matrix
println(length(A)) 

#describe the size of the matrix
size(A)   

#list of elements in A
for i in 1:10, j in 1:7
println(A[i,j])
end

#1.(c)
#number of unique elements in the matrix D
length(unique(D))
    
#list of unique elements of matrix D
vector_D = D[:];
println(unique(vector_D))

#1.(d) 
#convert a matrix to a vector
#using reshape function
E = reshape(B,70)  

#simpler way
E  = B[:]

#1.(e)
#creating a three-dimensional vector using A in the first column of the 3rd dimension and B in the second column of the 3rd dimension
F = cat(A,B,dims=3)  

#1.(f)
#using permutedims to transform F from (10*7*2) to (2*10*7)
#F = permutedims(F,(2,10,7)) # not working

#using reshape
F = reshape(F,(2,10,7)) 

#1.(g)
#kronecker product of B and C 
G = kron(A,B)

#kron(C,F)   ## kron does not work well for 3-D matrices

#1.(h)
#saving matrix A,B,C,D,E,F,G as a .jld file called matrixpractice

#saving all the matrices with their respective unique identifiers
save("/Users/prachi/Desktop/Econometrics6343/PS1/matrixpractice.jld","A",A,"B",B,"C",C,"D",D,"E",E,"F",F,"G",G)

#if I have to fetch only Matrix A 
#load("/Users/prachi/Desktop/Econometrics6343/PS1/matrixpractice.jld","A")

#1.(i)
#save only the matrices A, B, C, and D as a .jld file called firstmatrix.
save("/Users/prachi/Desktop/Econometrics6343/PS1/firstmatrix.jld","A",A,"B",B,"C",C,"D",D)

#1.(j)
#export C as a .csv file called Cmatrix -- you will first need to transform C into a Dataframe
#converting to dataframe
C = convert(DataFrame,C) 

#write a csv file
CSV.write("/Users/prachi/Desktop/Econometrics6343/PS1/Cmatrix.csv",C)

#1.(h)
#export D as a tab-delimited .dat file called Dmatrix -- you will first need to transform D into a DataFrame
D = convert(DataFrame,D)

#write a tab delimited .dat file
CSV.write("/Users/prachi/Desktop/Econometrics6343/PS1/Dmatrix.dat", D)


#1.(k)
#wrap it in function
function q1()
    Random.seed!(1234)
    A = rand(10,7) .* (10-(-5)).-5
    B = randn(10,7) .* (225-(-2)).-2
    C = hcat(A[1:5,1:5],B[1:5,6:7])
    D = similar(A)
    for i in 1:10, j = 1:7
        if A[i,j] <= 0
            D[i,j] = A[i,j]
        else
            D[i,j] = 0
        end
    end

    return A,B,C,D
end

A,B,C,D = q1()


############# Q2 #################
#2.(a)
#loop that computes element-by-element product of Matrix A and B, called AB
AB = zeros(10,7)
for i in 1:length(A)
    AB[i]=A[i]*B[i]
end


#without a loop element-by-element product of A and B
AB2 = A.*B

#2.(b)
#write a loop to create a column vector called Cprime which contains only elements of C that are between [-5,5] 
Cprime = C[:]
for item in eachrow(C), i in item
    if -5<= i <= 5
        append!(Cprime,i)
    end
end


#do above without a loop
Cprime2 = C[-5 .≤ C .≤ 5];

#2.(c)
#use a loop to craete a 3D array X of dimension 15*169*5 
X = zeros((15169,6,5))
for i in 1:size(X)[3], j in 1:size(X)[2]
    if j == 1
        X[:,j,i] .= 1
    elseif  j==2 
        a2 = rand(Binomial(1,(0.75*(6-i))/5),15169) 
        X[:,j,i] = a2
    elseif j==3 
        a3 = rand(Normal(15+i-1,5*(i-1)),15169)
        X[:,j,i] = a3
    elseif j == 4 
        a4 = rand(Normal((pi*(6-i))/3,1/ℯ),15169)
        X[:,j,i] = a4
    elseif  j == 5 
        a5 = rand(Normal(12,2.19),15169) 
        X[:,j,i] = a5
    elseif j == 6
        a6 = rand(Binomial(20,0.5),15169) 
        X[:,j,i]  = a6   
    end
end

#2.(d)
#use comprehension to create a matrix β (6*5) whose elements eveolve across time 
K = 6
T = 5
β = zeros(K,T)
β[1,:] = [1:0.25:2;]
β[2,:] = [ log(x) for x in 1:T ]
β[3,:] = [-sqrt(x) for x in 1:T ]
β[4,:] = [ exp(x)-exp(x+1) for x in 1:T ]
β[5,:] = [x for x in 1:T]
β[6,:] = [x/3 for x in 1:T]

#2.(e)
#use comprehensions to create matrix Y (N*T) defined by ...

Y = zeros((15169,5,5))
for i in 1:5
    ϵ = rand(Normal(0,0.36),(15169,5))
    Y[:,:,i] = X[:,:,i]*β + ϵ
end

################ Q3 ######################
#3.(a)
#import nlsw88 as DataFrames
## first import nlsw as CSV file
nlsw = CSV.read("/Users/prachi/Desktop/Econometrics6343/PS1/nlsw88.csv")

#check if there are missing values 
ismissing(nlsw)    #false -- no missing data

#convert nlsw88 into a DataFrame
df = convert(DataFrame, nlsw)

save("/Users/prachi/Desktop/Econometrics6343/PS1/nlsw88.jld", "df", df)

#make nullable
allowmissing!(df)

#describe data
describe(df)

#3.(b)
#what % of the sample have never been married?
m = freqtable(df, :never_married)  
 
never_married = (sum(df["never_married"])/(sum(df["never_married"]) + sum(df["married"]))) * 100;
println("Percentage of the sample that has never been married: $never_married %")

#3.(c)
#use tabulate() to report what percentage of sample is in each race category 
#using freqtable instead
race = prop(freqtable(df["race"]));

#3.(d)
#use the describe() function to create a matrix called summarystats which lists the mean, median, standard deviation, min, max, number of unique elements, and IQR
summarystats = describe(df);


#3.(e)
#show the joint distribution of industry and occupation using a cross-tabulation

joint_table = freqtable(df, :industry, :occupation);

#3.(g)
#wrap a function definition around all of the code for question 3
#call the function q3(). The function should have no inputs and no outputs
function q3()
    df = CSV.read("/Users/prachi/Desktop/Econometrics6343/PS1/nlsw88.csv");
    @save("nlsw88.jld", df)
    never_married = (sum(df["never_married"])/(sum(data["never_married"]) + sum(data["married"]))) * 100;
    race = prop(freqtable(df["race"]));
    summarystats = describe(df);    
end



############## Q4 ###############
# Practice with functions

#4.(a)
#load firstmatrix.jld
load("/Users/prachi/Desktop/Econometrics6343/PS1/firstmatrix.jld")

#write a function called matrixops that takes as inputs the matrices A and B from question (a) of problem 1 and has three outputs: (i) the element-by-element product of the inputs, 
# (ii) the product A′B, and (iii) the sum of all the elements of A + B.
#4.(d)
function matrixops(A, B)
    i = A .+ B
    ii = A' * B
    iii = sum(A + B)

    return i, ii, iii   
end

matrixops(A,B);


#4.(c)
function matrixops(A, B)
    """
    Takes  matrix A and B as inputs and returns following outputs:

    i = A .+ B
    ii = A' * B
    iii = sum(A + B)
    """
    i = A .+ B
    ii = A' * B
    iii = sum(A + B)

    return i, ii, iii   
end

#4.(e)
function matrixops(A, B)
    """
    Takes  matrix A and B as inputs and returns following outputs:

    i = A .+ B
    ii = A' * B
    iii = sum(A + B)
    """
    if size(A) != size(B)
        throw(DimensionMismatch("inputs must have the same size."))
    i = A .+ B
    ii = A' * B
    iii = sum(A + B)

    return i, ii, iii   
end



