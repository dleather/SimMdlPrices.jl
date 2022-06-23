module SimMdlPrices

using Parameters
using LinearAlgebra
using QuantEcon
using SparseArrays
using StaticArrays
using Statistics
using Infiltrator
using StatsFuns
using InteractiveUtils
using TimerOutputs
using RecursiveArrayTools
using QuadGK
using Distributions
using DelimitedFiles
using ForwardDiff
using Optim
using StatsBase
using FillArrays

function simulate_ms_var_1_cond_shocks(x0,s0,μ,Φ,Σ,Π,u,ϵ)
     

     s_mat = simulate_markov_switch_init_cond_shock(s0,Π,u)
     x_mat = simulate_msvar_cond_regime_path_shock(x0,s_mat,μ,Φ,Σ,ϵ)

     return (x_mat, s_mat)
end

function simulate_markov_switch_init_cond_shock(s0,Π,u)
     
     
     T,G = size(u)
     S = size(Π)[1]

     #Preallocate
     s_mat = fill(0,T+1,G)
     #s_mat = Array{Int64}(undef,T+1,G)
     s_mat[1,:] .= s0

     #Propogate
     Π₀ = cumsum(Π, dims = 2)
     @inbounds for g ∈ 1:G
          @inbounds for t ∈ 1:T
               @inbounds for s ∈ 1:S
                    if u[t, g] <= Π₀[s_mat[t, g], s]
                         s_mat[t+1, g] = s
                         break
                    end
               end
          end
     end

     return s_mat
     
end

function simulate_msvar_cond_regime_path_shock(x0,s_mat,μ,Φ,Σ,ϵ)

     #Unload and Preallocate
     N, N2 = size(x0)
     T = size(s_mat)[1] - 1
     G = size(s_mat)[2]
     #@assert G == size(ϵ)[3]
     #@assert T == size(ϵ)[2]
     #@assert N == size(ϵ)[1]
     

     x_mat = fill(0.,N,T+1,G)

     if N2==1
          x0 = repeat(x0, 1, G)
     end
          #else
          #@assert N2 == G
      #end

     #Propogate
     @inbounds for g ∈ 1:G

          x_mat[:, 1, g] = @view(x0[:, g])
          @inbounds for t ∈ 1:T

               s = s_mat[t+1, g]
               x_mat[:, t+1, g] = μ[s] .+ Φ[s] *  @view(x_mat[:,t,g])  .+ Σ[s] * @view(ϵ[:, t, g])

          end

     end

     return x_mat
     
end

function  construct_structural_matrices_macro(ϕ,δ,ρ₁,ρ₂,β₁,β₂,α₁,α₂,μ_g,μ_pi,m_g,m_r₁,m_r₂,σ_g,σ_pi,σ_r₁,σ_r₂)
     
     b0₁,b0₂ = construct_b0_macro_array(ϕ,δ,ρ₁,ρ₂,β₁,β₂)
     bm1₁,bm1₂ = construct_bm1_macro_array(μ_g,μ_pi,ρ₁,ρ₂);
     b1₁,b1₂ = construct_b1_macro_array(μ_g,μ_pi,ϕ,ρ₁,ρ₂,α₁,α₂)
     m0₁,m0₂ = construct_m0_macro_array(m_g,m_r₁,m_r₂)
     Γ₁,Γ₂ = construct_gamma_macro_array(σ_g,σ_pi,σ_r₁,σ_r₂)

     return b0₁,b0₂,bm1₁,bm1₂,b1₁,b1₂,m0₁,m0₂,Γ₁,Γ₂
end

function construct_b0_macro_array(ϕ,δ,ρ₁,ρ₂,β₁,β₂)
     b0₁ = @SMatrix [1. 0. ϕ; -δ  1. 0.; -(1. - ρ₁)*β₁ 0. 1.]
     b0₂ = @SMatrix [1. 0. ϕ; -δ  1. 0.; -(1. - ρ₂)*β₂ 0. 1.] 
     
     return b0₁,b0₂
end

function construct_bm1_macro_array(μ_g, μ_pi, ρ₁, ρ₂)
     bm1₁ = @SMatrix [(1. - μ_g) 0. 0.; 0. (1. - μ_pi) 0.; 0. 0. ρ₁]
     bm1₂ = @SMatrix [(1. - μ_g) 0. 0.; 0. (1. - μ_pi) 0.; 0. 0. ρ₂]

     return bm1₁,bm1₂
end

function construct_b1_macro_array(μ_g,μ_pi,ϕ,ρ₁,ρ₂,α₁,α₂)
     
     b1₁ = @SMatrix [μ_g ϕ 0.; 0. μ_pi 0.; 0. (1. - ρ₁)*α₁ 0.]
     b1₂ = @SMatrix [μ_g ϕ 0.; 0. μ_pi 0.; 0. (1. - ρ₂)*α₂ 0.]

     return b1₁,b1₂     
end

function construct_m0_macro_array(m_g,m_r₁,m_r₂)

     m0₁ = SMatrix{3,1,Float64}([m_g 0. m_r₁])
     m0₂ = SMatrix{3,1,Float64}([m_g 0. m_r₂]);

     return m0₁,m0₂
end

function construct_gamma_macro_array(σ_g, σ_pi, σ_r₁, σ_r₂)

     Γ₁ = @SMatrix [σ_g  0.  0.; 0. σ_pi  0.; 0. 0. σ_r₁]
     Γ₂ = @SMatrix [σ_g  0.  0.; 0. σ_pi  0.; 0. 0. σ_r₂]

     return Γ₁,Γ₂
end

function transform_struct_to_rf(b0₁, b0₂, bm1₁, bm1₂, b1₁, b1₂, m0₁, m0₂, Γ₁, Γ₂, π_m, π_d, Π_0, Π_X)
     
     maxₖ = 1000 #Maximum iterations for REE 

     # Compute composite Markov transition matrix and its ergodic distribution
     Π = kron(π_m, π_d)
     q = get_ergodic_markov_dist(Π)

     #Compute forward-looking RE equilibrium
     F1₁ = b0₁ \ bm1₁ 
     F1₂ = b0₂ \ bm1₂
     A1₁ = b0₁ \ b1₁
     A1₂ = b0₂ \ b1₂

     Σ1_11 = b0₁ \ Γ₁
     Σ1_12 = b0₁ \ Γ₂
     Σ1_21 = b0₂ \ Γ₁
     Σ1_22 = b0₂ \ Γ₂

     m1₁ = b0₁ \ m0₁
     m1₂ = b0₂ \ m0₂

     #Compute forward solution
     Φ, μ, Σ, det_dum = get_forward_solution_msre(π_m, A1₁, A1₂, F1₁, F1₂, m1₁, m1₂, Σ1_11, Σ1_12, Σ1_21,
                                                                                  Σ1_22, maxₖ)

     #Define covariance "cell"
     cov = [Σ[1] * Σ[1]', Σ[2] * Σ[2]', Σ[3] * Σ[3]', Σ[4] * Σ[4]']
     
     #Compute risk-neautral dynamic 
     μ_Q = [ μ[1] - cov[1] * Π_0,
                   μ[2] - cov[2] * Π_0,
                   μ[3] - cov[3] * Π_0,
                   μ[4] - cov[4] * Π_0 ] 

     Φ_Q = [ Φ[1] - cov[1] * Π_X,
                    Φ[2] - cov[2] * Π_X,
                    Φ[3] - cov[3] * Π_X,
                    Φ[4] - cov[4] * Π_X ]

     return μ, Φ, μ_Q, Φ_Q, Σ, cov, Π, q, det_dum    
end

function get_forward_solution_msre(π_m, A1₁, A1₂, F1₁, F1₂, m1₁, m1₂, Σ1_11, Σ1_12, Σ1_21, 
               Σ1_22, maxₖ)

     Φ, k_term = get_rf_Φ(π_m, F1₁, F1₂, A1₁, A1₂, maxₖ = maxₖ)

     μ, Σ =  compute_μ_Σ_fmmsre(Φ, A1₁, A1₂, m1₁, m1₂, Σ1_11, Σ1_12, Σ1_21, Σ1_22, π_m)

     if k_term == maxₖ
          det_dum = 0
     else
          det_dum = 1
     end

     Φ_out = [Φ[1], Φ[1], Φ[2], Φ[2]]
     μ_out = [μ[1], μ[1], μ[2], μ[2]]
     return Φ_out, μ_out, Σ, det_dum
end

function get_rf_Φ(π_m, F1₁, F1₂, A1₁, A1₂; maxₖ = 1000, tolK = 0.000001)
     
     B₁ = F1₁
     B₂ = F1₂
     Ωkm1₁ = B₁
     Ωkm1₂ = B₂
     Ωkm1 = [Ωkm1₁,Ωkm1₂]
     k_term = 0
     N = size(B₁)[1]
     Ωk₁ = SMatrix{N,N}(1. * I)
     Ωk₂ = SMatrix{N,N}(1. * I)

     function f!(Ωk₁, Ωk₂, k_term, Ωkm1,maxₖ,A1₁, A1₂,π_m,B₁,B₂,tolK)
               Ξ₁ = similar(Ωk₁)
               Ξ₂ = similar(Ωk₂)
               for i ∈ 1:maxₖ
                    #update Φₖ
                    Ξ₁ =  (I - A1₁ * (π_m[1,1] .* Ωkm1[1] + π_m[1,2] .* Ωkm1[2] ) ) 
                    Ξ₂ =  (I - A1₂ * (π_m[2,1] .* Ωkm1[1] + π_m[2,2] .* Ωkm1[2] ) ) 
                    Ωk₁ = Ξ₁ \ B₁
                    Ωk₂ = Ξ₂ \ B₂

                    #Check for convergence
                    dist = max( maximum(abs.( Ωk₁ - Ωkm1[1] )), 
                         maximum(abs.( Ωk₂ - Ωkm1[2] )) )

                    
                    #update values or break
                    if (dist <= tolK) || (i==maxₖ)
                         k_term = i
                         break
                    else
                         Ωkm1[1] = Ωk₁
                         Ωkm1[2] = Ωk₂
                    end

               end
          return Ωk₁, Ωk₂, k_term, Ξ₁, Ξ₂
     end

     Ωk₁,Ωk₂,k_term, Ξ₁, Ξ₂ = f!(Ωk₁, Ωk₂, k_term,Ωkm1,maxₖ,A1₁, A1₂,π_m,B₁,B₂,tolK)
     Φ = [ Ωk₁, Ωk₂ ]
     Ξ = [ Ξ₁, Ξ₂ ]
     return Φ, k_term, Ξ

end

function check_determinancy_fmsre(Φ, Ξ, A1₁, A1₂, Σ1_11, Σ1_12, Σ1_21, Σ1_22; maxK = 1000)
     # This code follows the MATLAB code provided by SeongHoon Cho's website for the paper
     # "Sufficient Conditions and Determinancy in a Class of Markov-Switching Rational
     # Expectations Models" (Cho & Moreno, Review of Economic Dynamics 2016).
     # It computes determinancy of the model in the case of no sunspot solution.
     # The code does not estimate determinacy in the case of a sun spot solutions.
     # All sunspot solutions will output indeterminant.

     S = size(Φ,1)
     n = size(Φ[1],1)
     m = n

     Ã = [A1₁, A1₁, A1₂, A1₂]
     C = [Σ1_11, Σ1_12, Σ1_21, Σ1_22]

     A = Array{Array{Float64}}(undef, S, S)
     for i in 1:S
          for j in 1:S
               A[i,j] = Ã[i] 
          end
     end

     R = Array{Array{Float64}}(undef, S)
     for j in 1:S
          R[j] = zeros(m,m)
     end
      
     FK = Array{Array{Float64}}(undef, S, S) # n x n
     for i=1:S
          for j=1:S
               FK[i,j] = Ξ[s] \ A[i,j] 
          end
     end

     Psi_RtkFK = Array{Float64}(undef, n*m*S,n*m*S)
     for i=1:S
          Psi_RtkFKrow = Array{Float64}(undef, n*m,n*m*S)
          for j=1:S
               Psi_RtkFKrow[:,1+(j-1)*n*m:j*m*n] = P[i,j].*kron(R[j]',FK[i,j])
          end
          Psi_RtkFK[1+(i-1)*n*m:i*m*n,:] = Psi_RtkFKrow
     end
     R_Psi_RtkFK=maximum(abs.(eigen(Psi_RtkFK).values))

     FCC1 = termK
     FCC2 = R_Psi_RtkFK
     FCC = [FCC1 FCC2]

     vvC = Vector{Float64}(undef, n*m*S)
     for i=1:S 
          #InvXiC=Xi(termK,i)\C(i,1) (n x m)
          InvXiC= Ξ[i] \ C[i,1]
          vvC[1+(i-1)*n*m:i*n*m]  = InvXiC[:]  
     end
     vvGamma = (I-Psi_RtkFK) \ vvC
     vGamma = reshape(vvGamma,n*m,S)

     GammaK = Array{Array{Float64}}(undef, S)
     for i=1:S
          GammaK[i] = reshape(vGamma[:,i],n,m) 
     end

     bdiagOm2 = zeros(n^2*S,n^2*S)
     for i=1:S 
          bdiagOm2[n^2*(i-1)+1:n^2*i,n^2*(i-1)+1:n^2*i] = kron(Φ[i],Φ[i])
     end
     BarPsi_OKkOK = bdiagOm2 * kron(P',Matrix{Float64}(I,n^2,n^2)) 
     
     Psi_FKkFK = Array{Float64}(undef,n*n*S,n*n*S) # Psi_[kron(FK,FK)]
     for i=1:S
          Psi_FKkFKrow = Array{Float64}(undef, n*n, n*n*S)
          for j=1:S
               Psi_FKkFKrow[:,1+(j-1)*n*n:j*n*n] = P[i,j].*kron(FK[i,j],FK[i,j])
          end
          Psi_FKkFK[1+(i-1)*n*n:i*n*n,:] = Psi_FKkFKrow
     end
     
     R_BarPsi_OKkOK = maximum(abs.(eigen(BarPsi_OKkOK).values)) 
     R_Psi_FKkFK  = maximum(abs.(eigen(Psi_FKkFK).values)) 
     DET1 = R_BarPsi_OKkOK
     DET2 = R_Psi_FKkFK
     DET = [DET1 DET2]

     #Determine if determinant or not
     if (FCC[1] < maxK) && (DET[1] < 1) && (DET[2] <= 1) 
          det_dum = 1 #Determinant
     else
          det_dum  = 0 #Indeterminant
     end

     return det_dum, FCC, DET
end


function  compute_μ_Σ_fmmsre(Φ, A1₁, A1₂, m1₁, m1₂, Σ1_11, Σ1_12, Σ1_21, Σ1_22, π_m)

     N = size(A1₁)[1]

     m1 = [m1₁; m1₂]
     A₁_star = ( I - A1₁ * ( π_m[1,1]  .*  Φ[1] + π_m[1,2] .* Φ[2] ) )
     A₂_star = ( I - A1₂ * ( π_m[2,1]  .*  Φ[1] + π_m[2,2] .* Φ[2] ) )

     A_star = SMatrix{6,6}([(A₁_star-π_m[1,1].*A1₁) -π_m[1,2].*A1₁;
                     -π_m[2,1].*A1₂ (A₂_star - π_m[2,2].*A1₂) ])

     μ_vec = A_star \ m1
     μ₁ =SMatrix{3,1}(μ_vec[1:N])
     μ₂ = SMatrix{3,1}(μ_vec[N+1:end])

     μ = [μ₁, μ₂]

     Σ_11 = A₁_star \ Σ1_11
     Σ_12 = A₁_star \ Σ1_12
     Σ_21 = A₂_star \ Σ1_21
     Σ_22 = A₂_star \ Σ1_22

     Σ = [Σ_11, Σ_12, Σ_21, Σ_22]

      return μ , Σ
end

@with_kw struct ModelStructuralParams

     Π_0::SMatrix{3,1,Float64,3}
     Π_X::SMatrix{3,3,Float64,9}
     m_g::Float64
     m_r₁::Float64
     m_r₂::Float64
     ϕ::Float64
     δ::Float64
     μ_g::Float64
     μ_pi::Float64
     ρ₁::Float64
     ρ₂::Float64
     β₁::Float64
     β₂::Float64
     α₁::Float64
     α₂::Float64
     σ_g::Float64
     σ_pi::Float64
     σ_r₁::Float64
     σ_r₂::Float64
     σ_y::Float64
     γ_gₐ::Float64
     γ_piₐ::Float64
     γ_cₐ::Float64
     γ_gᵢ::Float64
     γ_piᵢ::Float64
     γ_cᵢ::Float64
     γ_gₒ::Float64
     γ_piₒ::Float64
     γ_cₒ::Float64
     ρₐ::Float64
     ρᵢ::Float64
     ρₒ::Float64
     σ_wₐ::Float64
     σ_zₐ::Float64
     σ_νₐ::Float64
     σ_qₐ::Float64
     σ_wᵢ::Float64
     σ_zᵢ::Float64
     σ_νᵢ::Float64
     σ_qᵢ::Float64
     σ_wₒ::Float64
     σ_zₒ::Float64
     σ_νₒ::Float64
     σ_qₒ::Float64
     λ::Float64
     π_m::SMatrix{2, 2, Float64, 4}
     π_d::SMatrix{2, 2, Float64, 4}

end

@with_kw struct ModelReducedFormParams

     μ::Vector{SMatrix{3,1,Float64,3}}
     μ_Q::Vector{SMatrix{3,1,Float64,3}}
     μ_ν::Vector{SMatrix{6,1,Float64,6}}
     μ_Q_ν::Vector{SMatrix{6,1,Float64,6}}
     Φ::Vector{SMatrix{3,3,Float64,9}}
     Φ_Q::Vector{SMatrix{3,3,Float64,9}}
     Φ_ν::Vector{SMatrix{6,6,Float64,36}}
     Φ_Q_ν::Vector{SMatrix{6,6,Float64,36}}
     Σ::Vector{SMatrix{3,3,Float64,9}}
     Σ_ν::Vector{SMatrix{6,6,Float64,36}}
     cov::Vector{SMatrix{3,3,Float64,9}}
     cov_ν::Vector{SMatrix{6,6,Float64,36}}
     Π::SMatrix{4,4,Float64,16}
     q::Vector{Float64}
     Δ::SMatrix{3, 7, Float64, 21}
     Δ_Q::SMatrix{3, 7, Float64, 21}
     σ_w::SMatrix{3, 1, Float64, 3}
     σ_z::SMatrix{3, 1, Float64, 3}
     σ_q::SMatrix{3, 1, Float64, 3}
     σ_ν::SMatrix{3, 1, Float64, 3}
     σ_y::Float64
     det_dum::Int64
     S::Int64

end

function construct_structural_parameters(params)

     PiX = @SMatrix [ params[4] 0. 0.; params[7] params[5] 0.; params[8] 0. params[6] ]
     Pi0 = SMatrix{3,1}([params[1] params[2] params[3]])
     beta1 = (1-params[16]) \ params[18]
     beta2 = (1-params[17]) \ params[19]
     alpha1 = (1-params[16])\params[20]
     alpha2 = (1-params[17])\params[21]
     pi_m = @SMatrix [ params[52] 1-params[52]; 1-params[53] params[53] ]
     pi_d = @SMatrix [ params[54] 1-params[54]; 1-params[55] params[55] ]
     
     msp = ModelStructuralParams(Π_0 = Pi0, Π_X = PiX, m_g = params[9], m_r₁ = params[10], 
     m_r₂ = params[11], ϕ = params[12], δ = params[13], μ_g = params[14], μ_pi = params[15], ρ₁ = params[16], 
     ρ₂ = params[17], β₁ = beta1, β₂ = beta2, α₁ = alpha1, α₂ = alpha2, σ_g = params[22], σ_pi = params[23], 
     σ_r₁ = params[24], σ_r₂ = params[25], σ_y = params[26], γ_gₐ = params[27], γ_piₐ = params[28], 
     γ_cₐ = params[29], γ_gᵢ = params[30], γ_piᵢ = params[31], γ_cᵢ = params[32], γ_gₒ = params[33], 
     γ_piₒ = params[34], γ_cₒ = params[35], ρₐ = params[36], ρᵢ = params[37], ρₒ = params[38], σ_wₐ = params[39], 
     σ_zₐ = params[40], σ_νₐ = params[41], σ_qₐ = params[42], σ_wᵢ = params[43], σ_zᵢ = params[44], 
     σ_νᵢ = params[45], σ_qᵢ = params[46], σ_wₒ = params[47], σ_zₒ = params[48], σ_νₒ = params[49], 
     σ_qₒ = params[50], λ = params[51], π_m = pi_m, π_d = pi_d)

     return msp

end



function params_to_rf(params)

     msp = construct_structural_parameters(params)

     @unpack Π_0, Π_X, m_g, m_r₁, m_r₂, ϕ, δ, μ_g, μ_pi, ρ₁, ρ₂, β₁, β₂, α₁, α₂, σ_g, σ_pi, 
     σ_r₁, σ_r₂, σ_y, γ_gₐ, γ_piₐ, γ_cₐ, γ_gᵢ, γ_piᵢ, γ_cᵢ, γ_gₒ, γ_piₒ, γ_cₒ, ρₐ, ρᵢ, ρₒ, σ_wₐ, 
     σ_zₐ, σ_νₐ, σ_qₐ, σ_wᵢ, σ_zᵢ, σ_νᵢ, σ_qᵢ, σ_wₒ, σ_zₒ, σ_νₒ, σ_qₒ, λ, π_m, π_d = msp

     b0₁,b0₂,bm1₁,bm1₂,b1₁,b1₂,m0₁,m0₂,Γ₁,Γ₂ = 
          construct_structural_matrices_macro(ϕ, δ, ρ₁, ρ₂, β₁, β₂, α₁, α₂, μ_g, μ_pi, m_g, m_r₁, m_r₂, σ_g, σ_pi, σ_r₁, 
                                                                       σ_r₂)
     
     μ, Φ, μ_Q, Φ_Q, Σ, cov, Π, q, det_dum = 
          transform_struct_to_rf(b0₁, b0₂, bm1₁, bm1₂, b1₁, b1₂, m0₁, m0₂, Γ₁, Γ₂, π_m, π_d, Π_0, Π_X)
     
     μ_ν , Φ_ν, μ_Q_ν, Φ_Q_ν, Σ_ν, cov_ν, Δ, Δ_Q, σ_w, σ_z, σ_q, σ_ν = 
          augment_macro_fsmsre_nu(μ, Φ, μ_Q, Φ_Q, Σ, γ_gₐ, γ_gᵢ, γ_gₒ, γ_piₐ, γ_piᵢ, γ_piₒ, γ_cₐ, γ_cᵢ, γ_cₒ, ρₐ, ρᵢ, ρₒ, 
                          λ, σ_wₐ, σ_wᵢ, σ_wₒ, σ_zₐ, σ_zᵢ, σ_zₒ,σ_qₐ, σ_qᵢ, σ_qₒ, σ_νₐ, σ_νᵢ, σ_νₒ)
     
     S = size(Π)[1]

     mrfp = ModelReducedFormParams(μ = μ, μ_ν = μ_ν, μ_Q = μ_Q, μ_Q_ν = μ_Q_ν, Φ = Φ, Φ_ν = Φ_ν, 
          Φ_Q = Φ_Q, Φ_Q_ν = Φ_Q_ν, Σ = Σ, Σ_ν = Σ_ν, cov = cov, cov_ν = cov_ν, Π = Π, 
          q = q, det_dum = det_dum, Δ = Δ, Δ_Q = Δ_Q, σ_w = σ_w, σ_z = σ_z, σ_q = σ_q, 
          σ_ν = σ_ν, σ_y = σ_y, S = S)

     return mrfp
end

function augment_macro_fsmsre_nu(μ, Φ, μ_Q, Φ_Q, Σ, γ_gₐ, γ_gᵢ, γ_gₒ, γ_piₐ, γ_piᵢ, γ_piₒ, γ_cₐ, γ_cᵢ, γ_cₒ, ρₐ, ρᵢ, 
                                                            ρₒ, λ, σ_wₐ, σ_wᵢ, σ_wₒ, σ_zₐ, σ_zᵢ, σ_zₒ, σ_qₐ, σ_qᵢ, σ_qₒ, σ_νₐ, σ_νᵢ, σ_νₒ)

     S = size(μ)[1]
     N = size(μ[1])[1]
     Δ = @SMatrix [γ_gₐ γ_piₐ 0.; γ_gᵢ γ_piᵢ 0.; γ_gₒ γ_piₒ 0. ]
     ρ = @SMatrix [ρₐ 0. 0.; 0. ρᵢ 0.; 0. 0. ρₒ]
     C = SMatrix{3,1}([γ_cₐ, γ_cᵢ, γ_cₒ])
     σ_z = SMatrix{3,1}([σ_zₐ, σ_zᵢ, σ_zₒ])
     C_Q = C - λ.*σ_z
     σ2_ν = @SMatrix [(σ_wₐ^2 + σ_zₐ^2) σ_zₐ*σ_zᵢ σ_zₐ*σ_zₒ; 
                                    σ_zₐ*σ_zᵢ (σ_wᵢ^2 + σ_zᵢ^2) σ_zᵢ*σ_zₒ; 
                                    σ_zₐ*σ_zₒ  σ_zᵢ*σ_zₒ (σ_wₒ^2 + σ_zₒ^2)]
     σ_ν_L = cholesky(σ2_ν).L

     μ_ν = [ [μ[1]; Δ*μ[1]+C], [μ[2]; Δ*μ[2]+C], [μ[3]; Δ*μ[3]+C], [μ[4]; Δ*μ[4]+C] ] 

     μ_Q_ν = [ [μ_Q[1]; Δ*μ_Q[1]+C_Q], [μ_Q[2]; Δ*μ_Q[2]+C_Q], [μ_Q[3]; Δ*μ_Q[3]+C_Q],
                       [μ_Q[4];Δ*μ_Q[4]+C_Q] ] 

     Φ_ν = [ SMatrix{6,6,Float64,36}([Φ[1] zeros(N,N); Δ*Φ[1] ρ]), 
                  SMatrix{6,6,Float64,36}([Φ[2] zeros(N,N); Δ*Φ[2] ρ]),
                  SMatrix{6,6,Float64,36}( [Φ[3] zeros(N,N); Δ*Φ[3] ρ]),
                  SMatrix{6,6,Float64,36}([Φ[4] zeros(N,N); Δ*Φ[4] ρ])]

     Φ_Q_ν = [ SMatrix{6,6,Float64,36}([Φ_Q[1] zeros(N,N); Δ*Φ_Q[1] ρ]),
                       SMatrix{6,6,Float64,36}([Φ_Q[2] zeros(N,N); Δ*Φ_Q[2] ρ]),
                       SMatrix{6,6,Float64,36}([Φ_Q[3] zeros(N,N); Δ*Φ_Q[3] ρ]),
                       SMatrix{6,6,Float64,36}([Φ_Q[4] zeros(N,N); Δ*Φ_Q[4] ρ])]
     
     Σ_ν = [ SMatrix{6,6,Float64,36}([Σ[1] zeros(N,N); Δ*Σ[1] σ_ν_L]),
                 SMatrix{6,6,Float64,36}( [Σ[2] zeros(N,N); Δ*Σ[2] σ_ν_L]),
                 SMatrix{6,6,Float64,36}([Σ[3] zeros(N,N); Δ*Σ[3] σ_ν_L]),
                 SMatrix{6,6,Float64,36}([Σ[4] zeros(N,N); Δ*Σ[4] σ_ν_L])]

     cov_ν = [Σ_ν[1]*(Σ_ν[1]'),Σ_ν[2]*(Σ_ν[2]'),Σ_ν[3]*(Σ_ν[3]'),Σ_ν[4]*(Σ_ν[4]')]
     
     Δ_tmp = @SMatrix [γ_gₐ γ_piₐ 0. ρₐ 0. 0. γ_cₐ;
                                       γ_gᵢ γ_piᵢ 0. 0. ρᵢ 0. γ_cᵢ;
                                       γ_gₒ γ_piₒ 0. 0. 0. ρₒ γ_cₒ];

     Δ_Q = @SMatrix [γ_gₐ γ_piₐ 0. ρₐ 0. 0. (γ_cₐ-σ_zₐ*λ);
                                   γ_gᵢ γ_piᵢ 0. 0. ρᵢ 0. (γ_cᵢ-σ_zᵢ*λ);
                                   γ_gₒ γ_piₒ 0. 0. 0. ρₒ (γ_cₒ-σ_zₒ*λ) ];

    σ_w = SMatrix{3, 1, Float64, 3}([σ_wₐ; σ_wᵢ; σ_wₒ])
    σ_z = SMatrix{3, 1, Float64, 3}([σ_zₐ; σ_zᵢ; σ_zₒ])    
    σ_q = SMatrix{3, 1, Float64, 3}([σ_qₐ; σ_qᵢ; σ_qₒ])    
    σ_ν = SMatrix{3, 1, Float64, 3}([σ_νₐ; σ_νᵢ; σ_νₒ])                                

     return μ_ν , Φ_ν, μ_Q_ν, Φ_Q_ν, Σ_ν, cov_ν, Δ_tmp, Δ_Q, σ_w, σ_z, σ_q, σ_ν 
end

function split_shock_mat(u_mat, ϵ_mat, w_mat, z_mat, g)

     gk = size(u_mat)[2]
     #T = size(u_mat)[1]
     k = Int(gk / g)

     u = Vector{Array{Float64}}(undef, g)
     ϵ = Vector{Array{Float64}}(undef, g)
     w = Vector{Array{Float64}}(undef, g)
     z = Vector{Array{Float64}}(undef, g)

     @inbounds for gg ∈ 1:g
          u[gg] = u_mat[:,1+(gg-1)*k:gg*k]
          ϵ[gg] = ϵ_mat[:,:,1+(gg-1)*k:gg*k]
          w[gg] = w_mat[:,:,1+(gg-1)*k:gg*k]
          z[gg] = z_mat[:,1+(gg-1)*k:gg*k] 
     end

     return u, ϵ, w, z
end


function split_shock_mat!(u, ϵ, w, z, u_mat, ϵ_mat, w_mat, z_mat, g)

     gk = size(u_mat)[2]
     #T = size(u_mat)[1]
     k = Int(gk / g)

     @inbounds for gg ∈ 1:g
          u[gg] = @view(u_mat[:,1+(gg-1)*k:gg*k])
          ϵ[gg] = @view(ϵ_mat[:,:,1+(gg-1)*k:gg*k])
          w[gg] = @view(w_mat[:,:,1+(gg-1)*k:gg*k])
          z[gg] = @view(z_mat[:,1+(gg-1)*k:gg*k])
     end

     return
end

function simulate_nu_cond_x_i_shock_rn(x_mat,init_ν,mrfp,w,z)

     N = size(x_mat)[1]
     n_ts = size(w)[2]
     G = size(w)[3]
     N_re, N2 = size(init_ν)
     @unpack σ_w, σ_z, Δ_Q = mrfp


     ν_mat = Array{Float64}(undef, N_re, n_ts+1, G)
     if N2==1
          init_ν = repeat(init_ν,1,G)
     end

     ## Scale systematic shocks by std. dev. loading
     w_mat = repeat(σ_w, 1, n_ts, G) .* w

     tmp_z_mat = Array{Float64}(undef, N_re, n_ts, G)
     @inbounds for g ∈ 1:G
          tmp_z_mat[:, :, g] = repeat(z[:,g]', N_re)
     end
     z_mat = repeat(σ_z, 1, n_ts, G) .* tmp_z_mat
     wz_mat = w_mat + z_mat
               
     @inbounds for g=1:G
          ν_mat[:,1,g] = @view(init_ν[:, g])
          wz_path = @view(wz_mat[:, :, g])
          ##Create X_mat = [x_t; nu_{t-1}; 1]'; an (N + N_re + 1) x n_ts Array
          X_mat = Array{Float64}(undef, N+N_re+1, n_ts+1)
          X_mat[1:N, 1:end-1] = @view(x_mat[:, 2:end, g])
          X_mat[end,:] = ones(1, n_ts+1)
          X_mat[N+1:N+N_re, 1] = @view(ν_mat[:, 1, g])
          
          ##Compute nu_t iteratively 
          @inbounds for i=1:n_ts
               X_mat[N+1:N+N_re, i+1] = Δ_Q * @view(X_mat[:, i]) + @view(wz_path[: ,i])
          end

          ν_mat[:, 2:end, g] = @view(X_mat[N+1:N+N_re, 2:end])

     end

     return ν_mat
end

function compute_mc_real_estate_Q_cond_x_i_nofull(x_mat, ν_mat)
     T = size(x_mat)[2] - 1
     n_η = T
     G = size(x_mat)[3]
     N_re = size(ν_mat)[1]
     #T_ν = size(ν_mat)[2] - 1

     ## Preallocate
     Δ_ν = [0. 0. -1. 1. 0. 0.;
                                   0. 0. -1. 0. 1. 0.;
                                   0. 0. -1. 0. 0. 1.]

     η_sim = Array{Float64}(undef, N_re, T, G)
     m1_η_sim = Array{Float64}(undef, N_re, T, G)
     #m2_η_sim = Array{Float64}(undef, N_re, T, G)
     x_path_mat = [x_mat; ν_mat]

     @inbounds for g=1:G
          x_path = @view(x_path_mat[:,1:T,g])
          m1_η_sim[:, :, g] = cumsum(Δ_ν * x_path, dims = 2)
          #m2_η_sim[:, :, g] = @view(m1_η_sim[:, :, g]).^2
          #η_sim[:, :, g] = exp.(@view(m1_η_sim[:, :, g]))
     end
     #m2_η_sim = m1_η_sim .^ 2
     η_sim .= exp.(m1_η_sim)

     η_mc = dropdims(mean(η_sim, dims = 3), dims = 3)
     #m1_η_mc = dropdims(mean(m1_η_sim, dims = 3), dims = 3)'
     #m2_η_mc = dropdims(mean(m2_η_sim, dims = 3), dims = 3)'
     Q_mc = sum(η_mc[:,1:n_η], dims = 2)
     #log_Q_mc = mean(log.(dropdims(sum(η_sim, dims = 2), dims = 2)), dims = 2)
     
     #Get std mat
     N_std = Int64(floor(G / 30))
     Q_std_prep = Array{Float64}(undef, N_re, 30)
     #log_Q_std_prep = Array{Float64}(undef, N_re, 30)

     η_std_prep = Array{Float64}(undef, N_re, T, 30)
     @inbounds for i=1:30
          η_std_prep[:, :, i] =  dropdims(mean(@view(η_sim[:, :, 1+(i-1)*N_std:i*N_std]), dims = 3), dims = 3)
          Q_std_prep[:, i] = sum(@view(η_std_prep[:, 1:n_η, i]), dims = 2)
          #log_Q_std_prep[:, i] = mean(log.(dropdims(sum(@view(η_sim[:, :, 1+(i-1)*N_std:i*N_std]), dims = 2), dims = 2)),
          #                                               dims = 2)
     end

     #η_std_mc = dropdims(std(η_std_prep, dims = 3), dims = 3) ./ sqrt(30)
     Q_std_mc = std(Q_std_prep, dims = 2) ./ sqrt(30)
     #log_Q_std_mc = std(log_Q_std_prep, dims = 2) ./ sqrt(30)
     
     return Q_mc, η_mc, Q_std_mc, Q_std_prep

end

function compute_mc_term_structure_cond_x(x_mat, maturity_mat)
     G = size(x_mat)[3]
     N_terms = size(maturity_mat)[2]

     δ = [0. 0. -1.]
     ts_sim = Array{Float64}(undef, N_terms, G)
     m1_ts_sim = Array{Float64}(undef, N_terms,G)
     m2_ts_sim = Array{Float64}(undef, N_terms,G)

     @inbounds for g=1:G
          x_path = @view(x_mat[:, :, g])
          @inbounds for n=1:N_terms
             m1_ts_sim[n,g] = sum(δ *@view(x_path[:, 1:maturity_mat[n]]), dims = 2)[1]
          end
     end
     m2_ts_sim = m1_ts_sim.^2
     ts_sim = exp.(m1_ts_sim)
       
     ts_mc = -log.(mean(ts_sim, dims = 2)) ./ maturity_mat'
     m1_ts_mc = mean(m1_ts_sim, dims = 2);
     m2_ts_mc = mean(m2_ts_sim, dims = 2);

     N_std = Int64(floor(G/30))
     ts_std_prep = Array{Float64}(undef, N_terms, 30)
     @inbounds for i=1:30
          ts_std_prep[:, i] =  -log.(mean(@view(ts_sim[:, 1+(i-1)*N_std:i*N_std]), dims = 2)) ./ maturity_mat'
     end
     ts_std_mc = std(ts_std_prep, dims = 2) ./ sqrt(30)

     return ts_mc, ts_std_mc, m1_ts_mc, m2_ts_mc, ts_std_prep
end

function   get_total_mean_and_std(Q_mc_cell, eta_mc_cell, Q_std_prep_cell)

     Q_mc_tot = compute_mean_over_subsamples(Q_mc_cell)
     eta_mc_tot = compute_mean_over_subsamples(eta_mc_cell)

     Q_std_mc_tot = compute_std_over_subsamples(Q_std_prep_cell)

     return Q_mc_tot, eta_mc_tot, Q_std_mc_tot 
end

function compute_std_over_subsamples(in_cell_std)
     L = size(in_cell_std)[1]
     N1, N2 = size(in_cell_std[1])


     if L>1
          in_std_mat = dropdims(sum(reshape(reduce(hcat, in_cell_std), N1, N2, L), dims = 3) ./ L, dims = 3)
      else
          in_std_mat = in_cell_std[1]
      end
      
     b = size(in_std_mat)[2]
     out_std_mat = std(in_std_mat, dims = 2) ./ sqrt(b)

     return out_std_mat

end

function compute_mean_over_subsamples(in_cell)

     n,m = size(in_cell[1])
     L = size(in_cell)[1]
     
     if m==1
         mean_mat = mean(reduce(hcat,in_cell), dims = 2)
     else
         mean_mat =  dropdims(sum(reshape(reduce(hcat, in_cell), n, m, L), dims = 3) ./ L, dims = 3)
     end

     return mean_mat

end

function simulate_model_prices_cond_shock_acc_ts(x_init, ν_init, mrfp, T_sim, u, ϵ, w, z, tol, del, η_tol, t, s, 
                                                                                      n_grps, L)

     
     #tmr = TimerOutput()
     # Unload inputs
     N_macro = 3
     N_re = 3
     M = size(u)[1]
     G = size(ϵ)[3]
     #@assert floor(G / L)== (G/L)
     gg = Int64(G / L)

     @unpack μ_Q, Φ_Q, Σ, Π = mrfp

     Q_mc = Array{Float64}(undef, N_re, 1)
     Q_std_mc = Array{Float64}(undef, N_re, 1)
     η_mc= Array{Float64}(undef, N_re, M, 1)

     #=
     @timeit tmr "split_shock_mats" begin
          u_cell = Vector{Array{Float64,2}}(undef, L)
          ϵ_cell = Vector{Array{Float64,3}}(undef, L)
          w_cell = Vector{Array{Float64,3}}(undef, L)
          z_cell = Vector{Array{Float64,2}}(undef, L)
          split_shock_mat!(u_cell, ϵ_cell, w_cell, z_cell, u, ϵ, w, z, L)
     end
     =#
     X = Array(Array(x_init[:, t]')')
     ν = Array(Array(ν_init[:, t]')')
     trig = 0
     l = 1

     Q_mc_cell = Vector{Array{Float64,2}}(undef, L)
     Q_std_mc_cell = Vector{Array{Float64,2}}(undef, L)
     η_mc_cell = Vector{Array{Float64,2}}(undef, L)
     Q_std_prep_cell = Vector{Array{Float64,2}}(undef, L)
     x_mat = Array{Float64}(undef, N_macro, T_sim+1, gg)
     s_mat = Array{Int64}(undef, T_sim+1, gg)
     ν_mat = Array{Float64}(undef, N_re, T_sim+1, gg)

     T_bar = 0

     @inbounds while ((trig==0) && (l<=L)) #early simulation check

          if l==1

               #Detect early truncation point, T_bar
               trig_2 = 0
               cnt_2 = 1
               T_tmp = Int64(T_sim ./ n_grps)
               x_mat[:, 1, :] = repeat(X, 1, 1, gg)
               ν_mat[:, 1, :] = repeat(ν, 1, 1, gg)
               s_mat[1, :] .= s
              
               local Q_mc_tmp
               local Q_std_mc_tmp
               local Q_std_prep 
               @inbounds while (trig_2==0)&&(cnt_2<=n_grps)

                    tmp_ndx = (1+(cnt_2-1)*T_tmp):(cnt_2*T_tmp)

                    #@timeit tmr "inner_sim_msvar" begin
                         #Simulate model for length T_tmp
                         #tmp_x_mat, tmp_s_mat = simulate_ms_var_1_cond_shocks(x_mat[:, tmp_ndx[1], :],
                         #     s_mat[tmp_ndx[1], :], μ_Q, Φ_Q, Σ, Π, u_cell[l][tmp_ndx, :], ϵ_cell[l][: ,tmp_ndx, :])
                         tmp_x_mat, tmp_s_mat = simulate_ms_var_1_cond_shocks(x_mat[:, tmp_ndx[1], :],
                             s_mat[tmp_ndx[1], :], μ_Q, Φ_Q, Σ, Π, u[tmp_ndx, 1+(l-1)*gg:l*gg],
                             ϵ[: ,tmp_ndx, 1+(l-1)*gg:l*gg])
                    #end
                   # 
                    #@timeit tmr "inner_sim_nu" begin
                         #Simulate \nu
                         #tmp_nu_mat = simulate_nu_cond_x_i_shock_rn(tmp_x_mat, ν_mat[:, tmp_ndx[1], :],
                         #     mrfp, w_cell[l][:, tmp_ndx, :], z_cell[l][tmp_ndx, :])
                         tmp_nu_mat = simulate_nu_cond_x_i_shock_rn(tmp_x_mat, ν_mat[:, tmp_ndx[1], :],
                              mrfp, @view(w[:, tmp_ndx, 1+(l-1)*gg:l*gg]), @view(z[tmp_ndx, 1+(l-1)*gg:l*gg]))
                   #end

                    #Store simulations thus far
                    x_mat[:, tmp_ndx[1]+1:tmp_ndx[end]+1, :] = tmp_x_mat[:, 2:end, :]
                    ν_mat[:, tmp_ndx[1]+1:tmp_ndx[end]+1, :] = tmp_nu_mat[:, 2:end, :]
                    s_mat[tmp_ndx[1]+1:tmp_ndx[end]+1 ,:] = tmp_s_mat[2:end, :]

                    #@timeit tmr "compute_Q" begin
                         #Simulate Q
                         Q_mc_tmp, η_mc_tmp, Q_std_mc_tmp, Q_std_prep = 
                              compute_mc_real_estate_Q_cond_x_i_nofull(@view(x_mat[:, 1:tmp_ndx[end]+1, :]),
                                                                                                    @view(ν_mat[:, 1:tmp_ndx[end]+1, :]))
                   # end
                    #Check that all η_{T_sim,x_t)< cutoff for all t
                    if (~any(η_mc_tmp[:, end] .>= η_tol)) || (cnt_2==n_grps)

                         #Set loop trigger and T_bar
                         T_bar = cnt_2*T_tmp
                         trig_2 = 1

                         #Simulate Term_structure
                         #ts_mc_tmp, ts_std_mc_tmp, m1_ts_tmp, m2_ts_tmp, ts_std_prep_mat = 
                         #     compute_mc_term_structure_cond_x(x_mat, maturity_mat)

                    else

                         cnt_2 +=  1

                    end
               end

          else
               #Run simulation for T_bar periods
               # Simulate Macro model
          
               #@timeit tmr "outer_sim_msvar" begin
                    #x_mat, s_mat = simulate_ms_var_1_cond_shocks(X, s, μ_Q, Φ_Q, Σ,Π,u_cell[l][1:T_bar, :],
                    #                                                                                     ϵ_cell[l][:, 1:T_bar, :])
                    x_mat, s_mat = simulate_ms_var_1_cond_shocks(X, s, μ_Q, Φ_Q, Σ,Π,@view(u[1:T_bar, 1+(l-1)*gg:l*gg]),
                                                                                                         @view(ϵ[:, 1:T_bar, 1+(l-1)*gg:l*gg]))
               #end

               #@timeit tmr "outer_sim_nu" begin
               #Simulate \nu
               #ν_mat = simulate_nu_cond_x_i_shock_rn(x_mat, ν, mrfp ,w_cell[l][:, 1:T_bar, :], z_cell[l][1:T_bar, :])
               ν_mat = simulate_nu_cond_x_i_shock_rn(x_mat, ν, mrfp ,@view(w[:, 1:T_bar,  1+(l-1)*gg:l*gg]), 
                                                                                    @view(z[1:T_bar,  1+(l-1)*gg:l*gg]))

               #end

               #Given simulation-approximate term-structure
               #ts_mc_tmp, ts_std_mc_tmp, m1_ts_tmp, m2_ts_tmp, ts_std_prep_mat =
               #     compute_mc_term_structure_cond_x(x_mat, maturity_mat)

               #@timeit tmr "outer_compute_Qr" begin
                    #Approximate real estate Q
                    Q_mc_tmp, η_mc_tmp, Q_std_mc_tmp, Q_std_prep = 
                         compute_mc_real_estate_Q_cond_x_i_nofull(x_mat, ν_mat)
               #end
          end

          #Store in cell
          Q_mc_cell[l] = Q_mc_tmp
          η_mc_cell[l] = η_mc_tmp
          Q_std_mc_cell[l] = Q_std_mc_tmp
          Q_std_prep_cell[l] = Q_std_prep
          
          #Compute total mean and variance

          Q_mc_tot, η_mc_tot, Q_std_mc_tot =
               get_total_mean_and_std(Q_mc_cell[1:l],η_mc_cell[1:l],Q_std_prep_cell[1:l])
    
          #Check condition
          if (!any((2 .* (1. .- normcdf.((1.2 .* tol .* Q_mc_tot) ./ Q_std_mc_tot)) .> del)) || (l==L) )

             trig = 1;
             #println("l: $l")
             #println("T_bar: $T_bar")

             #If yes, store results
             Q_mc[:, 1] = Q_mc_tot
             η_mc[:, 1:T_bar] = η_mc_tot
             Q_std_mc[:, 1] = Q_std_mc_tot

          else

              l = l+1;

          end
    
     end

     #println(tmr)
     return Q_mc, η_mc, Q_std_mc, T_bar 

end

@with_kw struct MonteCarloStruct

     T::Int64
     S::Int64
     N_re::Int64
     N_θ::Int64
     Q_A_mat::Matrix{Float64}
     Q_I_mat::Matrix{Float64}
     Q_O_mat::Matrix{Float64}
     Q_A_std_mat::Matrix{Float64}
     Q_I_std_mat::Matrix{Float64}
     Q_O_std_mat::Matrix{Float64}
     cap_data::Matrix{Float64}
     ν_data::Matrix{Float64}
     Y_data::Matrix{Float64}
     Macro_data::Matrix{Float64}

end

function construct_mc_struct(T, S, N_re, Q_A_mat, Q_I_mat, Q_O_mat, Q_A_std_mat, Q_I_std_mat, 
     Q_O_std_mat, cap_data, ν_data, Y_data, Macro_data, N_θ)

     mcs = MonteCarloStruct(T = T, S = S, N_re = N_re, Q_A_mat = Q_A_mat, Q_I_mat = Q_I_mat, 
          Q_O_mat = Q_O_mat, Q_A_std_mat = Q_A_std_mat, Q_I_std_mat = Q_I_std_mat, 
          Q_O_std_mat = Q_O_std_mat, cap_data = cap_data, ν_data = ν_data, Y_data = Y_data, 
          Macro_data = Macro_data,N_θ = N_θ)

     return mcs

end


function mc_loglik_numint_1d(θ, yield_mat, macro_mat, cap_mat, ν_mat, scale_vec, 
     filter_dum, Q_mc_mat, σ_mc_mat)


     # θ to reduced-form model solution
     rf_struct = params_to_rf(θ ./ scale_vec)

     # Process data 
     data_struct = process_data(yield_mat, macro_mat, cap_mat, ν_mat)

     # Check for feasibility 
     feas_dum = is_feasible(rf_struct)
     if feas_dum==0
          neg_log_posterior = Inf
          fp = adjoint(Matrix{Float64}(undef,1,1))
          sp = NaN
          log_lik = Inf
          #log_prior = Inf
     else
     
          sp = NaN

          # rf_model to model predictions 
          pred_struct = make_model_predictions(data_struct, rf_struct, Q_mc_mat, σ_mc_mat)

          # rf_model, model_predictions, data --> log-likihood (todo)
          neg_log_lik, fp, ξ̂_tt, ξ̂_ttm1 = compute_model_neg_log_lik(pred_struct, data_struct, 
                                             rf_struct)
          
          #log_prior = log_prior_fn(θ)
          #neg_log_posterior = -log_prior + neg_log_lik
          log_lik = -neg_log_lik
     end
     return neg_log_lik, fp, sp, log_lik

end

function is_feasible(rf_struct)

     feas_dum = 1
     @unpack Φ, det_dum = rf_struct
     Φ₁ = Φ[1]
     Φ₂ = Φ[3]

     if det_dum == 0
          feas_dum = 0
     else
          if (any(Φ₁.==NaN))||(any(Φ₂.==NaN))
               feas_dum = 0
          end
     end

     return feas_dum
end

@with_kw struct DataStruct
     
     Y_macro::Matrix{Float64}
     Y0_macro::Matrix{Float64}
     Y1_macro::Matrix{Float64}
     Y_ts::Matrix{Float64}
     Y0_ts::Matrix{Float64}
     Y1_ts::Matrix{Float64}
     Y_cap::Matrix{Float64}
     Y0_cap::Matrix{Float64}
     Y1_cap::Matrix{Float64}
     Y_ν::Matrix{Float64}
     Y0_ν::Matrix{Float64}
     Y1_ν::Matrix{Float64}
     Y_q::Matrix{Float64}
     Y0_q::Matrix{Float64}
     Y1_q::Matrix{Float64}
     n_re::Int64
     n_yields::Int64
     n_macro::Int64
     n_ts::Int64
     T::Int64
end


function process_data(yield_data,macro_data,cap_data,ν_data)

     num_re = 3

     # Get matrix dimensions
     T, n_Y = size(yield_data)
     n_M = size(macro_data)[2]
     #n_re = size(cap_data)[2]

     #Make level and quarterly
     yield_data = yield_data ./ 400
     macro_data = macro_data ./ 400
     cap_data = cap_data ./ 400

     #Unload data
     Y_macro = [macro_data yield_data[:,1]]
     Y_ts = yield_data[:,2:n_Y]
     Y_ν = ν_data[:,1:num_re]
     Y_cap = cap_data
     Y_q = 1 ./ cap_data

     #Create lag matrix (TODO)
     Y0_macro,Y1_macro = create_lag_matrix(Y_macro)
     Y0_ts,Y1_ts = create_lag_matrix(Y_ts)
     Y0_cap,Y1_cap = create_lag_matrix(Y_cap)
     Y0_ν,Y1_ν = create_lag_matrix(Y_ν)
     Y0_q,Y1_q = create_lag_matrix(Y_q)

     data_struct = DataStruct(Y_macro = Y_macro, Y0_macro = Y0_macro, Y1_macro = Y1_macro,
          Y_ts = Y_ts, Y0_ts = Y0_ts, Y1_ts = Y1_ts, Y_cap = Y_cap, Y0_cap = Y0_cap,
          Y1_cap = Y1_cap, Y_ν = Y_ν, Y0_ν = Y0_ν, Y1_ν = Y1_ν, Y_q = Y_q, Y0_q = Y0_q,
          Y1_q = Y1_q, n_re = num_re, n_yields = n_Y - 1, n_macro = n_M + 1, n_ts = T, 
          T = T - 1)

     return data_struct
end

function create_lag_matrix(Y)

     Y0 = Y[2:end,:]
     Y1 = Y[1:end-1,:]

     return Y0, Y1
end

@with_kw struct ModelPredictionStruct

     Y0_hat_macro::Matrix{Float64}
     Y0_hat_ν::Matrix{Float64}
     Y0_hat_ts::Matrix{Float64}
     Y0_hat_q::Matrix{Float64}
     Y0_hat_q_std::Matrix{Float64}

end

function make_model_predictions(data_struct, rf_struct, Q_mc_mat, std_mc_mat)

     #unload structures
     @unpack Y1_macro, Y0_macro, Y0_ν, n_yields, T = data_struct
     @unpack Π = rf_struct
     S = size(Π)[1]

     #Given Y1_macro, predict Y0_macro 
     Y0_hat_macro = project_macro_model(Y1_macro, rf_struct)

     #Given Y0_macro predict NOI growth 
     Y0_hat_ν = project_ν_cond_macro(data_struct, rf_struct)

     #Given Y0_macro predict term-structure using quadratic approx 
     Y0_macro_ν = [Y0_macro Y0_ν]
     Y_hat_ts = approximate_model_prices(Y0_macro_ν', rf_struct)

     Y0_hat_ts = Array{Float64}(undef, T, n_yields * S)
     for s ∈ 1:S
          Y0_hat_ts[:,1+(s-1)*n_yields:s*n_yields] = Y_hat_ts[:,:,s]'
     end

     Y0_hat_q = Q_mc_mat[2:end,:]
     Y0_hat_q_std = std_mc_mat[2:end,:]

     pred_struct = ModelPredictionStruct(Y0_hat_macro = Y0_hat_macro, Y0_hat_ν = Y0_hat_ν,
          Y0_hat_ts = Y0_hat_ts, Y0_hat_q = Y0_hat_q, Y0_hat_q_std = Y0_hat_q_std)

     return pred_struct
end

function project_macro_model(Y, rf_struct)

     @unpack S, μ, Φ = rf_struct
     Y_hat = project_msvar(Y, μ, Φ, S)

     return Y_hat
end

function project_msvar(Y, μ, Φ, S)

     T,N = size(Y)

     B = Array{Float64}(undef, S*N, N+1)
     for s ∈ 1:S
          B[1+(s-1)*N:s*N,:] = [μ[s] Φ[s]]
     end

     Y_bar = [ones(1,T); Y']

     Ŷ = (B*Y_bar)'

     return Ŷ
end

function project_ν_cond_macro(data_struct, rf_struct)

     #unpack structures
     @unpack Y0_macro, Y1_ν = data_struct
     @unpack Δ = rf_struct

     #Δ = @SMatrix [γ_gₐ γ_piₐ 0. ρₐ 0. 0. γ_cₐ;
     #                  γ_gᵢ γ_piᵢ 0. 0. ρᵢ 0. γ_cᵢ;
     #                  γ_gₒ γ_piₒ 0. 0. 0. ρₒ γ_cₒ];

     C = Δ[:,end]
     B_macro = Δ[:,1:3]
     B_ν = Δ[:,4:6]

     Y0_hat_ν = (C .+ B_macro*(Y0_macro') + B_ν*(Y1_ν'))'

     return Y0_hat_ν   
     
end

function approximate_model_prices(x_mat, rf_struct)
#    Currently only approximates term structure     
#    TODO: Approximate Q using quadratic approximation



     #Get term structure factor loadings
     maturity_mat = [8 20 40]

     ts_factor_loading_struct = 
          get_term_structure_quadratic_pricing_factors(maturity_mat, rf_struct)

     ts_mat, m1_ts_mat, m2_ts_mat = approximate_term_structure_cond_x(x_mat[1:3,:], 
          ts_factor_loading_struct, maturity_mat)

     return ts_mat

end

function get_term_structure_quadratic_pricing_factors(maturity_mat, rf_struct)

     @unpack Π, μ_Q, Φ_Q, cov = rf_struct
     δ = @SMatrix [0. 0. -1.]

     A_cell, B_cell, C_cell, D_cell, F_cell = compute_quadratic_pricing_factors_msvar(δ,
          maturity_mat', Π, μ_Q, Φ_Q, cov)

     ts_factor_loading_struct = TermStructureFactorLoadingStruct(A_ts = A_cell,
          B_ts = B_cell, C_ts = C_cell, D_ts = D_cell, F_ts = F_cell)

     return ts_factor_loading_struct

end

@with_kw struct TermStructureFactorLoadingStruct

     A_ts::Matrix{Float64}
     B_ts::Array{SMatrix{3,1,Float64,3}}
     C_ts::Array{Float64}
     D_ts::Array{SMatrix{3, 1, Float64, 3}}
     F_ts::Array{SMatrix{3, 3, Float64, 9}}

end

function compute_quadratic_pricing_factors_msvar(δ, cf_mat, Π, μ_Q, Φ_Q, cov)
     
     #Unload inputs and preallocate
     N = size(μ_Q[1])[1]
     n_cf = size(cf_mat)[1]
     max_n = maximum(cf_mat)
     S = size(Π)[1]

     A_cell = Array{Float64}(undef, S, n_cf)
     B_cell = Array{SMatrix{N, 1, Float64, N}}(undef, S, n_cf)
     C_cell = Array{Float64}(undef, S, n_cf)
     D_cell = Array{SMatrix{N, 1, Float64, 3}}(undef, S, n_cf)
     F_cell = Array{SMatrix{N, N, Float64, N*N}}(undef, S, n_cf)
     
     tmp_A = zeros(S,1)
     tmp_B = zeros(N,S)
     tmp_C = zeros(S,1)
     tmp_D = zeros(N,S)
     tmp_F = zeros(N,N,S)

     δ = δ'
     i_cf = 1
     for cf ∈ 1:max_n
          if cf > 1
               A_term_mat = Array{Float64}(undef, S, S)
               B_term_mat = Array{Float64}(undef, N, S, S)
               C_term_mat = Array{Float64}(undef, S, S)
               D_term_mat = Array{Float64}(undef, N, S, S)
               F_term_mat = Array{Float64}(undef, N, N, S, S)

               for j ∈ 1:S #s_{t+1}
                    μⱼ = μ_Q[j]
                    Φⱼ = Φ_Q[j]
                    A_term = tmp_A[j] .+ μⱼ'*tmp_B[:,j]
                    B_term = Φⱼ'*tmp_B[:,j]
                    C_term = tmp_C[j] .+ μⱼ'*tmp_D[:,j] .+ μⱼ'*tmp_F[:,:,j]*μⱼ .+ 
                         tr(tmp_F[:,:,j]*cov[j])
                    D_term = Φⱼ' * (tmp_D[:,j] + 2. .* tmp_F[:,:,j]' * μⱼ)
                    F_term = Φⱼ' * tmp_F[:,:,j] * Φⱼ

                    for i ∈ 1:S #s_t
                         A_term_mat[i,j] = Π[i,j] ⋅ A_term
                         B_term_mat[:,i,j] =  Π[i,j] .* B_term
                         C_term_mat[i,j] = Π[i,j] ⋅ C_term
                         D_term_mat[:,i,j] =  Π[i,j] .* D_term
                         F_term_mat[:,:,i,j] = Π[i,j] .* F_term

                    end
               end

               A_term_i = sum(A_term_mat,dims = 2)
               B_term_i = dropdims(sum(B_term_mat, dims = 3), dims = 3)
               C_term_i = sum(C_term_mat, dims = 2)
               D_term_i = dropdims(sum(D_term_mat, dims = 3), dims = 3)
               F_term_i = dropdims( sum(F_term_mat, dims = 4), dims = 4)

               for i=1:S
                    A_cell[i,i_cf] = A_term_i[i]
                    B_cell[i,i_cf] = δ .+ B_term_i[:,i]
                    C_cell[i,i_cf] = C_term_i[i]
                    D_cell[i,i_cf] = 2. .* A_term_i[i] * δ .+  D_term_i[:,i]
                    F_cell[i,i_cf] = -δ*δ' + δ*B_cell[i,i_cf]' + B_cell[i,i_cf]*δ' + 
                                     F_term_i[:,:,i]
               end

          else

               for i ∈ 1:S
                    A_cell[i,i_cf] = 0
                    B_cell[i,i_cf] = δ
                    C_cell[i,i_cf] = 0
                    D_cell[i,i_cf] = @SMatrix zeros(N,1)
                    F_cell[i,i_cf] = δ*δ'
               end

          end

          for i ∈ 1:S
               tmp_A[i] = A_cell[i,i_cf]
               tmp_B[:,i] = B_cell[i,i_cf]
               tmp_C[i] = C_cell[i,i_cf]
               tmp_D[:,i] = D_cell[i,i_cf]
               tmp_F[:,:,i] = F_cell[i,i_cf]
          end

          if cf_mat[i_cf] == cf
               i_cf += 1
          end

     end


     return A_cell, B_cell, C_cell, D_cell, F_cell

end

function approximate_term_structure_cond_x(x_mat,ts_factor_struct,maturity_mat)

     T = size(x_mat)[2]
     @unpack A_ts, B_ts, C_ts, D_ts, F_ts = ts_factor_struct
     S = size(A_ts)[1]

     m1_mat = compute_m1_cond_x(x_mat, A_ts, B_ts)
     m2_mat = compute_m2_cond_x(x_mat, C_ts, D_ts, F_ts)

     ts_mat = -(m1_mat + 0.5 .* (m2_mat - m1_mat.^2)) ./ repeat(maturity_mat',1,T,S)

     return ts_mat, m1_mat, m2_mat 
end

function compute_m1_cond_x(x_mat, A, B)

     N,T = size(x_mat)
     S,N_terms = size(A)

     m1_mat = Array{Float64}(undef, N_terms, T, S)

     y_mat = [ones(T,1) x_mat']

     bar_y_1 = reshape(A', 1, N_terms*S)
     bar_y_2 = dropdims(convert(Array,VectorOfArray(vec(reshape(permutedims(B),1, N_terms*S)
          ))), dims = 2)
     bar_y = [bar_y_1; bar_y_2]

     m1_tmp_mat = y_mat * bar_y

     for s ∈ 1:S
        m1_mat[:,:,s] = m1_tmp_mat[:,1+(s-1)*N_terms:N_terms*s]'
     end

     return m1_mat

end

function compute_m2_cond_x(x_mat, C, D, F)

     N,T = size(x_mat)
     S,N_terms = size(C)
     m2_mat = Array{Float64}(undef, N_terms, T, S)

     #Define y_mat matrix: [1, x_t', vec(x_t * x_t')']_t (T x (1 + N + N^2))
     y_mat = Array{Float64}(undef, T, 1 + N + N^2)
     y_mat[:,1] .= 1.
     y_mat[:,2:N+1] = x_mat'

     for t ∈ 1:T
          y_mat[t,N+2:end] = vec(x_mat[:,t]*x_mat[:,t]')'
     end

     #Define y_bar matrix: ( (1 + N + N^2) x (N_terms*S) )
     bar_y = Array{Float64}(undef, 1 + N + N^2, N_terms*S)
     bar_y[1,:] = reshape(C', 1, N_terms*S)
     bar_y[2:N+1,:] = dropdims(convert(Array,VectorOfArray(vec(reshape(permutedims(D),1,
          N_terms*S)))), dims = 2)
     #bar_y = [bar_y_1; bar_y_2]
     
     cnt = 1
     for s ∈ 1:S
          for n ∈ 1:N_terms
               bar_y[N+2:end,cnt] = vec(F[s,n]')
               cnt += 1
          end
     end

     #Compute m2_mat
     m2_tmp_mat = y_mat * bar_y

     for s ∈ 1:S
          m2_mat[:,:,s] = m2_tmp_mat[:,1+(s-1)*N_terms:N_terms*s]'
     end


     return m2_mat
end


function compute_model_neg_log_lik(pred_struct, data_struct, rf_struct)
     
     #Unload structures
     @unpack Y0_macro, Y0_ν, Y0_ts, Y0_q, T, n_macro, n_yields, n_re = data_struct
     @unpack Y0_hat_macro, Y0_hat_ν, Y0_hat_ts, Y0_hat_q, Y0_hat_q_std = pred_struct
     @unpack S, Π, σ_y, cov, q = rf_struct

     #Compute Residuals: Y0 - Y0_hat
     ϵ_m = 100. .* ( repeat(Y0_macro, 1, S) - Y0_hat_macro)
     ϵ_ν = 100. .* (repeat(Y0_ν - Y0_hat_ν, 1, S))
     ϵ_y = 100. .* (repeat(Y0_ts, 1, S) - Y0_hat_ts)
     #ϵ_Q = repeat(log.(Y0_q), 1, S) - log.(Y0_hat_q)

     Q_mat = repeat(Y0_q, 1, S)
     Qhat_mat = Y0_hat_q

     #Precomputation
     Σ_re = construct_cov_re(rf_struct)
     Σ_ν = (100.0^2) .* Σ_re[4:6,4:6]
     Σ_q = Σ_re[1:3,1:3]
     σ_q = sqrt.(diag(Σ_q))
     twoP = 2. * π
     #srp2REfac = (twop)^-(n_re/2)
     log2p = log(twoP)
     logSrpMacrofac = (-n_macro/2)*log2p
     logSrpYieldsfac = (-n_yields/2)*log2p
     logSrpNufac = -(n_re/2)*log2p

     eye_s = Matrix(I, S, S)
     eye_macro = Matrix(I, n_macro, n_macro)
     ones_S = ones(1,S)

     macro_ll_cons = Array{Float64}(undef, S, 1)
     Σ_m = (100.0^2) .* reshape(convert(Array, VectorOfArray(cov)), (3,12))
     for s ∈ 1:S
          cov_m = Σ_m * kron(eye_s[:,s], eye_macro)
          macro_ll_cons[s] = logSrpMacrofac - sum(log.(diag(cholesky(cov_m).U)))
     end
     σ_y² = (100.0^2) .* (σ_y^2)*Matrix(I,n_yields, n_yields)
     yields_ll_cons = logSrpYieldsfac - sum(log.(diag(cholesky(σ_y²).U)))
     ν_ll_cons = logSrpNufac - sum(log.(diag(cholesky(Σ_ν).U)))

     F = Π'

     #Intialize

     ξ̂_ttm1 = Array{Float64}(undef, S, T+1)
     ξ̂_tt = Array{Float64}(undef, S, T+1)
     η = Array{Float64}(undef, S, T)
     lik_mat = Array{Float64}(undef, T, 1)
     ξ̂_ttm1[:, 1] = q
     ξ̂_tt[:, 1] = q

     #Filter
     @inbounds for t ∈ 1:T
          @inbounds for s ∈ 1:S
               Is = @view eye_s[:,s]
               cov_m = Σ_m * kron(Is, eye_macro)
               ε_m = @view ϵ_m[t,n_macro*(s-1)+1:n_macro*s]
               ε_y = @view ϵ_y[t,n_yields*(s-1)+1:n_yields*s]
               Q = @view Q_mat[t,n_re*(s-1)+1:n_re*s]
               Q̄ = @view Qhat_mat[t,n_re*(s-1)+1:n_re*s]
               Q̄_std = @view Y0_hat_q_std[t,n_re*(s-1)+1:n_re*s]
               ε_ν = @view ϵ_ν[t,n_re*(s-1)+1:n_re*s]

               if (t<57)||(t>70)
                    fₐ(Q̂) = pQQmc_1d(Q[1],Q̂,Q̄[1],σ_q[1],Q̄_std[1])
                    fᵢ(Q̂) = pQQmc_1d(Q[2],Q̂,Q̄[2],σ_q[2],Q̄_std[2])
                    fₒ(Q̂) = pQQmc_1d(Q[3],Q̂,Q̄[3],σ_q[3],Q̄_std[3])
                    n_sd = 5.
                    #if any((Q̄ - n_sd .* Q̄_std) .< 0.)
                    #     while (any((Q̄ - n_sd .* Q̄_std) .< 0.))||(n_sd==0.)
                    #          n_sd = n_sd - 1.
                    #     end
                    #end

                    #if n_sd==0.
                    #     n_sd = 5.
                    #end
                    num_intₐ, errₐ = quadgk(fₐ, maximum([Q̄[1] - n_sd*Q̄_std[1],eps()]), Q̄[1] + n_sd*Q̄_std[1])
                    num_intᵢ, errᵢ = quadgk(fᵢ, maximum([Q̄[2] - n_sd*Q̄_std[2],eps()]), Q̄[2] + n_sd*Q̄_std[2])
                    num_intₒ, errₒ = quadgk(fₒ, maximum([Q̄[3] - n_sd*Q̄_std[3],eps()]), Q̄[3] + n_sd*Q̄_std[3])

                    num_int_total = log(num_intₐ) + log(num_intᵢ) + log(num_intₒ)
                    prob_g0_Qmc = log( 1 - normcdf( -Q̄[1] / Q̄_std[1] )) +
                         log( 1 - normcdf( -Q̄[2] / Q̄_std[3] )) +
                         log( 1 - normcdf( -Q̄[2] / Q̄_std[3] ))

                    loglik_Q_term = num_int_total - prob_g0_Qmc

                    η[s,t] = macro_ll_cons[s] - 0.5*((ε_m')*(cov_m\ε_m)) + 
                         yields_ll_cons - 0.5*((ε_y') * (σ_y² \ ε_y)) +
                         ν_ll_cons - 0.5*((ε_ν') * (Σ_ν \ ε_ν )) + loglik_Q_term


               else

                    η[s,t] = macro_ll_cons[s] - 0.5*((ε_m') * (cov_m \ ε_m)) + 
                         yields_ll_cons - 0.5*((ε_y') * (σ_y² \ ε_y) ) +
                         ν_ll_cons - 0.5*((ε_ν') * (Σ_ν \ ε_ν )) 

               end
          end

          #Update filtered probabilities
          tmp_ll = exp.(η[:,t]) .* ξ̂_ttm1[:, t]
          lik_mat[t] = ones_S ⋅ tmp_ll
          ξ̂_tt[:,t+1] = tmp_ll ./ lik_mat[t]
          ξ̂_ttm1[:,t+1] = F * ξ̂_tt[:, t+1]

     end


     #set output 
     fp = ξ̂_ttm1[:,2:end]'
     neg_log_lik = -sum(log.(lik_mat))

     return neg_log_lik, fp, ξ̂_tt, ξ̂_ttm1
          
end

function construct_cov_re(rf_struct)
     @unpack σ_q, σ_z, σ_w, σ_ν = rf_struct
     N = 6 
     
     Σ_cap = @MMatrix zeros(N,N)
     Σ_cap[1,1] = σ_q[1]^2
     Σ_cap[2,2] = σ_q[2]^2
     Σ_cap[3,3] = σ_q[3]^2
     Σ_cap[4,4] = σ_z[1]^2 + σ_w[1]^2 + σ_ν[1]^2
     Σ_cap[5,5] = σ_z[2]^2 + σ_w[2]^2 + σ_ν[2]^2
     Σ_cap[6,6] = σ_z[3]^2 + σ_w[3]^2 + σ_ν[3]^2
     Σ_cap[5,4] = σ_z[1]*σ_z[2]
     Σ_cap[4,5] = σ_z[1]*σ_z[2]
     Σ_cap[6,4] = σ_z[1]*σ_z[3]
     Σ_cap[4,6] = σ_z[1]*σ_z[3]
     Σ_cap[6,5] = σ_z[2]*σ_z[3]
     Σ_cap[5,6] = σ_z[2]*σ_z[3]

     return Σ_cap
end

function pQQmc_1d(Q, Q̂, Q̄, σ_re, σ_mc)

     out_pQQ = exp.(-0.5 .* ( ( ( (log(Q)-log(Q̂)) ./ σ_re).^2) + 
          ( ( (Q̂-Q̄) ./ σ_mc).^2) ) ) ./ (σ_re*σ_mc*2*pi)

     return out_pQQ
end

function create_mvn_dist(μ,Ω)
     d = MvNormal(μ,Ω)
     return d
 end
 
 function create_n_dist(μₙ,σ)
     Ωₙ = diagm(vec(σ.^2))
     d = MvNormal(μₙ,Ωₙ)
     return d
 end
 
 function create_exp_dist(λ)
     N_λ = length(λ)
     μ = 1 ./ λ
 
     d_vec = Vector{Exponential{Float64}}(undef,N_λ)
     for i in 1:N_λ
         d_vec[i] = Exponential(μ[i])
     end
     return d_vec
 end
 
 function draw_mvn(N_draw,d,finv_array,mvn_cap_ndx,mvn_tsm_ndx)
     y = rand(d,N_draw)
     finv = finv_array[mvn_cap_ndx]
     N = length(finv)
     y = y[mvn_tsm_ndx,:]
 
     x_mat = Matrix{Float64}(undef,N,N_draw)
     [x_mat[i,:] = map(finv[i],y[i,:]) for i=1:N]
 
     return x_mat
 end
 
 function logpdf_mvn(θ,d,f_array,mvn_cap_ndx,mvn_tsm_ndx)
     N = length(mvn_cap_ndx)
     f_mvn = f_array[mvn_cap_ndx]
     f_mvn[mvn_tsm_ndx] = f_mvn 
     θ_mvn = θ[mvn_cap_ndx]
     θ_mvn[mvn_tsm_ndx] = θ_mvn
 
     y = Vector{Float64}(undef,N)
     [y[i] = map(f_mvn[i],θ_mvn[i]) for i=1:N]
     log_pdf = logpdf(d,y)
 
     return log_pdf
 end
 
 function pdf_mvn(θ,d,f_array,mvn_cap_ndx,mvn_tsm_ndx)
     N = length(mvn_cap_ndx)
     f_mvn = f_array[mvn_cap_ndx]
     f_mvn[mvn_tsm_ndx] = f_mvn 
     θ_mvn = θ[mvn_cap_ndx]
     θ_mvn[mvn_tsm_ndx] = θ_mvn
 
     y = Vector{Float64}(undef,N)
     [y[i] = map(f_mvn[i],θ_mvn[i]) for i=1:N]
     out_pdf = pdf(d,y)
 
     return out_pdf
 end
 
 #Draw Normal variables
 function draw_n(N_draw,d,finv_array,n_ndx)
     y = rand(d,N_draw)
     finv = finv_array[n_ndx]
     N = length(finv)
 
 
     x_mat = Matrix{Float64}(undef,N,N_draw)
     [x_mat[i,:] = map(finv[i],y[i,:]) for i=1:N]
 
     return x_mat
 end
 
 #Evaluate log-pdf of Normally distributed variables
 function logpdf_n(θ,d,f_array,n_ndx)
     N = length(n_ndx)
     f_n = f_array[n_ndx]
     θ_n = θ[n_ndx]
 
     y = Vector{Float64}(undef,N)
     [y[i] = map(f_n[i],θ_n[i]) for i=1:N]
     log_pdf = logpdf(d,y)
 
     return log_pdf
 end
 
 #Draw Exponential variables
 function draw_exp(N_draw,d_vec,finv_array,exp_ndx)
 
     #Define paramters and preallocte
     N_λ = length(d_vec)
     x_mat = Matrix{Float64}(undef,N_λ,N_draw)
     y_mat = Matrix{Float64}(undef,N_λ,N_draw)
     finv_exp = finv_array[exp_ndx]
 
     #Loop over individual distribution
     for i in 1:N_λ
 
         #draw
         d_i = d_vec[i]
         y_i = rand(d_i,N_draw)
 
         #Store
         y_mat[i,:] = y_i
     end
 
     #Transform to θ space
     [x_mat[i,:] = map(finv_exp[i],y_mat[i,:]) for i=1:N_λ]
 
     return x_mat
 end
 
 #Evaluate log-pdf of Exponentially distributed variables
 function logpdf_exp(θ,d_vec,f_array,exp_ndx)
     N_λ = length(d_vec)
     f_exp = f_array[exp_ndx]
     θ_exp = θ[exp_ndx]
 
     y = Vector{Float64}(undef,N_λ)
     #Map exponentiall distirbuted rv into prior space
     [y[i] = map(f_exp[i],θ_exp[i]) for i=1:N_λ] 
     
     
     log_pdf_vec = Vector{Float64}(undef,N_λ) #preallocate
     #loop
     [log_pdf_vec[i] = logpdf(d_vec[i],y[i]) for i=1:N_λ]
 
     log_pdf = sum(log_pdf_vec) #sum up individual log-likliehoods
 
     return log_pdf
 end
 
 #Define parameter structure
 @with_kw struct PriorStruct{F}
     μ::Vector{Float64}
     Ω::Matrix{Float64}
     μₙ::Vector{Float64}
     σ::Matrix{Float64}
     λ::Matrix{Float64}
     f_array::Vector{Function}
     finv_array::Vector{Function}
     mvn_cap_ndx::Vector{Int64}
     mvn_tsm_ndx::Vector{Int64}
     exp_ndx::Vector{Int64}
     n_ndx::Vector{Int64}
     d_mvn::FullNormal
     d_exp::Vector{Exponential{Float64}}
     d_n::FullNormal
     N_θ::Int64
     draw_prior::Function
     eval_logprior::F
 
 end
 
 #Generate distribution function
 function get_prior_distributions(μ,μₙ,Ω,σ,λ)
     d_mvn = create_mvn_dist(μ,Ω)
     d_n = create_n_dist(μₙ,σ)
     d_exp = create_exp_dist(λ)
 
     return (d_mvn, d_n, d_exp)
 end
 
 #Combine draws into final θ
 function combine_draws_to_θ(x_mvn,x_n,x_exp,mvn_cap_ndx,n_ndx,exp_ndx)
     N_θ = length(mvn_cap_ndx) + length(exp_ndx) + length(n_ndx)
     N_draws = size(x_mvn)[2]
     θ_mat = Matrix{Float64}(undef,N_θ,N_draws)
     θ_mat[mvn_cap_ndx,:] = x_mvn
     θ_mat[n_ndx,:] = x_n
     θ_mat[exp_ndx,:] = x_exp
 
     return θ_mat
 
 end
 
 function generate_draw_prior_fn(d_mvn,d_n,d_exp,mvn_cap_ndx,mvn_tsm_ndx,n_ndx,exp_ndx,finv_array)
     
     gen_draw_prior = function(d_mvn,d_n,d_exp,n_ndx,exp_ndx,mvn_cap_ndx,mvn_tsm_ndx,finv_array)
 
         function draw_prior(N_draw)
 
             x_mvn = draw_mvn(N_draw,d_mvn,finv_array,mvn_cap_ndx,mvn_tsm_ndx)
             x_n = draw_n(N_draw,d_n,finv_array,n_ndx)
             x_exp = draw_exp(N_draw,d_exp,finv_array,exp_ndx)
 
             θ_mat = combine_draws_to_θ(x_mvn,x_n,x_exp,mvn_cap_ndx,n_ndx,exp_ndx)
             return θ_mat
 
         end
 
     end
 
     draw_prior_fn = gen_draw_prior(d_mvn,d_n,d_exp,n_ndx,exp_ndx,mvn_cap_ndx,mvn_tsm_ndx,
                         finv_array)
 
     return draw_prior_fn
 end
 
 #Generate eval_logprior
 function generate_eval_logprior(d_mvn,d_n,d_exp,n_ndx,exp_ndx,mvn_cap_ndx,mvn_tsm_ndx,
                                 f_array,absdetjac_fn)
 
     gen_eval_logprior = function(d_mvn,d_n,d_exp,n_ndx,exp_ndx,mvn_cap_ndx,mvn_tsm_ndx,
                                  f_array,absdetjac_fn)
 
         function eval_logprior(θ)
 
             logp_mvn = logpdf_mvn(θ,d_mvn,f_array,mvn_cap_ndx,mvn_tsm_ndx)
             logp_n = logpdf_n(θ,d_n,f_array,n_ndx)
             logp_exp = logpdf_exp(θ,d_exp,f_array,exp_ndx)
 
             logprior = logp_mvn + logp_n + logp_exp + log(absdetjac_fn(θ))
 
             return logprior
 
         end
 
     end
 
     eval_logprior = gen_eval_logprior(d_mvn,d_n,d_exp,n_ndx,exp_ndx,mvn_cap_ndx,
                                       mvn_tsm_ndx,f_array,absdetjac_fn)
 
     return eval_logprior

 end
 
 
 #Write wrapper code to define prior_struct
 function construct_prior_struct(mvn_cap_ndx,mvn_tsm_ndx,n_ndx,exp_ndx,Ω_fp,μ_fp,λ_fp,μ_n_fp,σ_fp)
     #Current;y, f_array and f_inv are hardcoded to match MATLAB
     #TODO: Change this to general code
 
     #Define N_params
     N_θ = length(mvn_cap_ndx) + length(exp_ndx) + length(n_ndx)
 
     #Load from fp
     Ω = readdlm(Ω_fp, ',', Float64)
     μ = vec(readdlm(μ_fp, ',', Float64))
     λ = readdlm(λ_fp,',',Float64)
     μₙ = vec(readdlm(μ_n_fp,',',Float64))
     σ = readdlm(σ_fp,',',Float64)
 
     #Get distributions
     d_mvn, d_n, d_exp = get_prior_distributions(μ,μₙ,Ω,σ,λ)
 
     #Fill out function of transformations from θ -> f(θ)
     f(x) = x
     f_array = Array{Function}(undef, N_θ)
     finv_array = Array{Function}(undef, N_θ)
 
     f_array[1] = x -> log( (x  + 78.294795802327087)./400 + sqrt(eps()) )
     finv_array[1] = x -> 400 .* (exp(x) -78.294795802327087./400 - sqrt(eps()))
 
     f_array[2:3] .= x -> x ./ 400
     finv_array[2:3] .= x -> x .* 400
 
     f_array[4:8] .= x -> x ./ 400 ./ 400
     finv_array[4:8] .= x -> 400 .* 400 .* x
 
     f_array[9:11] .= f
     finv_array[9:11] .= f
 
     f_array[12] = x -> log(x + sqrt(eps()) - 1.000121712793949e-07)
     finv_array[12] = x -> exp(x) + 1.000121712793949e-07 - sqrt(eps())
 
     f_array[13:15] .= f
     finv_array[13:15] .= f
 
     f_array[16] = x -> 0.989999992343042 - x
     finv_array[16] = x -> 0.989999992343042 - x
 
     f_array[17:19] .= f
     finv_array[17:19] .= f
 
     f_array[20] = x -> log(x + sqrt(eps()) - 0.029636701253699)
     finv_array[20] = x -> exp(x) + 0.029636701253699- sqrt(eps())
 
     f_array[21:51] .= f
     finv_array[21:51] .= f
 
     f_array[52] = x -> log(0.994998852596728 + sqrt(eps()) - x)
     finv_array[52] = x -> 0.994998852596728 + sqrt(eps()) - exp(x)
 
     f_array[53] = x -> log(0.994999942585831 + sqrt(eps()) - x)
     finv_array[53] = x -> 0.994999942585831 + sqrt(eps()) - exp(x)
 
     f_array[54:55] .= f
     finv_array[54:55] .= f
 
     draw_prior = 
         generate_draw_prior_fn(d_mvn,d_n,d_exp,mvn_cap_ndx,mvn_tsm_ndx,n_ndx,exp_ndx,finv_array)
 
     #Create Radon-Nikodyn term
     
     
     function  gen_h(f_arr)
     
          function h_fn(x)
               θ_out = zero(x)
               for i in 1:size(x)[1]
                    θ_out[i] = f_arr[i](x[i])
               end
          
               return θ_out
          end
     
          return h_fn

     end
     
     h = gen_h(finv_array)
     g = gen_h(f_array)
     
     jacob_fn = x::Matrix{Float64} -> ForwardDiff.jacobian(h,g(x))::Matrix{Float64}
     absdetjac_fn = x::Matrix{Float64} -> abs(det(jacob_fn(x)))::Float64
 
     eval_logprior = generate_eval_logprior(d_mvn,d_n,d_exp,n_ndx,exp_ndx,mvn_cap_ndx,
                                            mvn_tsm_ndx,f_array,absdetjac_fn)
 
     #Define structure
     prior_struct = PriorStruct(μ = μ, Ω = Ω, μₙ = μₙ, σ = σ, λ = λ, f_array = f_array,
         finv_array = finv_array, mvn_cap_ndx = mvn_cap_ndx, n_ndx = n_ndx, exp_ndx = exp_ndx,
         d_mvn = d_mvn, d_n = d_n, d_exp = d_exp, N_θ = N_θ, draw_prior = draw_prior,
         eval_logprior = eval_logprior, mvn_tsm_ndx = mvn_tsm_ndx)
 
     return prior_struct
 
 end

#Contruct prior using defaults from MATLAB
#TODO: Eventually I may want to generalize this funciton
function construct_prior_default()
     #set parameters
     mvn_cap_ndx = [1:15;17:26;52:55]
     exp_ndx = [16;39:50]
     n_ndx = [27:38;51]
     mvn_tsm_ndx = [1:8;28:29;9:11;15:16;12:14;17:18;23:27;19:22]
     Ω_fp = "prior/cov_mvn.csv"
     μ_fp = "prior/mu_mvn.csv"
     μ_n_fp = "prior/n_mu_mat.csv"
     σ_fp = "prior/n_se_mat.csv"
     λ_fp = "prior/lambda_mat.csv"

     prior_struct = construct_prior_struct(mvn_cap_ndx,mvn_tsm_ndx,n_ndx,exp_ndx,Ω_fp,μ_fp,
                                          λ_fp,μ_n_fp,σ_fp)

     return prior_struct

end

function construct_Q_mats(Q_A_mc_mat,Q_I_mc_mat,Q_O_mc_mat,Q_A_std_mc_mat,Q_I_std_mc_mat,
     Q_O_std_mc_mat)

     T,S = size(Q_A_mc_mat)
     n_re = 3

     Q_mc_mat = Matrix{Float64}(undef,T,n_re*S)
     σ_mc_mat = Matrix{Float64}(undef,T,n_re*S)

     for s = 1:S
          Q_mc_mat[:,1+(s-1)*n_re:n_re*s] = [Q_A_mc_mat[:,s] Q_I_mc_mat[:,s] Q_O_mc_mat[:,s]]
          σ_mc_mat[:,1+(s-1)*n_re:n_re*s] = 
               [Q_A_std_mc_mat[:,s] Q_I_std_mc_mat[:,s] Q_O_std_mc_mat[:,s]]
     end

     return Q_mc_mat, σ_mc_mat
end


function get_mc_posterior_quadrature(θ,mcs,ps)

     @unpack Q_A_mat,Q_I_mat,Q_O_mat,Q_A_std_mat,Q_I_std_mat,Q_O_std_mat,T,S,
         cap_data, Macro_data, ν_data, Y_data, S = mcs
 
     ndx_vec = [26,41,42,45,46,49,50]

     
     Q_mc_mat, σ_mc_mat = construct_Q_mats(Q_A_mat,Q_I_mat,Q_O_mat,Q_A_std_mat,Q_I_std_mat,
                                           Q_O_std_mat)

 
     #Set inital values
     θ[ndx_vec[1]] = 0.005
     θ[ndx_vec[2:2:6]] .= 0.005
     for (cnt,j) in enumerate(3:2:7)
         tmp_err = repeat(log.(400. ./ cap_data[2:end,cnt]),1,S) - 
               log.(Q_mc_mat[2:end,1+(cnt-1)*4:cnt*4])
         θ[ndx_vec[j]] = std(tmp_err[:])
     end
 
 
     obj_fn_ll(in_vec) = mc_loglik_numint_1d(reshape([θ[1:25];in_vec[1];θ[27:40];
          in_vec[2:3];θ[43:44];in_vec[4:5];θ[47:48];in_vec[6:7];θ[51:55]],length(θ),1),
          Y_data,Macro_data, cap_data, ν_data,ones(55,1), 1, Q_mc_mat, σ_mc_mat)
          
     obj_fn_lp = let eval_logprior=ps.eval_logprior
          (x)-> eval_logprior(reshape([θ[1:25];x[1];θ[27:40];x[2:3];θ[43:44];x[4:5];
                              θ[47:48];x[6:7];θ[51:55]],length(θ),1))
     end
     
     obj_fn_lpost(in_vec) = obj_fn_ll(in_vec)[1] - obj_fn_lp(in_vec)

 
     init_vec = [θ[26];θ[41:42];θ[45:46];θ[49:50]]
     neg_log_posterior = obj_fn_lpost(init_vec)
 
 
     ccnt = 1
     trig = 0
     if (isnan(neg_log_posterior)|isinf(neg_log_posterior))
          while (trig==0)&(ccnt<=20)
               init_vec = init_vec .* 2.0
               neg_log_posterior = obj_fn_lpost(init_vec)
               if !(isnan(neg_log_posterior)|isinf(neg_log_posterior))
                    trig = 1
               else
                    ccnt = ccnt + 1
               end
          end
     end

     for j=1:size(ndx_vec)[1]

          obj_fn_ll_2(meas) = mc_loglik_numint_1d(reshape([θ[1:ndx_vec[j]-1];meas[1];
               θ[ndx_vec[j]+1:end]],length(θ),1), Y_data,Macro_data, cap_data, ν_data,
               ones(55,1), 1, Q_mc_mat, σ_mc_mat)
          obj_fn_lp_2 = let eval_logprior = ps.eval_logprior
               (x) -> eval_logprior(reshape([θ[1:ndx_vec[j]-1];x[1];θ[ndx_vec[j]+1:end]],
                                             length(θ),1))
          end
          obj_fn_lpost_2(meas) = obj_fn_ll_2(meas)[1] - obj_fn_lp_2(meas)

          obj_fn_uc(in_vec) = obj_fn_lpost_2(exp(in_vec[1]))

          res = optimize(obj_fn_uc,[log(init_vec[j])])
          
          θ̂ = exp(Optim.minimizer(res)[1])
          
          #init_vec[j] = θ̂
          #neg_log_posterior, fp, sp, log_lik, log_prior =  obj_fn(init_vec)
          
          θ[ndx_vec[j]] = θ̂
     end
     
     θ_new = θ
 
     init_vec = θ[ndx_vec]
     ~, ~, ~, ll = obj_fn_ll(init_vec)
     lprior = obj_fn_lp(init_vec)
     lpost = lprior + ll
     
     return lpost, ll, lprior, θ_new
 end



function  cond_moms_ms_var(α,Φ,λ,Π)

     K = size(Π,1)
     n = maximum(size(α[1]))

     #Get backwards transition probs
     M = Array{Float64}(undef, K, K)

     q = get_ergodic_markov_dist(Π)
     for i in 1:K
          for j in 1:K
               M[j,i] = Π[i,j] * (q[i]/q[j])
          end
     end
     M = M ./ sum(M,dims=2)
     
     #Precomputation
     kronMI = kron(M,Matrix{Float64}(I,n,n))


     #Construct diagonal matrices
     diagAlpha = spzeros(n*K,K)
     for i in 1:K
          diagAlpha[1+(i-1)*n:n*i,i] = α[i]
     end
    
     diagPhi = spzeros(n*K,n*K)
     for i in 1:K
          diagPhi[1+(i-1)*n:n*i,1+(i-1)*n:n*i] = Φ[i]
     end
    
     diagLambda = spzeros(n*K,n*K)
     for i in 1:K
          diagLambda[1+(i-1)*n:n*i,1+(i-1)*n:n*i] = λ[i]
     end
    
     # Get J's
     J_cell = Array{Array{Float64}}(undef,K) #cell(K,1)
     for i in 1:K
          J_cell[i] = zeros(n,n*K)
          J_cell[i][:,1+(i-1)*n:i*n] = I
     end
    
     #Get J
     J = spzeros(n,n*K)
     for i in 1:K
          J += J_cell[i] 
     end
    
     #Get es
     e_cell = Array{Array{Float64}}(undef,K) #cell(K,1)
     for i in 1:K
          e_cell[i] = zeros(K,1)
          e_cell[i][i] = 1.0
     end
    
     #Get A0
     tmp_diags = kronMI * 
                    (diagAlpha*diagAlpha' + diagLambda*diagLambda')*J'
     A0 = zeros(n*K)
     for i in 1:K
          A0[1+(i-1)*n:i*n,1+(i-1)*n:i*n] = tmp_diags[1+(i-1)*n:i*n,:]
     end
    
     #Get H
     H = spzeros(n*n*K*K,n*n*K)
     for i in 1:K
          H += kron(J_cell[i]',(J_cell[i]'*J_cell[i]))
     end
    
     #Get A
     A0Jp = A0*J'
     vecA = H* ( (I - (kron(J*diagPhi,kronMI*diagPhi)*H))\A0Jp[:] )
          
     A = reshape(vecA,n*K,n*K)
    
     #Get B01
     B01 = zeros(n*K,K)
     tmp_diags = kronMI*diagAlpha*ones(K,1)
     for ii=1:K
          B01[1+(ii-1)*n:ii*n,ii] = tmp_diags[1+(ii-1)*n:ii*n]
     end
    
     #Get B0
     B0 = zeros(n*K,K)
     tmp_diags = (I-kronMI*diagPhi)\B01*ones(K,1)
     for i in 1:K
          B0[1+(i-1)*n:i*n,i] = tmp_diags[1+(i-1)*n:i*n]
     end
    
     #Get B
     tmp = kronMI * diagPhi * B0 * diagAlpha' * J'
     vecB = (H / (I-(kron(J*diagPhi,kronMI*diagPhi)*H)) )*tmp[:]
     B = reshape(vecB,n*K,n*K)
    
    
     #Get Gamma
     Gamma_cell = Array{Array{Float64}}(undef, K)
     for i in 1:K
          Gamma_cell[i] = Φ[i]*J_cell[i]*B0*e_cell[i]*α[i]' + 
                          Φ[i]*J_cell[i]*B*J_cell[i]'*Φ[i]'
     end
    
     #First Moment
     m1_cell = Array{Array{Float64}}(undef, K)
     for i in 1:K
          m1_cell[i] = α[i] + Φ[i]*J_cell[i] * 
               ((I-kronMI*diagPhi)\(kronMI *
                diagAlpha*ones(K,1)))
     end
    
     #Second Moment
     m2_cell = Array{Array{Float64}}(undef, K)
     for i in 1:K
          m2_cell[i] = α[i]*α[i]' + λ[i]*λ[i]' + Φ[i]*J_cell[i]*A*J_cell[i]'*Φ[ii]' +
               Gamma_cell[i] + Gamma_cell[i]'
     end

     return m1_cell, m2_cell

end

function get_ergodic_markov_dist(Π)

     mc = MarkovChain(Π)
     q = stationary_distributions(mc)[1]

     return q
end

function get_lim_η(Φ_Q,μ_Q,δ,c,uM1,uVar,cM1_cell,cM2_cell,Π)

     ## Precompute stuff
     q = get_ergodic_markov_dist(Π)
     PiU = repeat(q',S,1) 
     S = size(Pi,1)
     n = size(mu_Q[1],1)
     mm = n*S

     g = (x) -> (x-1)%(n)+1
     f = (x) -> floor((x-1)/n)+1

     #Construct M
     M = Array{Float64}(undef, S, mm) #NaN(S,mm)
     for s in 1:S
          for m in 1:mm
               M[s,m] = ((Π[s,f(m)] - PiU[s,f(m)]).*μ_Q[f(m)])[g(m)]
          end
     end

     #Construct delta
     Δ = Array{Float64}(undef, mm) #NaN(mm,1)
     for m in 1:mm
          Δ[m] = δ[g(m)] 
     end

     #Contruct K
     K = Array{Float64}(undef, mm, mm)#NaN(mm)
     for m in 1:mm
          for mp in 1:mm
               K[m,mp] = Π[f(m),f(mp)] .* Φ_Q[f(mp)][g(mp),g(m)] 
          end
     end

     #Construct X
     X = Array{Float64}(undef, mm) 
     for m in 1:mm
          X[m] = cM1_cell[f(m)][g(m)]*q[f(m)]
     end

     #Construct Omega
     Ω = Array{Float64}(undef, mm, mm)
     for m in 1:mm
          for mp in 1:mm
               Ω[m,mp] = q[f(m)]*q[f(mp)]*cM2_cell[f(mp)][g(m),g(mp)]
          end
     end

     ## Compute limit of first moment
     lim_m1 = δ'*uM1 + c

     ## Compute limit of variance
     lim_var = δ'*uVar*δ

     ## Compute limit of sum of covariance terms
     lim_A = Δ' * 
          kron( (((I-Pi-PiU)\M)/(I-K) - (PiU*M/(I-K))/(I-K))',ones(size(uM1,1),1)') *
          Diagonal(X) * Δ 

     lim_cov = lim_A + Δ' * Ω * ((I-K) \ Δ) - (Δ' * X) * (X' * ((I-K) \ Δ))


     ## Put it all together
     out = lim_m1 + lim_cov - 0.5*lim_var

     return out, lim_m1, lim_var, lim_cov
end


export simulate_markov_switch_init_cond_shock, simulate_ms_var_1_cond_shocks, 
     simulate_msvar_cond_regime_path_shock, construct_gamma_macro_array, construct_m0_macro_array, 
     construct_b1_macro_array, construct_bm1_macro_array, construct_b0_macro_array, 
     construct_structural_matrices_macro, transform_struct_to_rf, get_forward_solution_msre, get_rf_Φ,
     compute_drift_sigma_fmmsre, compute_μ_Σ_fmmsre, construct_structural_parameters, params_to_rf, 
     augment_macro_fsmsre_nu, simulate_nu_cond_x_i_shock_rn, compute_mc_real_estate_Q_cond_x_i_nofull,
     compute_mc_term_structure_cond_x, compute_std_over_subsamples, compute_mean_over_subsamples,
     simulate_model_prices_cond_shock_acc_ts,construct_prior_default, construct_Q_mats,
     mc_loglik_numint_1d, construct_mc_struct, get_mc_posterior_quadrature, process_data,
     is_feasible, make_model_predictions, compute_model_neg_log_lik, construct_prior_struct,
     get_prior_distributions, generate_draw_prior_fn,  generate_eval_logprior, fmmsre


end