module SimMdlPrices

using Parameters
using LinearAlgebra
using QuantEcon
using SparseArrays
using StaticArrays
using Statistics
using Infiltrator
using StatsFuns
using BenchmarkTools
using InteractiveUtils

function simulate_ms_var_1_cond_shocks!(x_mat,s_mat,x0,s0,μ,Φ,Σ,Π,u,ϵ,T_start, g_start)

     simulate_markov_switch_init_cond_shock!(s_mat, s0, Π, u, T_start, g_start)
     simulate_msvar_cond_regime_path_shock!(x_mat, x0, s_mat, μ, Φ, Σ, ϵ,T_start, g_start)

end

function simulate_markov_switch_init_cond_shock!(s_mat,s0, Π, u,T_start,g_start)
     
     
     T,G = size(u)
     S = size(Π)[1]

     #s_mat = Array{Int64}(undef,T+1,G)
     if T_start == 1
          s_mat[1,:] .= s0
     end

     #Propogate
     Π₀ = cumsum(Π, dims = 2)
     cnt_g = 1
     for g ∈ g_start:G+g_start-1
          cnt_t  = 1
          for t ∈ T_start:T_start+T-1
               for s ∈ 1:S
                    if u[cnt_t, cnt_g] <= Π₀[s_mat[t, g], s]
                         s_mat[t+1, g] = s
                         break
                    end   
               end
               cnt_t += 1
          end
          cnt_g += 1
     end

     return s_mat
     
end

function simulate_msvar_cond_regime_path_shock!(x_mat,x0,s_mat,μ,Φ,Σ,ϵ,T_start, g_start)

     #Unload and Preallocate
     N, N2 = size(x0)
     T = size(ϵ)[2] 
     G = size(ϵ)[3]
     @assert N == size(ϵ)[1]
     
     if N2==1
          x0 = repeat(x0, 1, G)
      end

     #Propogate
     cnt_g = 1
     for g ∈ g_start:G+g_start-1
          cnt_t = 1
          if T_start==1
               x_mat[:, 1, g] = @view(x0[:, cnt_g])
          end

          for t ∈ T_start:T_start+T-1
               s = s_mat[t+1, g]
               x_mat[:, t+1, g] = μ[s] .+ Φ[s] *  @view(x_mat[:,t,g])  .+ Σ[s] * @view(ϵ[:, cnt_t, cnt_g])
               cnt_t += 1
          end

          cnt_g += 1

     end
     
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
     mc = MarkovChain(Π)
     q = stationary_distributions(mc)[1]

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

function get_forward_solution_msre(π_m, A1₁, A1₂, F1₁, F1₂, m1₁, m1₂, Σ1_11, Σ1_12, Σ1_21, Σ1_22, maxₖ)

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
     global k_term = 0
     global Ωk₁ = similar(Ωkm1₁)
     global Ωk₂ = similar(Ωkm1₁)
     function f(Ωkm1,maxₖ,A1₁, A1₂,π_m,B₁,B₂,tolK)
               for i ∈ 1:maxₖ
                    #update Φₖ
                    Ωk₁ = (I - A1₁ * (π_m[1,1] .* Ωkm1[1] + π_m[1,2] .* Ωkm1[2] ) ) \ B₁
                    Ωk₂ = (I - A1₂ * (π_m[2,1] .* Ωkm1[1] + π_m[2,2] .* Ωkm1[2] ) ) \ B₂

                    #Check for convergence
                    dist = max( maximum(abs.( Ωk₁ - Ωkm1[1] )), maximum(abs.( Ωk₂ - Ωkm1[2] )))

                    
                    #update values or break
                    if (dist <= tolK) || (i==maxₖ)
                         k_term = i
                         break
                    else
                         Ωkm1[1] = Ωk₁
                         Ωkm1[2] = Ωk₂
                    end

               end
          return Ωk₁, Ωk₂, k_term
     end

     Ωk₁,Ωk₂,k_term = f(Ωkm1,maxₖ,A1₁, A1₂,π_m,B₁,B₂,tolK)
     Φ = [ Ωk₁, Ωk₂ ]

     return Φ, k_term

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
     det_dum::Int64

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

     @unpack Π_0, Π_X, m_g, m_r₁, m_r₂, ϕ, δ, μ_g, μ_pi, ρ₁, ρ₂, β₁, β₂, α₁, α₂, σ_g, σ_pi, σ_r₁, σ_r₂, σ_y, γ_gₐ, 
     γ_piₐ, γ_cₐ, γ_gᵢ, γ_piᵢ, γ_cᵢ, γ_gₒ, γ_piₒ, γ_cₒ, ρₐ, ρᵢ, ρₒ, σ_wₐ, σ_zₐ, σ_νₐ, σ_qₐ, σ_wᵢ, σ_zᵢ, σ_νᵢ, σ_qᵢ, σ_wₒ, σ_zₒ,
     σ_νₒ, σ_qₒ, λ, π_m, π_d = msp

     b0₁,b0₂,bm1₁,bm1₂,b1₁,b1₂,m0₁,m0₂,Γ₁,Γ₂ = 
          construct_structural_matrices_macro(ϕ, δ, ρ₁, ρ₂, β₁, β₂, α₁, α₂, μ_g, μ_pi, m_g, m_r₁, m_r₂, σ_g, σ_pi, σ_r₁, 
                                                                       σ_r₂)
     
     μ, Φ, μ_Q, Φ_Q, Σ, cov, Π, q, det_dum = 
          transform_struct_to_rf(b0₁, b0₂, bm1₁, bm1₂, b1₁, b1₂, m0₁, m0₂, Γ₁, Γ₂, π_m, π_d, Π_0, Π_X)
     
     μ_ν , Φ_ν, μ_Q_ν, Φ_Q_ν, Σ_ν, cov_ν, Δ, Δ_Q, σ_w, σ_z, σ_q, σ_ν = 
          augment_macro_fsmsre_nu(μ, Φ, μ_Q, Φ_Q, Σ, γ_gₐ, γ_gᵢ, γ_gₒ, γ_piₐ, γ_piᵢ, γ_piₒ, γ_cₐ, γ_cᵢ, γ_cₒ, ρₐ, ρᵢ, ρₒ, 
                                                        λ, σ_wₐ, σ_wᵢ, σ_wₒ, σ_zₐ, σ_zᵢ, σ_zₒ,σ_qₐ, σ_qᵢ, σ_qₒ, σ_νₐ, σ_νᵢ, σ_νₒ)
     
     mrfp = ModelReducedFormParams(μ = μ, μ_ν = μ_ν, μ_Q = μ_Q, μ_Q_ν = μ_Q_ν, Φ = Φ, Φ_ν = Φ_ν, 
                                                               Φ_Q = Φ_Q, Φ_Q_ν = Φ_Q_ν, Σ = Σ, Σ_ν = Σ_ν, cov = cov, 
                                                               cov_ν = cov_ν, Π = Π, q = q, det_dum = det_dum, Δ = Δ, Δ_Q = Δ_Q,
                                                               σ_w = σ_w, σ_z = σ_z, σ_q = σ_q, σ_ν = σ_ν)

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
     σ_ν = cholesky(σ2_ν).L

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
     
     Σ_ν = [ SMatrix{6,6,Float64,36}([Σ[1] zeros(N,N); Δ*Σ[1] σ_ν]),
                 SMatrix{6,6,Float64,36}( [Σ[2] zeros(N,N); Δ*Σ[2] σ_ν]),
                 SMatrix{6,6,Float64,36}([Σ[3] zeros(N,N); Δ*Σ[3] σ_ν]),
                 SMatrix{6,6,Float64,36}([Σ[4] zeros(N,N); Δ*Σ[4] σ_ν])]

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

     u = Vector{Array{Float64,2}}(undef, g)
     ϵ = Vector{Array{Float64,3}}(undef, g)
     w = Vector{Array{Float64,3}}(undef, g)
     z = Vector{Array{Float64,2}}(undef, g)

     for gg ∈ 1:g
          u[gg] = u_mat[:,1+(gg-1)*k:gg*k]
          ϵ[gg] = ϵ_mat[:,:,1+(gg-1)*k:gg*k]
          w[gg] = w_mat[:,:,1+(gg-1)*k:gg*k]
          z[gg] = z_mat[:,1+(gg-1)*k:gg*k] 
     end

     return u, ϵ, w, z
end

function simulate_nu_cond_x_i_shock_rn!(ν_mat,x_mat,init_ν,mrfp,w,z,T_start, g_start)

     N = size(x_mat)[1]
     n_ts = size(w)[2]
     G = size(w)[3]
     N_re, N2 = size(init_ν)
     @unpack σ_w, σ_z, Δ_Q = mrfp

     if N2==1
          init_ν = repeat(init_ν,1,G)
     end

     ## Scale systematic shocks by std. dev. loading
     w_mat = repeat(σ_w, 1, n_ts, G) .* w

     tmp_z_mat = Array{Float64}(undef, N_re, n_ts, G)
     for g ∈ 1:G
          tmp_z_mat[:, :, g] = repeat(z[:,g]', N_re)
     end

     z_mat = repeat(σ_z, 1, n_ts, G) .* tmp_z_mat
     wz_mat = w_mat + z_mat

     cnt_g = 1
     for g=g_start:G+g_start-1
          if T_start == 1
               ν_mat[:,1,g] = @view(init_ν[:, cnt_g])
          end
          wz_path = @view(wz_mat[:, :, cnt_g])
          ##Create X_mat = [x_t; nu_{t-1}; 1]'; an (N + N_re + 1) x n_ts Array
          X_mat = Array{Float64}(undef, N+N_re+1, n_ts+1)
          X_mat[1:N, 1:end-1] = @view(x_mat[:, T_start+1:T_start+n_ts, g])
          X_mat[end,:] = ones(1, n_ts+1)
          X_mat[N+1:N+N_re, 1] = @view(ν_mat[:, 1, g])
          
          ##Compute nu_t iteratively 
          cnt = 1
          for i=T_start:n_ts+T_start-1

               X_mat[N+1:N+N_re, cnt+1] = Δ_Q * @view(X_mat[:, cnt]) + @view(wz_path[: ,cnt])
               cnt +=1

          end

          ν_mat[:, T_start+1:T_start+n_ts, g] = @view(X_mat[N+1:N+N_re, 2:end])
          cnt_g += 1

     end

     return ν_mat
end

function compute_mc_real_estate_Q_cond_x_i_nofull!( Q_mc, η_mc, Q_std_mc, η_std_mc,
                                                                                          Q_std_prep, η_std_prep,x_mat, ν_mat,T_bar,G)
     T = size(x_mat)[2] - 1
     n_η = T_bar
     N_re = size(ν_mat)[1]

     # Preallocate
     Δ_ν = [0. 0. -1. 1. 0. 0.;
                 0. 0. -1. 0. 1. 0.;
                 0. 0. -1. 0. 0. 1.]

     m1_η_sim = Array{Float64}(undef, N_re, T_bar, G)
     x_path_mat = [x_mat; ν_mat]

     for g=1:G

          x_path = @view(x_path_mat[:,1:T_bar,g])
          m1_η_sim[:, :, g] = cumsum(Δ_ν * x_path, dims = 2)

     end
     η_sim = exp.(m1_η_sim)
     η_mc[:,1:T] .= dropdims(mean(η_sim[:,:,1:G], dims = 3), dims = 3)
     Q_mc .= sum(η_mc[:,1:T], dims = 2)
     #Get std mat
     N_std = Int64(floor(G / 30))
     for i=1:30

          η_std_prep[:, 1:T_bar, i] =  dropdims(mean(@view(η_sim[:, :, 1+(i-1)*N_std:i*N_std]), dims = 3), dims = 3)
          Q_std_prep[:, i] = sum(@view(η_std_prep[:, 1:n_η, i]), dims = 2)

     end

     η_std_mc .= dropdims(std(η_std_prep, dims = 3), dims = 3) ./ sqrt(30)
     Q_std_mc .= std(Q_std_prep, dims = 2) ./ sqrt(30)

end

function compute_mc_term_structure_cond_x(x_mat, maturity_mat)
     G = size(x_mat)[3]
     N_terms = size(maturity_mat)[2]

     δ = [0. 0. -1.]
     ts_sim = Array{Float64}(undef, N_terms, G)
     m1_ts_sim = Array{Float64}(undef, N_terms,G)
     m2_ts_sim = Array{Float64}(undef, N_terms,G)

     for g=1:G
          x_path = @view(x_mat[:, :, g])
          for n=1:N_terms
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
     for i=1:30
          ts_std_prep[:, i] =  -log.(mean(@view(ts_sim[:, 1+(i-1)*N_std:i*N_std]), dims = 2)) ./ maturity_mat'
     end
     ts_std_mc = std(ts_std_prep, dims = 2) ./ sqrt(30)

     return ts_mc, ts_std_mc, m1_ts_mc, m2_ts_mc, ts_std_prep
end

function   get_total_mean_and_std!(Q_mc_tot, η_mc_tot, Q_std_mc_tot, Q_mc_cell, eta_mc_cell,
                                                            Q_std_prep_cell)

     Q_mc_tot .= compute_mean_over_subsamples(Q_mc_cell)
     η_mc_tot .= compute_mean_over_subsamples(eta_mc_cell)

     Q_std_mc_tot .= compute_std_over_subsamples(Q_std_prep_cell)
     #η_std_mc_tot .= compute_std_over_subsamples(η_std_prep_cell)

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
     # Unload inputs
     N_macro = 3
     N_re = 3
     M = size(u)[1]
     G = size(ϵ)[3]
     @assert floor(G / L)== (G/L)
     gg = Int64(G / L)

     @unpack μ_Q, Φ_Q, Σ, Π = mrfp

     Q_mc = Array{Float64}(undef, N_re, 1)
     Q_std_mc = Array{Float64}(undef, N_re, 1)
     #η_std_mc = Array{Float64}(undef, N_re, T_sim)
     η_mc= Array{Float64}(undef, N_re, T_sim)
     x_mat = Array{Float64}(undef, N_macro, T_sim+1, G)
     ν_mat = Array{Float64}(undef, N_re, T_sim + 1, G)
     s_mat = Array{Int64}(undef, T_sim+1, G)

     u_cell, ϵ_cell, w_cell, z_cell  = split_shock_mat(u, ϵ, w, z, L)

     X = Array(Array(x_init[:, t]')')
     ν = Array(Array(ν_init[:, t]')')
     T_bar = [0]     

     outer_while!(Q_mc, η_mc, Q_std_mc, x_mat, ν_mat, s_mat, T_bar, T_sim, n_grps, gg, X, ν, s, μ_Q, 
                                    Φ_Q, Σ, Π, u_cell, ϵ_cell, w_cell, z_cell, mrfp, η_tol, N_re, del, L, tol)

     return Q_mc, η_mc, Q_std_mc, s_mat, x_mat, ν_mat, T_bar

end

function inner_while!(x_mat,s_mat, ν_mat, T_bar, Q_mc_tmp, η_mc_tmp, Q_std_mc_tmp, η_std_mc_tmp,
                                   Q_std_prep, η_std_prep, l, n_grps, T_tmp, μ_Q, Φ_Q, Σ, Π, u_cell, ϵ_cell, w_cell,z_cell, 
                                   mrfp, η_tol,gg)
     
     trig_2 = 0
     cnt_2 = 1
     while (trig_2==0)&&(cnt_2<=n_grps)

          tmp_ndx = (1+(cnt_2-1)*T_tmp):(cnt_2*T_tmp)
          g_start = 1 + (l-1)*gg
          #Simulate model for length T_tmp
          simulate_ms_var_1_cond_shocks!(x_mat, s_mat, x_mat[:, tmp_ndx[1], :],
               s_mat[tmp_ndx[1], :], μ_Q, Φ_Q, Σ, Π, u_cell[l][tmp_ndx, :], ϵ_cell[l][: ,tmp_ndx, :],tmp_ndx[1],g_start)

          #Simulate \nu
          simulate_nu_cond_x_i_shock_rn!(ν_mat, x_mat, ν_mat[:, tmp_ndx[1], :], mrfp, w_cell[l][:, tmp_ndx, :], 
                                                                 z_cell[l][tmp_ndx, :],tmp_ndx[1],g_start)

          #Simulate Q
          compute_mc_real_estate_Q_cond_x_i_nofull!(Q_mc_tmp, η_mc_tmp, Q_std_mc_tmp, η_std_mc_tmp, 
                                                                                          Q_std_prep, η_std_prep , x_mat[:, 1:tmp_ndx[end]+1, :],
                                                                                          ν_mat[:, 1:tmp_ndx[end]+1, :],tmp_ndx[end],l*gg)
          #Check that all η_{T_sim,x_t)< cutoff for all t
          if (~any(η_mc_tmp[:, tmp_ndx[end]] .>= η_tol)) || (cnt_2==n_grps)

               #Set loop trigger and T_bar
               T_bar[1]  = cnt_2 * T_tmp
               trig_2 = 1

          else

               cnt_2 +=  1

          end
     end

     return 

end

function outer_while!(Q_mc, η_mc, Q_std_mc, x_mat, ν_mat, s_mat, T_bar, T_sim, n_grps, gg, X, ν, s, 
                                        μ_Q, Φ_Q, Σ, Π, u_cell, ϵ_cell, w_cell, z_cell, mrfp, η_tol, N_re, del, L, tol)

     T_tmp = Int64(T_sim ./ n_grps)
     G = gg * L
     x_mat[:, 1, :] = repeat(X, 1, 1, G)
     ν_mat[:, 1, :] = repeat(ν, 1, 1, G)
     s_mat[1, :] .= s
     
     trig = 0
     l = 1

     η_std_prep = Array{Float64}(undef, N_re, T_sim, 30)
     Q_std_prep = Array{Float64}(undef, N_re, 30)
     Q_mc_tmp = Array{Float64}(undef, N_re, 1)
     η_mc_tmp = Array{Float64}(undef, N_re, T_sim)
     Q_std_mc_tmp = Array{Float64}(undef, N_re, 1)
     η_std_mc_tmp = Array{Float64}(undef, N_re, T_sim)
     Q_mc_tot = Array{Float64}(undef, N_re,1)
     η_mc_tot = Array{Float64}(undef, N_re, T_sim)
     Q_std_mc_tot = Array{Float64}(undef, N_re, 1)
     #η_std_mc_tot = Array{Float64}(undef, N_re, T_sim)
     Q_mc_cell = Vector{Array{Float64,2}}(undef, L)
     Q_std_mc_cell = Vector{Array{Float64,2}}(undef, L)
     η_mc_cell = Vector{Array{Float64,2}}(undef, L)
     η_std_mc_cell = Vector{Array{Float64,2}}(undef, L)
     Q_std_prep_cell = Vector{Array{Float64,2}}(undef, L)
     η_std_prep_cell = Vector{Array{Float64,3}}(undef, L)

     while ((trig==0) && (l<=L)) #early simulation check

          if l==1

               #Detect early truncation point, T_bar
               inner_while!(x_mat, s_mat, ν_mat, T_bar, Q_mc_tmp, η_mc_tmp, Q_std_mc_tmp, η_std_mc_tmp,
                                        Q_std_prep, η_std_prep, l, n_grps, T_tmp, μ_Q, Φ_Q, Σ, Π, u_cell, ϵ_cell, w_cell, z_cell,
                                         mrfp, η_tol,gg)
               TT_bar = T_bar[1]

          else
               TT_bar = T_bar[1]
               g_start = 1 + (l-1)*gg
               #Run simulation for T_bar periods
               # Simulate Macro model
               simulate_ms_var_1_cond_shocks!(x_mat, s_mat,X, s, μ_Q, Φ_Q, Σ,Π,u_cell[l][1:TT_bar, :],
                                                                                                    ϵ_cell[l][:, 1:TT_bar, :],1,g_start)

               #Simulate \nu
               simulate_nu_cond_x_i_shock_rn!(ν_mat, x_mat, ν, mrfp ,w_cell[l][:, 1:TT_bar, :], z_cell[l][1:TT_bar, :],
                                                                      1, g_start)

               #Approximate real estate Q
               compute_mc_real_estate_Q_cond_x_i_nofull!(Q_mc_tmp, η_mc_tmp, Q_std_mc_tmp, η_std_mc_tmp, 
                                                                                         Q_std_prep, η_std_prep, x_mat[:,1:TT_bar+1,:], 
                                                                                         ν_mat[:,1:TT_bar+1,:],TT_bar, l * gg)
          end
          
          Q_mc_cell[l] = copy(Q_mc_tmp)
          η_mc_cell[l] = copy(η_mc_tmp)
          Q_std_mc_cell[l] = copy(Q_std_mc_tmp)
          η_std_mc_cell[l] = copy(η_std_mc_tmp)
          Q_std_prep_cell[l] = copy(Q_std_prep)
          η_std_prep_cell[l] = copy(η_std_prep)
          
          #Compute total mean and variance
          get_total_mean_and_std!(Q_mc_tot, η_mc_tot, Q_std_mc_tot, Q_mc_cell[1:l], η_mc_cell[1:l], 
                                                    Q_std_prep_cell[1:l])
    
          #Check condition
          if (!any((2 .* (1. .- normcdf.((1.2 .* tol .* Q_mc_tot) ./ Q_std_mc_tot)) .> del)) || (l==L) )

               trig = 1;
               println("l: $l")
               println("T_bar: $T_bar")

               Q_mc .= Q_mc_tot
               η_mc .= η_mc_tot
               Q_std_mc .= Q_std_mc_tot
               #η_std_mc[:, 1:T_bar] = η_std_mc_tot

          else

              l = l+1;

          end
     end

     return

end

export simulate_markov_switch_init_cond_shock, simulate_ms_var_1_cond_shocks, 
     simulate_msvar_cond_regime_path_shock, construct_gamma_macro_array, construct_m0_macro_array, 
     construct_b1_macro_array, construct_bm1_macro_array, construct_b0_macro_array, 
     construct_structural_matrices_macro, transform_struct_to_rf, get_forward_solution_msre, get_rf_Φ,
     compute_drift_sigma_fmmsre, compute_μ_Σ_fmmsre, construct_structural_parameters, params_to_rf, 
     augment_macro_fsmsre_nu, simulate_nu_cond_x_i_shock_rn, compute_mc_real_estate_Q_cond_x_i_nofull,
     compute_mc_term_structure_cond_x, compute_std_over_subsamples, compute_mean_over_subsamples,
     simulate_model_prices_cond_shock_acc_ts, split_shock_mat


end