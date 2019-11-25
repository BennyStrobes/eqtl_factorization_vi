import numpy as np 
import os
import sys
import pdb
import eqtl_factorization_vi 



def simulate_data_for_eqtl_factorization(I, Ni, T, K, alpha_0, beta_0, a_0, b_0):
	# Simulate gammas
	gamma_U = np.random.gamma(shape=alpha_0, scale=1.0/beta_0,size=K)
	gamma_V = np.random.gamma(shape=alpha_0, scale=1.0/beta_0,size=K)
	gamma_F = np.random.gamma(shape=alpha_0,scale=1.0/beta_0)
	# Simulate thetas
	theta_U = np.random.beta(a_0, b_0, size=K)
	theta_V = np.random.beta(a_0, b_0, size=K)
	theta_F = np.random.beta(a_0, b_0)
	# Simulate other variances
	psi = np.random.gamma(shape=alpha_0, scale=1.0/beta_0, size=T)
	tau = np.random.gamma(shape=alpha_0, scale=1.0/beta_0, size=T)
	# Simulate alphas (random effects)
	alphas = np.zeros((I, T))
	for test_index in range(T):
		for individual_index in range(I):
			alphas[individual_index, test_index] = np.random.normal(loc=0, scale=np.sqrt(1.0/psi[test_index]))
	# Simulate Us
	sample_num = 0
	U = np.zeros((I*Ni, K))
	S_U = np.zeros((I*Ni,K))
	Z = []
	for individual_index in range(I):
		for ni in range(Ni):
			Z.append(str(individual_index))
			for k in range(K):
				U[sample_num, k] = np.random.normal(loc=0, scale=np.sqrt(1.0/gamma_U[k]))
				S_U[sample_num, k] = np.random.binomial(n=1,p=theta_U[k])
			sample_num = sample_num + 1
	# Simulate Vs
	V = np.zeros((K, T))
	S_V = np.zeros((K,T))
	F = np.zeros(T)
	S_F = np.zeros(T)
	for test_index in range(T):
		F[test_index] = np.random.normal(loc=0, scale=np.sqrt(1.0/gamma_F))
		S_F[test_index] = np.random.binomial(n=1,p=theta_F)
		for k in range(K):
			V[k,test_index] = np.random.normal(loc=0, scale=np.sqrt(1.0/gamma_V[k]))
			S_V[k, test_index] = np.random.binomial(n=1,p=theta_V[k])
	# Simulate Genotypes
	G = np.zeros((I*Ni, T))
	for test_number in range(T):
		sample_num = 0
		genotypes = np.random.normal(0,1,size=I)
		for individual_index in range(I):
			for ni in range(Ni):
				G[sample_num, test_number] = genotypes[individual_index]
				sample_num = sample_num + 1
	# Simulate Expression
	pdb.set_trace()
	Y = np.zeros((I*Ni, T))
	sample_num = 0
	for individual_index in range(I):
		for ni in range(Ni):
			for test_number in range(T):
				predicted_mean = 0
				for k in range(K):
					predicted_mean = predicted_mean + U[sample_num,k]*S_U[sample_num,k]*V[k, test_number]*S_V[k,test_number]
				predicted_mean = predicted_mean + F[test_number]*S_F[test_number]
				predicted_mean = predicted_mean + alphas[individual_index, test_number]
				Y[sample_num, test_number] = np.random.normal(loc=predicted_mean, scale=np.sqrt(1.0/tau[test_number]))
			sample_num = sample_num + 1
	data = {}
	data['V'] = V
	data['S_V'] = S_V
	data['U'] = U
	data['S_U'] = S_U
	data['F'] = F
	data['S_F'] = S_F
	data['psi'] = psi
	data['tau'] = tau
	data['theta_U'] = theta_U
	data['theta_V'] = theta_V
	data['theta_F'] = theta_F
	data['gamma_V'] = gamma_V
	data['gamma_U'] = gamma_U
	data['gamma_F'] = gamma_F
	data['alpha'] = alphas
	return Y, G, Z, data


def simulate_data_for_eqtl_factorization2(I, Ni, T, K):
	N = I*Ni
	U_true = np.random.randn(N, K)
	V_true = np.random.randn(K, T)
	G = np.random.randn(N, T)
	F_true = np.random.randn(1,T)


	# Simulate thetas
	theta_U = np.random.beta(a_0, b_0, size=K)
	theta_V = np.random.beta(a_0, b_0, size=K)
	theta_F = np.random.beta(a_0, b_0)

	S_U = np.zeros((N,K))
	for n in range(N):
		for k in range(K):
			S_U[n,k] = np.random.binomial(n=1,p=theta_U[k])
	S_V = np.zeros((K,T))
	for k in range(K):
		for t in range(T):
			S_V[k,t] = np.random.binomial(n=1,p=theta_V[k])
	S_F = np.zeros((1,T))
	for t in range(T):
		S_F[0,t] = np.random.binomial(n=1,p=theta_F)

	Y = G*np.dot(U_true*S_U, V_true*S_V) + G*np.dot(np.ones((N,1)),F_true*S_F)
	gene_residual_sdevs = np.sqrt(np.random.exponential(size=T))
	for m in range(T):
		Y[:,m] = Y[:, m] + np.random.normal(0, gene_residual_sdevs[m],size=N)
	sample_num = 0
	Z = []
	for individual_index in range(I):
		for ni in range(Ni):
			Z.append(str(individual_index))


	gene_random_effects_sdevs = np.sqrt(np.random.exponential(size=T))
	alphas = np.zeros((I, T))
	for t in range(T):
		sample_num = 0
		for individual_index in range(I):
			individual_intercept = np.random.normal(0, gene_random_effects_sdevs[t])
			alphas[individual_index, t] = individual_intercept
			for ni in range(Ni):
				Y[sample_num, t] = Y[sample_num, t] + individual_intercept
				sample_num = sample_num + 1
	data = {}
	data['U'] = U_true
	data['V'] = V_true
	data['resid'] = gene_residual_sdevs
	data['theta_U'] = theta_U
	data['theta_V'] = theta_V
	data['theta_F'] = theta_F
	data['S_U'] = S_U
	data['S_V'] = S_V
	data['S_F'] = S_F
	data['alphas'] = alphas
	data['random_effect_sdev'] = gene_random_effects_sdevs
	return Y, G, Z, data



#########################
# Set Seed
#########################
np.random.seed(1)

##########################
# Simulate data
##########################
I = 22
Ni = 30
T = 3000
K = 3
alpha_0 = 1e-3
beta_0 = 1e-3
a_0 = 1
b_0 = 1

#Y, G, z, data = simulate_data_for_eqtl_factorization(I, Ni, T, K, 1, 1, a_0, b_0)
Y, G, z, data = simulate_data_for_eqtl_factorization2(I, Ni, T, K)

##################
# Fit eqtl factorization using home-built variational inference
##################
eqtl_vi = eqtl_factorization_vi.EQTL_FACTORIZATION_VI(K=5, alpha=1e-3, beta=1e-3, a=1, b=1, max_iter=3000, delta_elbo_threshold=.01)
eqtl_vi.fit(G=G, Y=Y, z=z)
pdb.set_trace()
