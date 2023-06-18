import numpy as np
from Dynamics_Constraints_Controller import *
from scipy.linalg import sqrtm
from ParticleFilter import ParticleFilter
import random

class StateSelectionAlgorithm(ParticleFilter):
    def __init__(self, x0, Cov0, num_particles, stateDynamics, measurementDynamics, Q, R,
                 Pred_Horizon_N, Controller, number_of_simulations, CostAndConstraints, LangrangeMultp):
        super().__init__(x0, Cov0, num_particles, stateDynamics, measurementDynamics, Q, R)
        self.Pred_Horizon_N = Pred_Horizon_N
        self.Controller = Controller
        self.LangrangeMultp = LangrangeMultp
        self.number_of_simulations = number_of_simulations
        self.CostAndConstraints = CostAndConstraints


    def sample_xk_prime(self, x0prime): #Generating state sequence x_k' for k=0,...,N-1
        xkprime=np.full((rx, self.Pred_Horizon_N), np.nan)
        Wprime=np.sqrt(self.Q)@np.random.randn(rx, self.Pred_Horizon_N)
        xkprime[:,0]=x0prime
        for k in range(self.Pred_Horizon_N-1):
            xkprime[:,k+1]=stateDynamics(xkprime[:,k],controller(xkprime[:,k]),Wprime[:,k])   
        return xkprime

    def sample_xk_dblPrime(self,x0prime): #Generating state sequence x_k'' for k=0,...,N-1
        xkprimeW = self.sample_xk_prime(x0prime)
        xk2prime=np.full((2, self.Pred_Horizon_N+1), np.nan)
        W2prime=np.sqrt(self.Q)@np.random.randn(rx, self.Pred_Horizon_N)
        x02prime=self.particles[:,random.sample(range(0, self.num_particles), 1)]
        xk2prime[:,0]=x02prime.reshape(rx,)
        for k in range(self.Pred_Horizon_N):
            u=controller(xkprimeW[:,k])
            xk2prime[:,k+1]=stateDynamics(xk2prime[:,k],u,W2prime[:,k])    
        return xkprimeW, xk2prime

    # State Selection Algorithm and related functions
    def StateSelector(self):
        StateCandidateCost=np.zeros((self.num_particles,))
        ViolationRate=np.zeros((self.num_particles,))
        for i in range(self.num_particles):
            x0prime=self.particles[:,i]
            for j in range(self.number_of_simulations):
                xkprimeW, xk2prime = self.sample_xk_dblPrime(x0prime)
                cost, number_of_violations = self.CostAndConstraints(xkprimeW, xk2prime)
                StateCandidateCost[i] += (cost + 
                    self.LangrangeMultp * number_of_violations) / self.Pred_Horizon_N
                
            StateCandidateCost[i] = StateCandidateCost[i] / self.number_of_simulations
        minCost=StateCandidateCost.argmin()
        x0star=self.particles[:,minCost]
            return x0star

        def CostAndConstraints(self,xkprimeW,xk2prime):
            ConstraintX=0
            ConstraintU=0
            x=state[0]
            y=state[1]
            if ((3<x<5)&(-4<y<2)|(-2<x<5)&(-7<y<-4)):
                    ConstraintX=1
            if (abs(u)>Ulim):
                    ConstraintU=1
            return np.array([ConstraintX, ConstraintU])


    def checkConstraints(xkprimeW,xk2primeW,Ulim,N): # Count constraints violations
        Violation_X=np.zeros([1, N])
        Violation_U=np.zeros([1, N])
        for k in range(N):
            u=controller(xkprimeW[:,k])
            x=xk2primeW[:,k+1]
            Check_constraints=Constraints(u,x,Ulim)
            Violation_X[0,k]=Check_constraints[0]
            Violation_U[0,k]=Check_constraints[1]
        return Violation_X, Violation_U


    def AchievedAlpha(Xi): #calculates violation rates
        AlphaAchieved=0
        N=Xi.shape[1]
        for i in range(N):
            x=Xi[:,i]
            check=Constraints(0,x,10**6)
            AlphaAchieved += check[0]


        AlphaAchieved=AlphaAchieved/N
        return AlphaAchieved


