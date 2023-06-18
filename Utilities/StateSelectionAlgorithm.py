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
        Control_seq = np.full((ru, self.Pred_Horizon_N), np.nan)
        Wprime=np.sqrt(self.Q)@np.random.randn(rx, self.Pred_Horizon_N)
        xkprime[:,0]=x0prime
        for k in range(self.Pred_Horizon_N-1):
            Control_seq[:,k] = controller(xkprime[:,k])
            xkprime[:,k+1]=stateDynamics(xkprime[:,k], Control_seq[:,k], Wprime[:,k])
        Control_seq[:,k+1] = controller(xkprime[:,k+1])
        return xkprime, Control_seq

    def sample_xk_dblPrime(self,x0prime): #Generating state sequence x_k'' for k=0,...,N-1
        _, Control_seq = self.sample_xk_prime(x0prime)
        xk2prime = np.full((2, self.Pred_Horizon_N+1), np.nan)
        W2prime = np.sqrt(self.Q)@np.random.randn(rx, self.Pred_Horizon_N)
        x02prime = self.particles[:,random.sample(range(0, self.num_particles), 1)]
        xk2prime[:,0]=x02prime.reshape(rx,)
        for k in range(self.Pred_Horizon_N):
            u = Control_seq[:,k]
            xk2prime[:,k+1]=stateDynamics(xk2prime[:,k],u,W2prime[:,k])    
        return Control_seq, xk2prime

    # State Selection Algorithm and related functions
    def StateSelector(self):
        StateCandidateCost=np.zeros((self.num_particles,))
        ViolationRate=np.zeros((self.num_particles,))
        for i in range(self.num_particles):
            x0prime=self.particles[:,i]
            for j in range(self.number_of_simulations):
                Control_seq, xk2prime = self.sample_xk_dblPrime(x0prime)
                cost, number_of_violations = self.CostAndConstraints(Control_seq, xk2prime)
                StateCandidateCost[i] += (cost + 
                    self.LangrangeMultp * number_of_violations) / self.Pred_Horizon_N
                
            StateCandidateCost[i] = StateCandidateCost[i] / self.number_of_simulations
        minCost=StateCandidateCost.argmin()
        x0star=self.particles[:,minCost]
        return x0star

    def CostAndConstraints(self,Control_seq,xk2prime):
        cost = (xk2prime ** 2).sum() + (Control_seq ** 2).sum()
        number_of_violations = 0
        Length = xk2prime.shape[1]
        """
        for j in range(Length):

            if ((3<x<5)&(-4<y<2)|(-2<x<5)&(-7<y<-4)):
                    ConstraintX=1
            if (abs(u)>Ulim):
                    ConstraintU=1
        """
        return cost, number_of_violations

    def ViolationProb(self): #calculates violation rates
        num_of_violations=0
        for i in range(self.num_particles):
            x=self.particles[:,i]
            _, x_violates=self.CostAndConstraints(0,x,10**6)
            num_of_violations += x_violates
        ViolationRate = num_of_violations / self.num_particles
        return ViolationRate


