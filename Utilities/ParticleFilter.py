import numpy as np
from scipy.linalg import sqrtm

# Particle filter for multivariate Gaussian disturbance and measurement noise

class Model:
    def __init__(self, stateDynamics, measurementDynamics, Q, R):
        self.f = stateDynamics
        self.g = measurementDynamics
        self.Q = np.atleast_2d(Q)
        self.R = np.atleast_2d(R)  


class ParticleFilter(Model): #x0 2D column vector
    def __init__(self, x0, Cov0, number_of_samples, stateDynamics, measurementDynamics, Q, R):
        super().__init__(stateDynamics, measurementDynamics, Q, R)
        self.rx = self.Q.shape[0]
        self.ry = self.R.shape[0]
        self.number_of_samples = number_of_samples
        self.particles = x0 + sqrtm(np.atleast_2d(Cov0)).real @ np.random.randn(self.rx, self.number_of_samples)
        self.likelihoods = np.ones((self.number_of_samples,)) / self.number_of_samples

    def initialize(self, x0, Cov0):
        self.particles = x0 + sqrtm(np.atleast_2d(Cov0)).real @ np.random.randn(self.rx, self.number_of_samples)
        self.likelihoods = np.ones((self.number_of_samples,)) / self.number_of_samples

    def sampleAverage(self):
        hat_x=(self.likelihoods * self.particles).sum(axis = 1)
        return hat_x
    
    def sampleCov(self):
        hat_x = self.sampleAverage()
        hat_Cov = (self.likelihoods * (self.particles - hat_x)) @ (self.particles - hat_x).T
        return hat_Cov
    
    def MeasurementUpdate(self, u, y):
        log_prob_perSample = np.full((self.number_of_samples,), np.nan)
        y = y.squeeze()
        u = u.squeeze()
        for j in range(self.number_of_samples):
            xj = self.particles[:,j]
            measurementError = y - self.g(xj, u)
            measurementError = measurementError.squeeze()
            log_prob_perSample[j] = -1/2 * measurementError @ np.linalg.inv(self.R) @ measurementError
        Likelihoods=np.exp(log_prob_perSample)
        self.likelihoods = self.likelihoods * Likelihoods/(self.likelihoods * Likelihoods).sum() #Normalizing Likelihoods vector

    def TimeUpdate(self, u):
        Xiplus=np.full((self.rx, self.number_of_samples), np.nan)
        for j in range(self.number_of_samples):
            xj = self.particles[:,j]
            xj_plus = self.f(xj, u) + sqrtm(self.Q).real @ np.random.randn(self.rx,)
            Xiplus[:,j] = xj_plus.squeeze()
        self.particles = Xiplus
    
    def Resampler(self):
        x_resampled=np.full((self.rx, self.number_of_samples), np.nan)
        CDF=self.likelihoods.cumsum()
        for i in range(self.number_of_samples):
            I=np.array(np.where(CDF>=np.random.rand(1,))) #CDF inverse of a uniformly sampled point in (0,1)
            x_resampled[:,i]=self.particles[:,I.min()] #Corresponding particle
        self.particles = x_resampled
        self.likelihoods = np.ones((self.number_of_samples,)) / self.number_of_samples
    
    def Apply_PF(self, u, y):
        self.TimeUpdate(u)
        self.MeasurementUpdate(u, y)
        self.Resampler()