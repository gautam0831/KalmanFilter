import numpy as np

class KalmanFilter(object):
    
    def __init__(self, dt, u_x, u_y, std_acc, x_std_measure, y_std_measure):
        """
        dt: sampling time per cycle
        u_x: acceleration in x-direction
        u_y: acceleration in y-direction
        std_acc: process noise magnitude
        x_std_measure: standard deviation of the measurement in the x-direction
        y_std_measure: standard deviation of the measurement in the y-direction
        """
        
        self.dt = dt
        self.u = np.matrix([[u_x], [u_y]])
        #Define initial state
        self.x = np.matrix([[0], [0], [0], [0]])
        #Define state transformation matrix A
        self.A = np.matrix([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        #Define control input matrix B
        self.B = np.matrix([[(self.dt**2)/2, 0], [0, (self.dt**2)/2], [self.dt, 0], [0, self.dt]])
        #Define measurement mapping matrix H
        self.H = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0]])
        #Define initial process noise covariance matrix Q
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0], 
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2], 
                            [(self.dt**3)/2, 0, self.dt**2, 0], 
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2

        #Define initial measurement noise covariance matrix R
        self.R = np.matrix([[x_std_measure**2, 0], 
                            [0, y_std_measure**2]])

        #Initial covariance matrix
        self.P = np.eye(self.A.shape[1])

    
    def predict(self):
        #Time update process that projects forward the current state to the next time step

        #Update time state
        '''
        x_k = A * x_(k-1) + B * u_(k-1)
        '''
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)

        #Calculate error cov
        '''
        P_k = A*P_(k-1)*A^T + Q
        '''
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        
        return self.x[:2]
    
    def update(self, z):
        #Computing Kalman gain, then updating predicted state estimate and predicted error cov

        #Get Kalman gain K
        inv = np.linalg.inv(np.dot(self.H, np.dot(self.P, self.H.T)) + self.R)
        K = np.dot(np.dot(self.P, self.H.T), inv)

        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))
        I = np.eye(self.H.shape[1])

        #Update error cov matrix P
        self.P = (I - K * self.H) * self.P

        return self.x[0:2]

        
