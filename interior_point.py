import numpy as np

class InteriorPoint:
    '''
    Interior-point method implementation, specifically
    the short-step primal-dual path-following algorithm.

    It applies Newton's method to the perturbed
    primal-dual conditions, and iterate inside feasible
    region to the solution. When simplex method iterate
    over the vertices of the feasible solution, interior
    point methos iterate inside the polyhedron.
    
    
    Parameters:
        A (numpy array): A matrix of linear programming problem.
        initial_point (tuple of numpy arrays): Initial feasible
            solution (x_0, y_0, s_0).
        sigma (float): number between 0 and 1, exclusive.
        eps (float): positive tolerance for iteration.

    Attributes:
        m (int): dimensionality of y. Also the one of
            dimensions of A (m x n).
        n (int): dimensionality of x and s. Also the one of
            dimensions of A (m x n).
        mu (float): the duality measure.
    '''

    def __init__(self, A, initial_point, sigma, eps):
        self.A = A
        self.m, self.n = self.A.shape
        self.initial_point = initial_point
        self.point = self.initial_point.copy()
        self.sigma = sigma
        self.epsilon = eps
        self.mu = self.initial_measure()
        self.iteration()

    def initial_measure(self):
        '''Initial duality measure.'''
        self.mu = np.dot(self.initial_point[0], self.initial_point[1])/self.n

    def iteration(self):
        '''
        Method iteration.

        It starts at the initial solution,
        and iterates until reaches the tolerance.
        '''

        while self.n*self.mu < self.epsilon:
            step = self.calculate_step()
            self.point += step
            self.mu *= self.sigma
        
        self.solution = self.point.copy()

    def calculate_step(self):
        '''
        Calculate the step of algorithm.

        In order to do this, it solves the linear system
        resulting of applying Newton's method to the perturbed
        primal-dual conditions. The system looks like this:

        |0      A^T     -I  |   |\Delta x_k|        |0                          |
        |A      0       0   |   |\Delta y_k|    =   |0                          | 
        |S_k    0       X_k |   |\Delta s_k|        |-X_k*S_k*e + \sigma \mu_k e|
        '''
        M = self.get_M_matrix()
        b = self.get_b_vector()
        step = np.linalg.solve(M, b)
        return step

    def get_M_matrix(self):
        '''
        Create the necessary matrix.
        
                |0      A^T     -I  |
        M   =   |A      0       0   |
                |S_k    0       X_k |
        '''
        A = self.A
        m = self.m
        n = self.n
        x, y, s = self.point
        X = np.diag(x)
        S = np.diag(s)
        
        M = np.vstack([
            np.hstack([np.zeros((n, n)), A.transpose(), -np.identity(n)]),
            np.hstack([A, np.zeros((m, m+n))),
            np.hstack([S, np.zeros(n, m), X])
        ])
        return M

    def get_b_vector(self):
        '''
        Create the necessary vector.
        
                |0                          |
        b   =   |0                          |
                |-X_k*S_k*e + \sigma \mu_k e|
        '''
        sigma = self.sigma
        mu = self.mu
        e = np.ones((self.n, 1))
        x, y, s = self.point
        X = np.diag(x)
        S = np.diag(s)

        b = np.vstack([
            np.zeros(),
            -np.matmul(np.matmul(X, S), e) + sigma*mu*e
        ])
        return b
