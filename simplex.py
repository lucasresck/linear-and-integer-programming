import numpy as np

class Simplex:
    def __init__(self, tableau, inequalities, domain, max_iterations=10000):
        """
        The constructor for the Simplex class.

        As an example, if the optimization problem is

        Maximize    z = c1 x1 + ... + c4 x4 + z0
        subject to
                    a11 x1 + ... + a14 x4 = b1
                    a21 x1 + ... + a24 x4 = b2
                    xi >= 0

        we must have

        tableau =       [[  a11 a12 a13 a14 b1  ]
                        [   a21 a22 a23 a24 b2  ]
                        [   c1  c2  c3  c4  -z0 ]] 
        inequalities =  ['==', '==']
        domain =        ['>=0', '>=0', '>=0', '>=0']
  
        Parameters:
           tableau (numpy.ndarray): Tableau with linear optimization coefficients.
           inequalities (list of str): A list of '<=', '>=' or '==' for the constraints.
           domain (list of str): A list of 'R' or '>=0' indicating the domain of the variables.
        """
        self.tableau = tableau.astype('float')
        self.inequalities = inequalities
        self.domain = domain
        self.max_iterations = max_iterations
        self.canonize()

    def canonize(self):
        '''Canonize the linear programming problem.'''

        # (1) Free variables
        # Convert 'R' variables to difference of '>=0' variables

        for i, domain in list(enumerate(self.domain))[::-1]:
            if domain == 'R':
                self.tableau = np.hstack([self.tableau[:, :(i+1)], -self.tableau[:, [i]], self.tableau[:, (i+1):]])

        # (2) Inequality constraints
        # The inequality constraints are converted to equality
        # via adding slack and surplus variables

        for i, ineq in enumerate(self.inequalities):
            if ineq == '<=':
                self.tableau = np.hstack([self.tableau[:, :-1], np.zeros((len(self.tableau), 1)), self.tableau[:, [-1]]])
                self.tableau[i, -2] = 1
            elif ineq == '>=':
                self.tableau = np.hstack([self.tableau[:, :-1], np.zeros((len(self.tableau), 1)), self.tableau[:, [-1]]])
                self.tableau[i, -2] = -1

    def simplex(self):
        '''
        Apply simplex method to a tableau in canonical form.

        Maximize    z = c1 x1 + ... + c4 x4 + z0
        subject to
                    a11 x1 + ... + a14 x4 = b1
                    a21 x1 + ... + a24 x4 = b2
                    xi >= 0

        We also have
        -- bi >= 0
        -- each constraint has a decision variable xi with coefficient +1
        -- isolated variables do not appear in other constraints
        -- isolated variables have coefficient zero in objective function

        The representation is a tableau such as:

            x1  x2  x3  x4
        [[  a11 a12 a13 a14 b1  ]
        [   a21 a22 a23 a24 b2  ]
        [   c1  c2  c3  c4  -z0 ]]        
        '''

        m, n = np.array(self.tableau.shape) - 1
        
        # Step (0)
        # The problem is already canonical

        for _ in range(self.max_iterations):

            # Step (1)
            # We check optimality

            # If we are optimal
            if self.is_optimal():
                self.generate_solution(n)
                break
            else:

                # Step (2)
                # Choose column s to pivot

                s = np.argmax(self.tableau[-1])
                if self.is_unbounded(s):
                    print('The primal problem is unbounded.')
                    break
                else:
                    
                    # Step (3)
                    # Choose row r to pivot

                    r = self.ratio_test(m, s)

                    # Normalization
                    self.tableau[r] = self.tableau[r] / self.tableau[r, s]

                    # Pivoting
                    self.pivoting(m, r, s)

    def is_optimal(self):
        '''Check optimality for canonical form.'''
        return not np.sum(self.tableau[-1, :-1] > 0)

    def generate_solution(self, n):
        '''Generate solution for the optimal canonical form.'''
        self.solution = []
        for j in range(n):
            if self.tableau[-1][j] != 0:
                self.solution.append(0)
            else:
                i = np.where(self.tableau[:, j] == 1)[0][0]
                self.solution.append(self.tableau[i, -1])
        self.solution = np.array(self.solution)
        self.objective = np.dot(self.solution, self.tableau[-1, :-1]) - self.tableau[-1, -1]

    def is_unbounded(self, s):
        '''Check is objective function is unbounded.'''
        return not np.sum(self.tableau[:-1, s] > 0)

    def ratio_test(self, m, s):
        '''Ratio test for choosing which variable will be introduced into the basis.'''
        b = self.tableau[:-1, -1]
        a_s = self.tableau[:-1, s]
        r = np.argmin([b[i]/a_s[i] if a_s[i] > 0 else np.inf for i in range(m)])
        return r

    def pivoting(self, m, r, s):
        '''Replace the basic variable in row r with variable s via pivoting.'''
        for i in range(m + 1):
            if i == r:
                continue
            self.tableau[i] = self.tableau[i] - self.tableau[r] * self.tableau[i, s]
