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
        print('Canonizing the linear programming problem...')

        # (1) Free variables
        # Convert 'R' variables to difference of '>=0' variables

        print('    Converting free variables to nonnegative variables...', end='')

        self.convert_free_var()

        print(' Done.')

        # (2) Inequality constraints
        # The inequality constraints are converted to equality
        # via adding slack and surplus variables

        print('    Converting inequalities to equalities using slack and surplus variables...', end='')

        self.convert_inequalities()

        print(' Done.')

        # (3) Negative righthand side coefficients
        # Multiply equations with a negative righthand side coefficient by −1

        print('    Multiplying equations with a negative righthand side coefficient by −1...', end='')

        self.multiply_minus_one()

        print(' Done.')

        # (4) Artificial variables
        # Create a basis if there is not one

        print('    Checking artificial variables needed...', end='')

        self.generate_basis()

        artificial_needed = self.check_artificial_variables()

        print(' We need {} artificial variable(s).'.format(len(artificial_needed)))

        if len(artificial_needed) > 0:
            print('        Adding artificial variable(s)...', end='')

            self.add_artificial_variables(artificial_needed)
            
            print(' Done.')
            print('        Starting Phase I...', end='')

            self.phase_i(artificial_needed)

            print(' Done.')

            # If the artificial variables are not zero...
            if not self.is_feasible(artificial_needed):
                print('The problem if not feasible.')
                return
            else:
                pass

        print('Finished canonization.')

    def convert_free_var(self):
        '''Convert free variables to nonnegative variables.'''
        for i, domain in list(enumerate(self.domain))[::-1]:
            if domain == 'R':
                self.tableau = np.hstack([
                    self.tableau[:, :(i+1)],
                    -self.tableau[:, [i]],
                    self.tableau[:, (i+1):]
                ])

    def convert_inequalities(self):
        '''Convert inequalities to equalities, adding slack and surplus variables.'''
        for i, ineq in enumerate(self.inequalities):
            if ineq == '<=':
                self.tableau = np.hstack([
                    self.tableau[:, :-1],
                    np.zeros((len(self.tableau), 1)),
                    self.tableau[:, [-1]]
                ])
                self.tableau[i, -2] = 1
            elif ineq == '>=':
                self.tableau = np.hstack([
                    self.tableau[:, :-1],
                    np.zeros((len(self.tableau), 1)),
                    self.tableau[:, [-1]]
                ])
                self.tableau[i, -2] = -1

    def multiply_minus_one(self):
        '''Multiply equations with a negative righthand side coefficient by −1.''' 
        for i in range(len(self.tableau) - 1):
            if self.tableau[i, -1] < 0:
                self.tableau[i, :] = -1*self.tableau[i, :]

    def generate_basis(self):
        '''Generate a basis for the canonical linear programming problem.'''
        self.basis = dict()
        already_i = []
        for j in range(self.tableau.shape[1] - 1):
            if np.sum(self.tableau[:, j] != np.zeros(len(self.tableau))) == 1:
                if np.max(self.tableau[:, j]) == 1:
                    i = np.argmax(self.tableau[:, j])
                    if i not in already_i:
                        self.basis.update({i: j})
                        already_i.append(i)

    def check_artificial_variables(self):
        '''Check if artificial variables are needed.'''
        artificial_not_needed = self.basis.keys()

        artificial_needed = set(range(self.tableau.shape[0] - 1)) - set(artificial_not_needed)
        artificial_needed = list(artificial_needed)
        return artificial_needed

    def add_artificial_variables(self, artificial_needed):
        '''Add needed artificial variables.'''
        self.tableau = np.hstack([
            self.tableau[:, :-1],
            np.zeros((len(self.tableau), len(artificial_needed))),
            self.tableau[:, [-1]]
        ])

        artificial_needed = sorted(artificial_needed, reverse=True)

        self.artificial_variables = set()
        for j, i in enumerate(artificial_needed):
            self.tableau[i, -2-j] = 1
            self.artificial_variables.add(-2-j)
            self.basis.update({i: -2-j})

    def phase_i(self, artificial_needed):
        '''Phase 1 of simplex algorithm.

        Solve the problem of determining the initial solution, if it exists.'''
        w_objective = np.sum(self.tableau[artificial_needed], axis=0)
        w_objective[list(self.artificial_variables)] = 0
        self.tableau = np.vstack([self.tableau, w_objective])
        self.simplex(phase_i=True)

    def is_feasible(self, artificial_needed):
        '''Check if the problem is feasible.
        
        Check if the artificial variables, in the solution of Phase I, are all zero'''        
        return np.sum(np.abs(self.solution[list(self.artificial_variables)])) == 0

    def simplex(self, phase_i=False):
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

                s = np.argmax(self.tableau[-1, :-1])

                # If we are in Phase I, we must not consider z objective row
                # when checking unboundedness
                if self.is_unbounded(s, phase_i=phase_i):
                    print('The primal problem is unbounded.')
                    break
                else:
                    
                    # Step (3)
                    # Choose row r to pivot

                    # In Phase I, we can't consider z objective row
                    # in ratio test
                    r = self.ratio_test(m - int(phase_i), s)

                    # Normalization
                    self.tableau[r] = self.tableau[r] / self.tableau[r, s]

                    # Pivoting
                    self.pivoting(m, r, s)

    def is_optimal(self):
        '''Check optimality for canonical form.'''
        return not np.sum(self.tableau[-1, :-1] > 0)

    def generate_solution(self, n):
        '''Generate solution for the optimal canonical form.'''
        # Our solution starts with everything equal to zero
        self.solution = [0]*n
        for i, j in self.basis.items():
            self.solution[j] = self.tableau[i, -1]
        self.solution = np.array(self.solution)
        self.objective = np.dot(self.solution, self.tableau[-1, :-1]) - self.tableau[-1, -1]

    def is_unbounded(self, s, phase_i):
        '''Check is objective function is unbounded.'''
        return not np.sum(self.tableau[:-1-int(phase_i), s] > 0)

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
        self.basis.update({r: s})
