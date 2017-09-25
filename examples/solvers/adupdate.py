# This tiny example wants to solve the problem
# 
# ( 1  1/2 1/3 1/4) (x_1)   (a)
# (1/2 1/3 1/4 1/5) (x_2) = (b)
# (1/3 1/4 1/5 1/6) (x_3)   (c)
# (1/4 1/5 1/6 1/7) (x_4)   (d),
#
#
# the solution of which is given in closed form by
#
# (  16  -120   240  -140) (a)
# (-120  1200 -2700  1680) (b)
# ( 240 -2700  6480 -4200) (c)
# (-140  1680 -4200  2800) (d).
#
# In the case (a, b, c, d) = 1, we obtain the solution (-4, 60, -660, 140)
#
# We do this by the algorithm of Fessler and McGaffin and TV regularization
# (which is only for testing purposes and has no practical relevance).

import numpy
import odl

class MatrixOperator(odl.Operator):
    def __init__(self, matrix):
        self.matrix = matrix
        dom = odl.rn(matrix.shape[1])
        ran = odl.rn(matrix.shape[0])
        odl.Operator.__init__(self, dom, ran)

    def _call(self, x, out):
        return self.matrix.dot(x, out=out.asarray())

def adupgrades(funcs, ops, majs, x, rhs, mu, niter, random=False, callback=None,
        callback_loop='outer'):
    '''Implements the alternating dual upgrade method of Fessler

    Solves the optimization problem

        minimize sum_{i = 1}^m L_i(A_i x)

    under the conditions that

    * L_i is a convex functional whose proximal points can be calculated,
    * A_i is a linear mapping such that
    * either A_i A_i^\\top is diagonal or it can be majorized by a diagonal matrix

    Parameters
    ----------
    funcs : sequence of functionals
    ops : sequence of `Operator`'s
        Linear transformations.
    majs : sequence of symmetric, positive definite operators
    x : ``op.domain`` element
        Element to which the result is written. Its initial value is
        used as starting point of the iteration, and its values are
        updated in each iteration step.
    rhs : sequence of ``ops[i].range`` elements
        Right-hand side of the equation defining the inverse problem.
    mu : positive real number
        A damping/stepsize parameter for the algorithm
    niter : int
        Number of iterations.
    random : bool, optional
        If `true`, the order of the operators is randomized in each iteration.
    callback : callable, optional
        Object executing code per iteration, e.g. plotting each iterate.
    callback_loop : {'inner', 'outer'}
        Whether the callback should be called in the inner or outer loop.

    Notes
    -----
    This method calculates an approximate least-squares solution of
    the inverse problem of the first kind

    References
    ----------
    Madison G. McGaffin and Jeffrey A. Fessler, "Alternating dual
    updates algorithm for X-ray CT reconstruction on the GPU", IEEE
    Transactions on Computational Imaging, 1(3):186-199 (2015).
    '''
    domain = ops[0].domain
    if any(domain != opi.domain for opi in ops):
        raise ValueError('`ops[i].domain` are not all equal')
    else:
        print('Common domain of all elements of `ops` is {!r}. Ok.'.format(domain))

    # TODO: check if ranges of the operators match the domains of the functionals

    if x not in domain:
        raise TypeError('`x` {!r} is not in the domain of `ops` {!r}'
                        ''.format(x, domain))
    else:
        print('Initial value `x` {!r} is an element of the common domain. Ok.'.format(x))

    if len(ops) != len(rhs) or len(ops) != len(funcs):
        raise ValueError('`number of `ops` {} does not match number of '
                         '`rhs` {} or number of `funcs` {}'.format(len(ops), len(rhs), len(funcs)))
    else:
        print('Solving problem for {} operators and right-hand sides. Ok.'.format(len(ops)))

    # This is from the Kaczmarz implementation and has to be adapted to our parameters
    # omega = normalized_scalar_param_list(omega, len(ops), param_conv=float)

    # Get all the range spaces of our operators
    ranges = [opi.range for opi in ops]

    # Containers for the dual variables,
    # according to Fessler initialized with zero.
    # TODO: In the future, this behavior possibly could/should be changed.
    duals = [ran.zero() for ran in ranges]

    # Reusable elements in the range, one per type of space
    unique_ranges = set(ranges)
    tmp_rans = {ran: ran.element() for ran in unique_ranges}

    # Single reusable element in the domain
    tmp_dom = domain.element()

    # auxiliary element in the domain

    # Iteratively find solution
    for _ in range(niter):
        # TODO: Form the prospective next iteration of the primal variable
        # xtilde = x - 1/mu * sum(ops[i].adjoint(duals[i], i)
        if random:
            rng = numpy.random.permutation(range(len(ops)))
        else:
            rng = range(len(ops))

        for i in rng:
            # TODO: Calculate the new dual variable i
            # (we shall still need the old one, thus, we use a temporary container)
            tmp_ran = tmp_rans[ops[i].range]
            # tmp_ran = ... (Proximal point with respect to majs[i])

            # TODO: Update xtilde with ops[i] and difference of tmp_ran and duals[i]
            # xtilde = ... (xtilde - 1/mu ops[i].adjoint(tmp_ran - duals[i])

            duals[i] = tmp_ran

            # Make dual updates

            # Update the primal variable

            # From Kaczmarz:
            # Find residual
            # tmp_ran = tmp_rans[ops[i].range]
            # ops[i](x, out=tmp_ran)
            # tmp_ran -= rhs[i]

            # Update x
            # ops[i].derivative(x).adjoint(tmp_ran, out=tmp_dom)
            # x.lincomb(1, x, -omega[i], tmp_dom)

            # if projection is not None:
            #     projection(x)

            if callback is not None and callback_loop == 'inner':
                callback(x)
        if callback is not None and callback_loop == 'outer':
            callback(x)



# Create the solution space
input_space = odl.rn(4)
output_space1 = odl.rn(4)
output_space1 = odl.rn(4)

# Matrices for finite differences
fd1m = numpy.array( [[1, -1,  0,  0],
                     [0,  0,  1, -1]] )
fd2m = numpy.array( [[0,  1, -1,  0]] )

# Matrices for the system
sys1m = numpy.array( [[    1.0, 1.0/2.0, 1.0/3.0, 1.0/4.0],
                      [1.0/2.0, 1.0/3.0, 1.0/4.0, 1.0/5.0]] )
sys2m = numpy.array( [[1.0/3.0, 1.0/4.0, 1.0/5.0, 1.0/6.0],
                      [1.0/4.0, 1.0/5.0, 1.0/6.0, 1.0/7.0]] )

# Majorising matrices
maj1m = numpy.diag(numpy.dot(numpy.dot(sys1m, sys1m.transpose()), [1, 1]))
maj2m = numpy.diag(numpy.dot(numpy.dot(sys2m, sys2m.transpose()), [1, 1]))

# ODL Operators for the linear transformations
fd1 = MatrixOperator(fd1m)
fd2 = MatrixOperator(fd2m)
fds = [fd1, fd2]

sys1 = MatrixOperator(sys1m)
sys2 = MatrixOperator(sys2m)
syss = [sys1, sys2]

maj1 = MatrixOperator(maj1m)
maj2 = MatrixOperator(maj2m)
majs = [maj1, maj2]

rhs1 = odl.rn(2).element(numpy.array( [1, 1] ))
rhs2 = odl.rn(2).element(numpy.array( [1, 1] ))
rhs = [rhs1, rhs2]
x = odl.rn(4).element(numpy.array( [0, 0, 0, 0] ))

adupgrades(syss, x, rhs, 30)
