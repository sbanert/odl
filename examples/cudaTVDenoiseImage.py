# -*- coding: utf-8 -*-
"""
simple_test_astra.py -- a simple test script

Copyright 2014, 2015 Holger Kohr

This file is part of RL.

RL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RL.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import division, print_function, unicode_literals, absolute_import
from future import standard_library

import numpy as np
from RL.operator.operator import *
from RL.space.space import *
from RL.space.product import powerspace
from RL.space.euclidean import *
from RL.space.function import *
import RL.space.cuda as CS
import RL.space.discretizations as DS
import RLcpp

from pooled import makePooledSpace

from RL.utility.testutils import Timer

import matplotlib.pyplot as plt

standard_library.install_aliases()


class RegularizationType(object):
    Anisotropic, Isotropic = range(2)

class ForwardDiff2D(LinearOperator):
    """ Calculates the circular convolution of two CUDA vectors
    """

    def __init__(self, space):
        if not isinstance(space, CS.CudaRN):
            raise TypeError("space must be CudaPixelDiscretization")

        self.domain = space
        self.range = powerspace(space, 2)

    def _apply(self, rhs, out):
        RLcpp.cuda.forwardDiff2D(rhs.data, out[0].data, out[1].data,
                                 self.domain.shape[0], self.domain.shape[1])

    @property
    def adjoint(self):
        return ForwardDiff2DAdjoint(self.domain)


class ForwardDiff2DAdjoint(LinearOperator):
    """ Calculates the circular convolution of two CUDA vectors
    """

    def __init__(self, space):
        if not isinstance(space, CS.CudaRN):
            raise TypeError("space must be CudaPixelDiscretization")

        self.domain = powerspace(space, 2)
        self.range = space

    def _apply(self, rhs, out):
        RLcpp.cuda.forwardDiff2DAdj(rhs[0].data, rhs[1].data, out.data,
                                    self.range.shape[0], self.range.shape[1])

    @property
    def adjoint(self):
        return ForwardDiff2D(self.range)


def TVdenoise2DIsotropic(x0, la, mu, iterations=1):
    diff = ForwardDiff2D(x0.space)

    dimension = 2

    L2 = diff.domain
    L2xL2 = diff.range

    x = x0.copy()
    f = x0.copy()
    b = L2xL2.zero()
    d = L2xL2.zero()
    xdiff = L2xL2.zero()
    tmp = L2.zero()

    C1 = mu/(mu+2*la)
    C2 = la/(mu+2*la)

    for i in range(iterations):
        # x = ((f * mu + (diff.T(diff(x)) + 2*x + diff.T(d-b)) * la)/(mu+2*la))
        diff.apply(x, xdiff)
        xdiff += d
        xdiff -= b
        diff.adjoint.apply(xdiff, tmp)

        L2.lincomb(C1, f, 2*C2, x)
        x.lincomb(C2, tmp)

        # d = diff(x)+b
        diff.apply(x, xdiff)
        xdiff += b
        d.assign(xdiff)

        # s = xdiff[0] = sqrt(dx^2+dy^2)
        for i in range(dimension):
            xdiff[i].multiply(xdiff[i])

        for i in range(1, dimension):
            xdiff[0].lincomb(1, xdiff[i])

        CS.sqrt(xdiff[0], xdiff[0])

        # c = tmp = max(s - la^-1, 0) / s
        CS.addScalar(xdiff[0], -1.0/la, tmp)
        CS.maxVectorScalar(tmp, 0.0, tmp)
        CS.divideVectorVector(tmp, xdiff[0], tmp)

        # d = d * c = d * max(s - la^-1, 0) / s
        for i in range(dimension):
            d[i].multiply(tmp)

        # b = b + diff(x) - d
        diff.apply(x, xdiff)
        b += xdiff
        b -= d

    return x


def TVdenoise2DOpt(x0, la, mu, iterations=1):
    diff = ForwardDiff2D(x0.space)

    dimension = 2
    L2 = diff.domain
    L2xL2 = diff.range

    x = x0.copy()
    f = x0.copy()
    b = L2xL2.zero()
    d = L2xL2.zero()
    xdiff = L2xL2.zero()
    tmp = L2.zero()

    C1 = mu/(mu+2*la)
    C2 = la/(mu+2*la)
    for i in range(iterations):
        # x = (f * mu + (diff.T(diff(x)) + 2*x + diff.T(d-b)) * la)/(mu+2*la)
        diff.apply(x, xdiff)
        xdiff += d
        xdiff -= b
        diff.adjoint.apply(xdiff, tmp)
        L2.lincomb(C1, f, 2*C2, x)
        x.lincomb(C2, tmp)

        # d = diff(x)+b
        diff.apply(x, d)
        d += b

        for i in range(dimension):
            # tmp = d/abs(d)
            CS.sign(d[i], tmp)

            # d = sign(diff(x)+b) * max(|diff(x)+b|-la^-1,0)
            CS.abs(d[i], d[i])
            CS.addScalar(d[i], -1.0/la, d[i])
            CS.maxVectorScalar(d[i], 0.0, d[i])
            d[i].multiply(tmp)

        # b = b + diff(x) - d
        diff.apply(x, xdiff)
        b += xdiff
        b -= d

    return x


def TVdenoise2D(x0, la, mu, iterations=1):
    diff = ForwardDiff2D(x0.space)

    dimension = 2
    L2 = diff.domain
    L2xL2 = diff.range

    x = x0.copy()
    f = x0.copy()

    b = L2xL2.zero()
    d = L2xL2.zero()
    tmp = L2.zero()

    for i in range(iterations):
        x = (f * mu + (diff.T(diff(x)) + 2*x + diff.T(d-b)) * la)/(mu+2*la)

        d = diff(x)+b

        for i in range(dimension):
            # tmp = d/abs(d)
            CS.sign(d[i], tmp)

            # d = sign(diff(x)+b) * max(|diff(x)+b|-la^-1,0)
            CS.abs(d[i], d[i])
            CS.addScalar(d[i], -1.0/la, d[i])
            CS.maxVectorScalar(d[i], 0.0, d[i])
            d[i].multiply(tmp)

        b = b + diff(x) - d

    return x

# Continuous definition of problem
I = Rectangle([0, 0], [1, 1])
space = L2(I)

# Complicated functions to check performance
n = 2000
m = 2000

# Underlying RN space
rn = CS.CudaRN(n*m)
# Example of using an vector pool to reduce allocation overhead
rnpooled = makePooledSpace(rn, maxPoolSize=5)

# Discretize
d = DS.uniform_discretization(space, rnpooled, (n, m))
x, y = d.points()
data = RLcpp.utils.phantom([n, m])
data[1:-1, 1:-1] += np.random.rand(n-2, m-2) - 0.5
fun = d.element(data)

# Show input
plt.figure()
p = plt.imshow(fun[:].reshape(n, m), interpolation='nearest')
plt.set_cmap('bone')
plt.axis('off')

# Call denoising
la = 0.3
mu = 5.0
with Timer("denoising time"):
    result = TVdenoise2D(fun, la, mu, 100)

# Show result
plt.figure()
p = plt.imshow(result[:].reshape(n, m), interpolation='nearest')
plt.set_cmap('bone')
plt.axis('off')

plt.show()
