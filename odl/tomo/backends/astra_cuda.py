# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Backend for ASTRA using CUDA."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
from pkg_resources import parse_version
try:
    import astra
    ASTRA_CUDA_AVAILABLE = astra.astra.use_cuda()
except ImportError:
    ASTRA_CUDA_AVAILABLE = False

from odl.discr import DiscreteLp, DiscreteLpElement
from odl.tomo.backends.astra_setup import (
    ASTRA_VERSION,
    astra_projection_geometry, astra_volume_geometry, astra_projector,
    astra_data, astra_algorithm)
from odl.tomo.geometry import (
    Geometry, Parallel2dGeometry, FanFlatGeometry, Parallel3dAxisGeometry,
    HelicalConeFlatGeometry)


__all__ = ('ASTRA_CUDA_AVAILABLE',
           'AstraCudaProjectorImpl', 'AstraCudaBackProjectorImpl')


class AstraCudaProjectorImpl(object):

    """Thin wrapper around ASTRA."""

    algo_id = None
    vol_id = None
    sino_id = None
    proj_id = None

    def __init__(self, geometry, reco_space, proj_space, use_cache):
        """Run an ASTRA forward projection on the given data using the GPU.

        Parameters
        ----------
        geometry : `Geometry`
            Geometry defining the tomographic setup
        out : ``proj_space`` element, optional
            Element of the projection space to which the result is written. If
            ``None``, an element in ``proj_space`` is created.
        use_cache : bool
            True if data should be cached. Default: True
        """
        if not isinstance(geometry, Geometry):
            raise TypeError('geometry  {!r} is not a Geometry instance'
                            ''.format(geometry))
        self.geometry = geometry

        if not isinstance(reco_space, DiscreteLp):
            raise TypeError('reconstruciton space {!r} is not a DiscreteLp '
                            'instance'.format(proj_space))
        self.reco_space = reco_space

        if not isinstance(proj_space, DiscreteLp):
            raise TypeError('projection space {!r} is not a DiscreteLp '
                            'instance'.format(proj_space))
        self.proj_space = proj_space

        self.use_cache = bool(use_cache)

        if self.use_cache:
            self.create_ids()

    def call_forward(self, vol_data, out=None):
        """Run an ASTRA forward projection on the given data using the GPU.

        Parameters
        ----------
        vol_data : `DiscreteLpElement`
            Volume data to which the projector is applied
        out : ``proj_space`` element, optional
            Element of the projection space to which the result is written. If
            ``None``, an element in ``proj_space`` is created.

        Returns
        -------
        out : ``proj_space`` element
            Projection data resulting from the application of the projector.
            If ``out`` was provided, the returned object is a reference to it.
        """
        if not isinstance(vol_data, DiscreteLpElement):
            raise TypeError('volume data {!r} is not a DiscreteLpElement '
                            'instance'.format(vol_data))
        if vol_data.ndim != self.geometry.ndim:
            raise ValueError('dimensions {} of volume data and {} of geometry '
                             'do not match'
                             ''.format(vol_data.ndim, self.geometry.ndim))
        if out is not None:
            if not isinstance(out, DiscreteLpElement):
                raise TypeError('`out` {} is neither None nor a '
                                'DiscreteLpElement instance'.format(out))

        # Create ids if they don't exist
        if not self.use_cache:
            self.create_ids()

        # Copy data to CUDA
        if self.geometry.ndim == 2:
            astra.data2d.store(self.vol_id, vol_data.asarray())
        elif self.geometry.ndim == 3:
            astra.data3d.store(self.vol_id, vol_data.asarray())

        # Run algorithm
        astra.algorithm.run(self.algo_id)

        # Copy result to host
        if self.geometry.ndim == 2:
            if out is None:
                out = self.proj_space.element()

            out[:] = self.out_array
        elif self.geometry.ndim == 3:
            if out is None:
                out = self.proj_space.element()

            out[:] = np.rollaxis(self.out_array, 0, 3)
        else:
            raise RuntimeError('unknown ndim')

        # Fix inconsistent scaling
        if isinstance(self.geometry, Parallel2dGeometry):
            # parallel2d scales with pixel stride
            out *= 1 / float(self.geometry.det_partition.cell_sides[0])

        if not self.use_cache:
            self.delete_ids()

        return out

    def create_ids(self):
        ndim = self.geometry.ndim

        # Create input and output arrays
        if ndim == 2:
            astra_proj_shape = self.proj_space.shape
        elif ndim == 3:
            astra_proj_shape = (self.proj_space.shape[2],
                                self.proj_space.shape[0],
                                self.proj_space.shape[1])

        self.in_array = np.empty(self.reco_space.shape,
                                 dtype='float32', order='C')
        self.out_array = np.empty(astra_proj_shape,
                                  dtype='float32', order='C')

        # Create ASTRA data structures
        vol_geom = astra_volume_geometry(self.reco_space)
        proj_geom = astra_projection_geometry(self.geometry)
        self.vol_id = astra_data(vol_geom, datatype='volume',
                                 data=self.in_array, allow_copy=False)

        self.proj_id = astra_projector('nearest', vol_geom, proj_geom,
                                       ndim, impl='cuda')

        self.sino_id = astra_data(proj_geom,
                                  datatype='projection',
                                  ndim=self.proj_space.ndim,
                                  data=self.out_array,
                                  allow_copy=False)

        # Create algorithm
        self.algo_id = astra_algorithm(
            'forward', ndim, self.vol_id, self.sino_id,
            proj_id=self.proj_id, impl='cuda')

    def delete_ids(self):
        """Delete ASTRA objects."""
        if self.geometry.ndim == 2:
            adata, aproj = astra.data2d, astra.projector
        else:
            adata, aproj = astra.data3d, astra.projector3d

        if self.algo_id is not None:
            astra.algorithm.delete(self.algo_id)
            self.algo_id = None
        if self.vol_id is not None:
            adata.delete(self.vol_id)
            self.vol_id = None
        if self.sino_id is not None:
            adata.delete(self.sino_id)
            self.sino_id = None
        if self.proj_id is not None:
            aproj.delete(self.proj_id)
            self.proj_id = None

        self.in_array = None
        self.out_array = None

    def __del__(self):
        self.delete_ids()


class AstraCudaBackProjectorImpl(object):

    """Thin wrapper around ASTRA."""

    algo_id = None
    vol_id = None
    sino_id = None
    proj_id = None

    def __init__(self, geometry, reco_space, proj_space, use_cache):
        """Run an ASTRA forward projection on the given data using the GPU.

        Parameters
        ----------
        geometry : `Geometry`
            Geometry defining the tomographic setup
        reco_space : `DiscreteLp`
            Space to which the calling operator maps
        use_cache : bool
            True if data should be cached. Default: True
        """
        if not isinstance(geometry, Geometry):
            raise TypeError('geometry  {!r} is not a Geometry instance'
                            ''.format(geometry))
        self.geometry = geometry

        if not isinstance(reco_space, DiscreteLp):
            raise TypeError('reconstruction space {!r} is not a DiscreteLp '
                            'instance'.format(reco_space))
        self.reco_space = reco_space

        if not isinstance(proj_space, DiscreteLp):
            raise TypeError('projection space {!r} is not a DiscreteLp '
                            'instance'.format(proj_space))
        self.proj_space = proj_space

        self.use_cache = bool(use_cache)

        if self.use_cache:
            self.create_ids()

    def call_backward(self, proj_data, out=None):
        """Run an ASTRA backward projection on the given data using the GPU.

        Parameters
        ----------
        proj_data : `DiscreteLp` element
            Projection data to which the backward projector is applied
        out : ``reco_space`` element, optional
            Element of the reconstruction space to which the result is written.
            If ``None``, an element in ``reco_space`` is created.

        Returns
        -------
        out : ``reco_space`` element
            Reconstruction data resulting from the application of the backward
            projector. If ``out`` was provided, the returned object is a
            reference to it.
        """
        if not isinstance(proj_data, DiscreteLpElement):
            raise TypeError('projection data {!r} is not a DiscreteLpElement '
                            'instance'.format(proj_data))
        if self.reco_space.ndim != self.geometry.ndim:
            raise ValueError('dimensions {} of reconstruction space and {} of '
                             'geometry do not match'.format(
                                 self.reco_space.ndim, self.geometry.ndim))
        if out is not None:
            if not isinstance(out, DiscreteLpElement):
                raise TypeError('`out` {} is neither None nor a '
                                'DiscreteLpElement instance'.format(out))

        if not self.use_cache:
            self.create_ids()

        # Create geometries
        if self.geometry.ndim == 2:
            astra.data2d.store(self.sino_id, proj_data.asarray())
        elif self.geometry.ndim == 3:
            swapped_proj_data = np.ascontiguousarray(
                np.rollaxis(proj_data.asarray(), 2, 0))
            astra.data3d.store(self.sino_id, swapped_proj_data)

        # Run algorithm
        astra.algorithm.run(self.algo_id)

        # Reconstruction volume
        if out is None:
            out = self.reco_space.element()

        out[:] = self.out_array

        out *= astra_cuda_bp_scaling_factor(self.reco_space, self.geometry)

        if not self.use_cache:
            self.delete_ids()

        return out

    def create_ids(self):
        ndim = self.geometry.ndim

        if ndim == 2:
            astra_proj_shape = self.proj_space.shape
        elif ndim == 3:
            astra_proj_shape = (self.proj_space.shape[2],
                                self.proj_space.shape[0],
                                self.proj_space.shape[1])

        self.in_array = np.empty(astra_proj_shape,
                                 dtype='float32', order='C')
        self.out_array = np.empty(self.reco_space.shape,
                                  dtype='float32', order='C')

        # Create geometries
        vol_geom = astra_volume_geometry(self.reco_space)
        proj_geom = astra_projection_geometry(self.geometry)
        self.sino_id = astra_data(proj_geom, datatype='projection',
                                  data=self.in_array,
                                  allow_copy=False)

        # Create projector
        self.proj_id = astra_projector('nearest', vol_geom, proj_geom,
                                       ndim, impl='cuda')

        # Wrap the array in correct dtype etc if needed
        self.vol_id = astra_data(vol_geom, datatype='volume',
                                 data=self.out_array,
                                 ndim=self.reco_space.ndim,
                                 allow_copy=False)

        # Create algorithm
        self.algo_id = astra_algorithm(
            'backward', ndim, self.vol_id, self.sino_id,
            proj_id=self.proj_id, impl='cuda')

    def delete_ids(self):
        """Delete ASTRA objects."""
        if self.geometry.ndim == 2:
            adata, aproj = astra.data2d, astra.projector
        else:
            adata, aproj = astra.data3d, astra.projector3d

        if self.algo_id is not None:
            astra.algorithm.delete(self.algo_id)
            self.algo_id = None
        if self.vol_id is not None:
            adata.delete(self.vol_id)
            self.vol_id = None
        if self.sino_id is not None:
            adata.delete(self.sino_id)
            self.sino_id = None
        if self.proj_id is not None:
            aproj.delete(self.proj_id)
            self.proj_id = None

    def __del__(self):
        self.delete_ids()


def astra_cuda_bp_scaling_factor(reco_space, geometry):
    """Volume scaling accounting for differing adjoint definitions.

    ASTRA defines the adjoint operator in terms of a fully discrete
    setting (transposed "projection matrix") without any relation to
    physical dimensions, which makes a re-scaling necessary to
    translate it to spaces with physical dimensions.

    Behavior of ASTRA changes slightly between versions, so we keep
    track of it and adapt the scaling accordingly.
    """
    # Angular integration weighting factor
    # angle interval weight by approximate cell volume
    angle_extent = float(geometry.motion_partition.extent())
    num_angles = float(geometry.motion_partition.size)
    scaling_factor = angle_extent / num_angles

    if parse_version(ASTRA_VERSION) < parse_version('1.8rc1'):
        # Fix inconsistent scaling
        if isinstance(geometry, Parallel2dGeometry):
            # Scales with 1 / cell_volume
            scaling_factor *= float(reco_space.cell_volume)
        elif isinstance(geometry, FanFlatGeometry):
            # Scales with 1 / cell_volume
            scaling_factor *= float(reco_space.cell_volume)
            # Additional magnification correction
            src_radius = geometry.src_radius
            det_radius = geometry.det_radius
            scaling_factor *= ((src_radius + det_radius) / src_radius)
        elif isinstance(geometry, Parallel3dAxisGeometry):
            # Scales with voxel stride
            # In 1.7, only cubic voxels are supported
            voxel_stride = reco_space.cell_sides[0]
            scaling_factor /= float(voxel_stride)
        elif isinstance(geometry, HelicalConeFlatGeometry):
            # Scales with 1 / cell_volume
            # In 1.7, only cubic voxels are supported
            voxel_stride = reco_space.cell_sides[0]
            scaling_factor /= float(voxel_stride)
            # Magnification correction
            src_radius = geometry.src_radius
            det_radius = geometry.det_radius
            scaling_factor *= ((src_radius + det_radius) / src_radius) ** 2

    else:
        if isinstance(geometry, Parallel2dGeometry):
            # Scales with 1 / cell_volume
            scaling_factor *= float(reco_space.cell_volume)
        elif isinstance(geometry, FanFlatGeometry):
            # Scales with 1 / cell_volume
            scaling_factor *= float(reco_space.cell_volume)
            # Magnification correction
            src_radius = geometry.src_radius
            det_radius = geometry.det_radius
            scaling_factor *= ((src_radius + det_radius) / src_radius)
        elif isinstance(geometry, Parallel3dAxisGeometry):
            # Scales with cell volume
            # currently only square voxels are supported
            scaling_factor /= reco_space.cell_volume
        elif isinstance(geometry, HelicalConeFlatGeometry):
            # Scales with cell volume
            scaling_factor /= reco_space.cell_volume
            # Magnification correction
            src_radius = geometry.src_radius
            det_radius = geometry.det_radius
            scaling_factor *= ((src_radius + det_radius) / src_radius) ** 2

            # Correction for scaled 1/r^2 factor in ASTRA's density weighting
            det_px_area = geometry.det_partition.cell_volume
            scaling_factor *= (src_radius ** 2 * det_px_area ** 2 /
                               reco_space.cell_volume ** 2)

        # TODO: add case with new ASTRA release

    return scaling_factor


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
