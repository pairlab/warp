from typing import Any
from enum import Enum
import math

import warp as wp
import warp.types

from warp.fem.types import vec6

_SQRT_2 = wp.constant(math.sqrt(2.0))
_SQRT_3 = wp.constant(math.sqrt(3.0))
_SQRT_1_2 = wp.constant(math.sqrt(1.0 / 2.0))
_SQRT_1_3 = wp.constant(math.sqrt(1.0 / 3.0))


class DofMapper:
    """Base class from mapping node degrees of freedom to function values"""

    value_dtype: type
    dof_dtype: type
    DOF_SIZE: int

    @wp.func
    def dof_to_value(dof: Any):
        raise NotImplementedError

    @wp.func
    def value_to_dof(val: Any):
        raise NotImplementedError

    def __str__(self):
        return f"{self.value_dtype.__name__}_{self.DOF_SIZE}"


class IdentityMapper(DofMapper):
    """Identity mapper"""

    def __init__(self, dtype: type):
        if dtype == float:
            dtype = wp.float32

        self.value_dtype = dtype
        self.dof_dtype = dtype

        size = warp.types.type_length(dtype)
        self.DOF_SIZE = wp.constant(size)

    @wp.func
    def dof_to_value(dof: Any):
        return dof

    @wp.func
    def value_to_dof(val: Any):
        return val


class SymmetricTensorMapper(DofMapper):
    """Orthonormal isomorphism from R^{n (n+1)} to nxn symmetric tensors,
    using usual L2 norm for vectors and half Frobenius norm, (tau : tau)/2 for tensors.
    """

    class Mapping(Enum):
        VOIGT = 0
        """Voigt ordering of vector coefficients:
            first the three diagonal terms, then off-diagonal coefficients"""
        DB16 = 1
        """Ordering that also separates normal from tangential coefficients:
            first trace, then other diagonal terms, then off-diagonal coefficients.
           See [Daviet and Bertails-Descoubes 2016]"""

    def __init__(self, dtype: type, mapping: Mapping = Mapping.VOIGT):
        self.value_dtype = dtype
        self.mapping = mapping

        if dtype == wp.mat22:
            self.dof_dtype = wp.vec3
            self.DOF_SIZE = wp.constant(3)
            if mapping == SymmetricTensorMapper.Mapping.VOIGT:
                self.dof_to_value = SymmetricTensorMapper.dof_to_value_2d_voigt
                self.value_to_dof = SymmetricTensorMapper.value_to_dof_2d_voigt
            else:
                self.dof_to_value = SymmetricTensorMapper.dof_to_value_2d
                self.value_to_dof = SymmetricTensorMapper.value_to_dof_2d
        elif dtype == wp.mat33:
            self.dof_dtype = vec6
            self.DOF_SIZE = wp.constant(6)
            if mapping == SymmetricTensorMapper.Mapping.VOIGT:
                self.dof_to_value = SymmetricTensorMapper.dof_to_value_3d_voigt
                self.value_to_dof = SymmetricTensorMapper.value_to_dof_3d_voigt
            else:
                self.dof_to_value = SymmetricTensorMapper.dof_to_value_3d
                self.value_to_dof = SymmetricTensorMapper.value_to_dof_3d
        else:
            raise ValueError("Unsupported value dtype: ", dtype)

    def __str__(self):
        return f"{self.mapping}_{self.DOF_SIZE}"

    @wp.func
    def dof_to_value_2d(dof: wp.vec3):
        a = dof[0]
        b = dof[1]
        c = dof[2]
        return wp.mat22(a + b, c, c, a - b)

    @wp.func
    def value_to_dof_2d(val: wp.mat22):
        a = 0.5 * (val[0, 0] + val[1, 1])
        b = 0.5 * (val[0, 0] - val[1, 1])
        c = 0.5 * (val[0, 1] + val[1, 0])
        return wp.vec3(a, b, c)

    @wp.func
    def dof_to_value_2d_voigt(dof: wp.vec3):
        a = _SQRT_2 * dof[0]
        b = _SQRT_2 * dof[1]
        c = dof[2]
        return wp.mat22(a, c, c, b)

    @wp.func
    def value_to_dof_2d_voigt(val: wp.mat22):
        a = _SQRT_1_2 * val[0, 0]
        b = _SQRT_1_2 * val[1, 1]
        c = 0.5 * (val[0, 1] + val[1, 0])
        return wp.vec3(a, b, c)

    @wp.func
    def dof_to_value_3d(dof: vec6):
        a = dof[0] * _SQRT_2 * _SQRT_1_3
        b = dof[1]
        c = dof[2] * _SQRT_1_3
        d = dof[3]
        e = dof[4]
        f = dof[5]
        return wp.mat33(
            a + b - c,
            f,
            e,
            f,
            a - b - c,
            d,
            e,
            d,
            a + 2.0 * c,
        )

    @wp.func
    def value_to_dof_3d(val: wp.mat33):
        a = (val[0, 0] + val[1, 1] + val[2, 2]) * _SQRT_1_3 * _SQRT_1_2
        b = 0.5 * (val[0, 0] - val[1, 1])
        c = 0.5 * (val[2, 2] - (val[0, 0] + val[1, 1] + val[2, 2]) / 3.0) * _SQRT_3

        d = 0.5 * (val[2, 1] + val[1, 2])
        e = 0.5 * (val[0, 2] + val[2, 0])
        f = 0.5 * (val[1, 0] + val[0, 1])

        return vec6(a, b, c, d, e, f)

    @wp.func
    def dof_to_value_3d_voigt(dof: vec6):
        a = _SQRT_2 * dof[0]
        b = _SQRT_2 * dof[1]
        c = _SQRT_2 * dof[2]
        d = dof[3]
        e = dof[4]
        f = dof[5]
        return wp.mat33(
            a,
            f,
            e,
            f,
            b,
            d,
            e,
            d,
            c,
        )

    @wp.func
    def value_to_dof_3d_voigt(val: wp.mat33):
        a = _SQRT_1_2 * val[0, 0]
        b = _SQRT_1_2 * val[1, 1]
        c = _SQRT_1_2 * val[2, 2]

        d = 0.5 * (val[2, 1] + val[1, 2])
        e = 0.5 * (val[0, 2] + val[2, 0])
        f = 0.5 * (val[1, 0] + val[0, 1])

        return vec6(a, b, c, d, e, f)
