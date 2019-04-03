import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import logging

import cv2
from typing import NamedTuple
from skimage.filters import gabor, gaussian
from skimage import data
from cosfire.base import (Cosfire,
                          )

from cosfire.function_filters import (FunctionFilter,
                                      )

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


class GaborParameters(NamedTuple):
    ksize: tuple
    sigma: float
    theta: float
    lambd: float
    gamma: float
    psi: float = np.pi * 0.5
    ktype: int = cv2.CV_32F


preset_1 = dict(filter_name="Gabor",
                center_coordinate_x=166,
                center_coordinate_y=96,
                rho_list=[0, 12],
                eta=np.pi / 8,
                t1=0.99,
                t2=0.75,
                t3=0.9,
                filter_parameters=GaborParameters(ksize=(10, 10), sigma=5, theta=np.array([0, np.pi / 2]),
                                                  lambd=np.array([12]), gamma=0.5, psi=np.pi),
                sigma0=0.67,
                alpha=0.04,
                reflection_invariant=0,
                scale_invariant=[1],
                rotation_invariant=[0]
                )

preset_2 = dict(filter_name="Gabor",
                center_coordinate_x=166,
                center_coordinate_y=96,
                rho_list=[0, 12],
                eta=np.pi / 8,
                t1=0.99,
                t2=0.75,
                t3=0.9,
                filter_parameters=[(10, 10),
                                   5,
                                   np.array([0, np.pi / 2]),
                                   np.array([12]),
                                   0.5,
                                   np.pi],
                sigma0=0.67,
                alpha=0.04,
                reflection_invariant=True,
                scale_invariant=[1],
                rotation_invariant=[0]
                )

preset_3 = dict(filter_name="Gabor",
                center_coordinate_x=24,
                center_coordinate_y=28,
                rho_list=[0, 2, 4, 7, 10, 13, 16, 20, 25],
                eta=np.pi / 8,
                t1=0.1,
                t2=0.75,
                t3=0.99,
                filter_parameters=[(10, 10),
                                   0.8,
                                   np.array([(i * np.pi) / 8.0 for i in range(16)]),
                                   np.array([4]),
                                   0.5,
                                   np.pi / 2],
                sigma0=0.67,
                alpha=0.04,
                reflection_invariant=False,
                scale_invariant=[1],
                rotation_invariant=[0]
                )

preset_6 = dict(filter_name="Gabor",
                center_coordinate_x=17,
                center_coordinate_y=17,
                rho_list=[0, 3, 8],
                eta=np.pi / 8,
                t1=0.05,
                t2=0.75,
                t3=0.3,
                filter_parameters=[np.array([(i * np.pi) / 8.0 for i in range(16)]),
                                   np.array([2 * np.sqrt(2)]),
                                   (5, 5),
                                   0.8,
                                   0.5,
                                   np.pi / 2],
                sigma0=0.83,
                alpha=0.1,
                reflection_invariant=False,
                scale_invariant=[1],
                rotation_invariant=[0]
                )


class TestFunctionFilter(unittest.TestCase):
    def setUp(self):
        self.image = data.coins()
        self.gabor_filter = FunctionFilter(gabor, 1, 0)
        self.gaussian_filter = FunctionFilter(gaussian, 1)

    def test_function_filter__gabor_fit(self):
        some_filter = self.gabor_filter
        self.assertTrue(some_filter.fit() is some_filter)

    def test_function_filter__gabor_transform(self):
        some_image = self.image
        some_filter = self.gabor_filter
        filter_response = some_filter.transform(some_image)
        self.assertEqual(filter_response[0].shape, some_image.shape)


class TestCosfire(unittest.TestCase):
    def setUp(self):
        self.pattern = 'media/patron.1'

    def test_COSFIRE(self):
        im1 = cv2.imread("media/patron1.jpg", 0)
        a = Cosfire(**preset_1)
        a.fit(im1)
        im2 = cv2.imread("media/prueba1.jpg", 0)
        r = a.transform(im2)
        # viewPattern(r, im2, 20)
        # cv2.imshow("Response", r)
        expected = np.load('media/test_cosfire__preset_1_expected.npy')
        assert_almost_equal(r, expected)
