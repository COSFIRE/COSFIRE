import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import logging

import cv2
from skimage.filters import gabor, gaussian
from skimage import data
from cosfire.base import (Cosfire,
                          CosfireCircularGaborTuple,
                          GaborParameters,
                          π,
                          )

from cosfire.function_filters import (FunctionFilter,
                                      )

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)

preset_1 = dict(filter_name="Gabor",
                center_x=166,
                center_y=96,
                rho_list=[0, 12],
                eta=np.pi / 8,
                t1=0.99,
                t2=0.75,
                t3=0.9,
                filter_parameters=GaborParameters(ksize=(10, 10),
                                                  σ=5,
                                                  θ=np.array([0, π / 2]),
                                                  λ=np.array([12]),
                                                  γ=0.5,
                                                  ψ=π,
                                                  ktype=cv2.CV_32F),
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
                eta=π / 8,
                t1=0.99,
                t2=0.75,
                t3=0.9,
                filter_parameters=[(10, 10),
                                   5,
                                   np.array([0, π / 2]),
                                   np.array([12]),
                                   0.5,
                                   π],
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
                eta=π / 8,
                t1=0.1,
                t2=0.75,
                t3=0.99,
                filter_parameters=[(10, 10),
                                   0.8,
                                   np.array([(i * π) / 8.0 for i in range(16)]),
                                   np.array([4]),
                                   0.5,
                                   π / 2],
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
                eta=π / 8,
                t1=0.05,
                t2=0.75,
                t3=0.3,
                filter_parameters=[np.array([(i * π) / 8.0 for i in range(16)]),
                                   np.array([2 * np.sqrt(2)]),
                                   (5, 5),
                                   0.8,
                                   0.5,
                                   π / 2],
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

    def test_cosfire__preset_1(self):
        im1 = cv2.imread("media/patron1.jpg", 0)
        im2 = cv2.imread("media/prueba1.jpg", 0)
        for iter in range(1):
            a = Cosfire(**preset_1)
            a.fit(im1)
            r = a.transform(im2)
        # viewPattern(r, im2, 20)
        # cv2.imshow("Response", r)
        expected = np.load('media/test_cosfire__preset_1_expected.npy')
        assert_almost_equal(r, expected)

    def test_shift_responses(self):
        specific_image = np.array([[3, 3, 3, 3, 3],
                                   [3, 2, 2, 2, 3],
                                   [3, 2, 1, 2, 3],
                                   [3, 2, 2, 2, 3],
                                   [3, 3, 3, 3, 3]], dtype=np.float32)
        tupla_collection = [CosfireCircularGaborTuple(ρ=1, ϕ=0, λ=0.1, θ=0),
                            CosfireCircularGaborTuple(ρ=1, ϕ=π * 0.5, λ=0.1, θ=0),
                            CosfireCircularGaborTuple(ρ=1, ϕ=π, λ=0.1, θ=0),
                            CosfireCircularGaborTuple(ρ=1, ϕ=π * 1.5, λ=0.1, θ=0),
                            ]
        response_collection = {tupla: specific_image for tupla in tupla_collection}
        cosfire_filter = Cosfire(**preset_1)
        result_collection = {key: cosfire_filter.shift_responses({key: response})[key]
                             for key, response in response_collection.items()}
        expected_collection = {
            CosfireCircularGaborTuple(ρ=1, ϕ=0, λ=0.1, θ=0): np.array([[3, 3, 3, 3, 0],
                                                                       [2, 2, 2, 3, 0],
                                                                       [2, 1, 2, 3, 0],
                                                                       [2, 2, 2, 3, 0],
                                                                       [3, 3, 3, 3, 0]], dtype=np.float32),
            CosfireCircularGaborTuple(ρ=1, ϕ=π * 0.5, λ=0.1, θ=0): np.array([[0., 0., 0., 0., 0.],
                                                                             [3., 3., 3., 3., 3.],
                                                                             [3., 2., 2., 2., 3.],
                                                                             [3., 2., 1., 2., 3.],
                                                                             [3., 2., 2., 2., 3.]],
                                                                            dtype=np.float32),
            CosfireCircularGaborTuple(ρ=1, ϕ=π, λ=0.1, θ=0): np.array([[0., 3., 3., 3., 3.],
                                                                       [0., 3., 2., 2., 2.],
                                                                       [0., 3., 2., 1., 2.],
                                                                       [0., 3., 2., 2., 2.],
                                                                       [0., 3., 3., 3., 3.]], dtype=np.float32),
            CosfireCircularGaborTuple(ρ=1, ϕ=π * 1.5, λ=0.1, θ=0): np.array([[3., 2., 2., 2., 3.],
                                                                             [3., 2., 1., 2., 3.],
                                                                             [3., 2., 2., 2., 3.],
                                                                             [3., 3., 3., 3., 3.],
                                                                             [0., 0., 0., 0., 0.]],
                                                                            dtype=np.float32), }

        assert_almost_equal(result_collection[CosfireCircularGaborTuple(ρ=1, ϕ=π, λ=0.1, θ=0)],
                            expected_collection[CosfireCircularGaborTuple(ρ=1, ϕ=π, λ=0.1, θ=0)])

        assert_almost_equal(result_collection[CosfireCircularGaborTuple(ρ=1, ϕ=0, λ=0.1, θ=0)],
                            expected_collection[CosfireCircularGaborTuple(ρ=1, ϕ=0, λ=0.1, θ=0)])

        assert_almost_equal(result_collection[CosfireCircularGaborTuple(ρ=1, ϕ=π * 0.5, λ=0.1, θ=0)],
                            expected_collection[CosfireCircularGaborTuple(ρ=1, ϕ=π * 0.5, λ=0.1, θ=0)])

        assert_almost_equal(result_collection[CosfireCircularGaborTuple(ρ=1, ϕ=π * 1.5, λ=0.1, θ=0)],
                            expected_collection[CosfireCircularGaborTuple(ρ=1, ϕ=π * 1.5, λ=0.1, θ=0)])

        # for key, result in result_collection.items():
        #     assert_almost_equal(result, expected_collection[key])


class TestCosfireCircularGabor(unittest.TestCase):
    def setUp(self):
        self.pattern = np.zeros((256, 256))
        cv2.rectangle(img=self.pattern, pt1=(50, 100), pt2=(100, 106), color=255, thickness=-1)
        cv2.rectangle(img=self.pattern, pt1=(100, 100), pt2=(97, 50), color=255, thickness=-1)

    def test_cosfire_process(self):
        some_cosfire = Cosfire(filter_name='Gabor',
                               search_strategy='Circular',
                               center_x=100,
                               center_y=100,
                               rho_list=range(100),
                               t1=0.99,
                               t2=0.75,
                               t3=0.9,
                               filter_parameters=GaborParameters(ksize=(10, 10),
                                                                 σ=5,
                                                                 θ=np.linspace(start=0,
                                                                               stop=π,
                                                                               num=180,
                                                                               endpoint=False),
                                                                 λ=np.linspace(start=0.01,
                                                                               stop=15,
                                                                               num=100,
                                                                               endpoint=False),
                                                                 γ=0.5,
                                                                 ψ=π,
                                                                 ktype=cv2.CV_32F),
                               sigma0=0.67,
                               alpha=0.04,
                               reflection_invariant=0,
                               scale_invariant=[1],
                               rotation_invariant=[0]
                               )
        some_cosfire.prototype_image = self.pattern
        some_cosfire._prototype_responses = some_cosfire.compute_response_to_filters(some_cosfire.prototype_image)
        some_cosfire.suppress_responses_threshold_1()
        self.assertAlmostEqual(some_cosfire._maximum_response, 5060.316, places=3)
        some_cosfire._cosfire_tuples = some_cosfire.get_cosfire_tuples()
        expected =  [CosfireCircularGaborTuple(λ=0.7595000000000001, θ=0.8552113334772214, ρ=5, φ=2.6179938779914944), CosfireCircularGaborTuple(λ=0.7595000000000001, θ=0.8552113334772214, ρ=6, φ=2.478367537831948), CosfireCircularGaborTuple(λ=0.7595000000000001, θ=0.7155849933176751, ρ=6, φ=2.8797932657906435), CosfireCircularGaborTuple(λ=0.7595000000000001, θ=0.8552113334772214, ρ=7, φ=2.356194490192345), CosfireCircularGaborTuple(λ=0.7595000000000001, θ=0.7155849933176751, ρ=7, φ=2.91469985083053), CosfireCircularGaborTuple(λ=0.7595000000000001, θ=0.7155849933176751, ρ=8, φ=2.949606435870417)]
        self.assertEqual(some_cosfire._cosfire_tuples, expected)
