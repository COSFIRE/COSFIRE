
import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import logging

from skimage.filters import gabor, gaussian
from skimage import data

from cosfire.base import (FunctionFilter,
                          )

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


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
