"""
============================
Plotting Template Classifier
============================

An example plot of :class:`cosfire.template.TemplateClassifier`
"""
import numpy as np
import cv2
from cosfire.base import (Cosfire,
                          CosfireCircularGaborTuple,
                          GaborParameters,
                          π,
                          )
from matplotlib import pyplot as plt

some_cosfire = Cosfire(strategy_name='Circular Gabor',
                               center_x=100,
                               center_y=100,
                               rho_list=range(0, 100, 10),
                               t1=0.99,
                               t2=0.75,
                               t3=0.9,
                               filter_parameters=GaborParameters(ksize=(10, 10), σ=5,
                                                                 θ=np.linspace(start=0, stop=π, num=30, endpoint=False),
                                                                 λ=np.linspace(start=7, stop=8, num=10, endpoint=False),
                                                                 γ=0.5, ψ=π, ktype=cv2.CV_32F),
                               sigma0=0.67,
                               alpha=0.04,
                               reflection_invariant=0,
                               scale_invariant=[1],
                               rotation_invariant=[0]
                               )

some_cosfire.prototype_image = np.zeros((256, 256))
some_cosfire._prototype_bank = some_cosfire.compute_bank_of_responses(some_cosfire.prototype_image)
some_cosfire.threshold_prototype_bank_of_responses(some_cosfire.threshold_1)
# self.assertAlmostEqual(some_cosfire._maximum_response, 5060.316, places=3)
some_cosfire._Sf = some_cosfire.fit_Sf()
expected = [CosfireCircularGaborTuple(λ=7.5, θ=0.0, ρ=10, φ=1.3089969389957472),
            CosfireCircularGaborTuple(λ=7.5, θ=0.0, ρ=10, φ=2.0420352248333655),
            CosfireCircularGaborTuple(λ=7.5, θ=1.5707963267948966, ρ=10, φ=2.9845130209103035),
            CosfireCircularGaborTuple(λ=7.5, θ=1.5707963267948966, ρ=10, φ=4.171336912266447),
            CosfireCircularGaborTuple(λ=7.5, θ=0.0, ρ=20, φ=1.7976891295541595),
            CosfireCircularGaborTuple(λ=7.5, θ=1.5707963267948966, ρ=20, φ=3.07177948351002),
            CosfireCircularGaborTuple(λ=7.5, θ=1.5707963267948966, ρ=20, φ=3.5779249665883754),
            CosfireCircularGaborTuple(λ=7.5, θ=0.0, ρ=30, φ=1.710422666954443),
            CosfireCircularGaborTuple(λ=7.5, θ=1.5707963267948966, ρ=30, φ=3.420845333908886),
            CosfireCircularGaborTuple(λ=7.5, θ=0.0, ρ=40, φ=1.6755160819145565),
            CosfireCircularGaborTuple(λ=7.5, θ=1.5707963267948966, ρ=40, φ=3.351032163829113)]

plt.imshow(some_cosfire.prototype_image)
plt.show()
