#   Keypoint detection by a COSFIRE filter
#   Refactored from Daniel Acosta 2017 code
#
#   COSFIRE is an algorithm created by George Azzopardi and Nicolai Petkov,
#   for more details view original paper:
#   George Azzopardi and Nicolai Petkov, "Trainable COSFIRE filters for
#   keypoint detection and pattern recognition", IEEE Transactions on Pattern 
#   Analysis and Machine Intelligence, vol. 35(2), pp. 490-503, 2013.

import numpy as np
import cv2
import peakutils
from typing import NamedTuple
import itertools
from math import floor, cos, sin

π = np.pi


class Pixel(NamedTuple):
    row: int
    column: int

    def is_inside(self, image):
        rows, cols = image.shape
        is_pixel_inside_image = (0 <= self.row < rows ) and (0 <= self.column < cols)
        return is_pixel_inside_image

    def maximum_response(self, bank):
        maximum_response = 0
        for image in bank.values():
            some_response_map = image
            break  # I just want one image to check the size

        if self.is_inside(some_response_map):
            maximum_response = max([response[self] for _, response in bank.items()])
        return maximum_response


class GaborParameters(NamedTuple):
    ksize: tuple
    σ: float
    θ: float
    λ: float
    γ: float
    ψ: float = π * 0.5
    ktype: int = cv2.CV_32F


class GaborKey(NamedTuple):
    θ: float
    λ: float


class CosfireCircularGaborTuple(NamedTuple):
    λ: float
    θ: float
    ρ: float
    ϕ: float


class CosfireCircularDoGTuple(NamedTuple):
    ρ: float
    ϕ: float
    σ: float


class Cosfire:
    def __init__(self,
                 filter_name='Gabor',
                 search_strategy='Circular',
                 center_x=0,
                 center_y=0,
                 rho_list=None,
                 eta=0,
                 t1=0,
                 filter_parameters=None,
                 t2=0,
                 alpha=0,
                 sigma0=0,
                 t3=0,
                 reflection_invariant=False,
                 rotation_invariant=None,
                 scale_invariant=None,
                 ddepth=cv2.CV_32F):
        self.filter_name = filter_name
        self.search_strategy = search_strategy
        self.center_x = center_x
        self.center_y = center_y
        self.ρ_list = [] if rho_list is None else rho_list
        self.η = eta  # TODO: unused!!!
        self.threshold_1 = t1
        self.filter_parameters = [] if filter_parameters is None else filter_parameters  # Parameters of filter
        self.threshold_2 = t2
        self.alpha = alpha
        self.σ0 = sigma0
        self.threshold_3 = t3
        self.reflection_invariant = reflection_invariant
        self.scale_invariant = [] if scale_invariant is None else scale_invariant
        self.rotation_invariant = [] if rotation_invariant is None else rotation_invariant
        self.ddepth = ddepth

        strategy_key = (search_strategy, filter_name)
        self._compute_response_to_filters = compute_responses_to_filters_dictionary[strategy_key]
        self._i_scale_cosfire = i_scale_cosfire_dictionary[strategy_key]
        self._i_rotation_cosfire = i_rotation_cosfire_dictionary[strategy_key]
        self._i_reflection_cosfire = i_reflection_cosfire_dictionary[strategy_key]
        self._blur_gaussian = blur_gaussian_dictionary[strategy_key]
        self._compute_tuples = compute_tuples_dictionary[strategy_key]
        self._get_cosfire_tuples = get_cosfire_tuples_dictionary[strategy_key]

        self._responses_to_image = {}
        self._cosfire_tuples = []  # Struct of filter COSFIRE (S_f)
        self._prototype_responses = {}  # Bank of responses pattern Image
        self._maximum_response = 0
        self._cosfire_tuples_invariant = []  # operator invariant to rotation, escala and reflection

    # 1-Configuration COSFIRE Filter
    def fit(self, X, **kwargs):
        self.prototype_image = X
        self._prototype_responses = self.compute_response_to_filters(self.prototype_image)  # 1.1
        self.suppress_responses_threshold_1()  # 1.2
        self._cosfire_tuples = self.get_cosfire_tuples()  # 1.3

    # 2-Apply the COSFIRE filtler
    def transform(self, X, **kwargs):
        input_image = X
        self._responses_to_image = self.compute_tuples(input_image)  # 2.1
        self._responses_to_image = self.shift_responses(self._responses_to_image)  # 2.2
        output = self.i_reflection_cosfire(input_image, self._cosfire_tuples, self._responses_to_image)  # 2.3
        maximum_output = output.max()
        cv2.threshold(src=output, dst=output, thresh=self.threshold_3 * maximum_output, maxval=maximum_output,
                      type=cv2.THRESH_TOZERO)
        return output

    # (1.1) Get response filter
    def compute_response_to_filters(self, image, **kwargs):
        return self._compute_response_to_filters(self, image, **kwargs)

    # (1.2) Suppres Resposes
    def suppress_responses_threshold_1(self, **kwargs):
        self._maximum_response = max([value.max() for key, value in self._prototype_responses.items()])
        [cv2.threshold(src=image, dst=image, thresh=self.threshold_1 * self._maximum_response,
                       maxval=self._maximum_response, type=cv2.THRESH_TOZERO)
         for key, image in self._prototype_responses.items()]  # Desired collateral effect: modify self.responses_map
        return

    # (1.3) Get descriptor set (Sf)
    def get_cosfire_tuples(self, **kwargs):
        return self._get_cosfire_tuples(self, **kwargs)

    # 2.1 For each tuple in the set Sf compute response
    def compute_tuples(self, inputImage, **kwargs):
        return self._compute_tuples(self, inputImage, **kwargs)

    # (2.1.1)Blurr
    def blur_gaussian(self, bankFilters, **kwargs):
        return self._blur_gaussian(self, bankFilters, **kwargs)

    # (2.2) Shift
    def shift_responses(self, resp, **kwargs):
        response_maps = {}
        for tupla, response in resp.items():
            rows, cols = response.shape
            x = -tupla.ρ * cos(tupla.ϕ)
            y = tupla.ρ * sin(tupla.ϕ)
            M = np.float32([[1, 0, x],
                            [0, 1, y]])
            dst = cv2.warpAffine(response, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            response_maps[tupla] = dst
        return response_maps

    # (2.3) invariant under reflection
    def i_reflection_cosfire(self, image, operator, response_bank, **kwargs):
        return self._i_reflection_cosfire(self, image, operator, response_bank, **kwargs)

    # (2.4) invariant to rotation
    def i_rotation_cosfire(self, image, operator, response_bank, **kwargs):
        return self._i_rotation_cosfire(self, image, operator, response_bank, **kwargs)

    # (2.5) invariant to scale
    def i_scale_cosfire(self, image, operator, response_bank, **kwargs):
        return self._i_scale_cosfire(self, image, operator, response_bank, **kwargs)

    # Compute response
    def compute_response(self, conj, operator, **kwargs):
        rows, cols = conj[operator[0]].shape
        resp = np.ones((rows, cols))
        ρ_maximum = max([tupla.ρ for tupla in operator])
        tuple_weight_σ = np.sqrt(-((ρ_maximum ** 2) / (2 * np.log(0.5))))

        sum_of_scalar_power = 0
        for tupla in operator:
            scalar_power = np.exp(-(tupla.ρ ** 2) / (2 * tuple_weight_σ ** 2))
            sum_of_scalar_power += scalar_power
            resp *= np.float_power(conj[tupla], scalar_power)
        resp = np.power(resp, 1.0 / sum_of_scalar_power)
        return resp


def compute_response_to_filters__Circular_Gabor(self, image, **kwargs):
    ksize, σ, θ_array, λ_array, γ, ψ, ktype = self.filter_parameters
    if 'θ_array' in kwargs:
        θ_array = kwargs['θ_array']
    if 'λ_array' in kwargs:
        λ_array = kwargs['λ_array']

    response_dict = {
        GaborKey(θ=θ, λ=λ): cv2.filter2D(src=image, ddepth=self.ddepth,
                                         kernel=cv2.getGaborKernel(ksize=ksize, sigma=σ, theta=θ, lambd=λ, gamma=γ,
                                                                   psi=ψ, ktype=ktype))
        for θ, λ in itertools.product(θ_array, λ_array)}
    # TODO: DANIEL ACOSTA COMMENT: Faltaria normalizar cada respuesta de Gabor
    return response_dict


def compute_response_to_filters__Circular_DoG(self, image, **kwargs):
    response = {}
    for σ in self.filter_parameters.σ:
        g1 = cv2.GaussianBlur(image, (3, 3), σ)
        g2 = cv2.GaussianBlur(image, (3, 3), 0.5 * σ)
        r = g1 - g2
        response[σ] = r
    return response


def i_scale_cosfire__Circular_Gabor(self, image, operator, response_bank, **kwargs):
    maximum_output = np.zeros_like(image)
    for scale in self.scale_invariant:
        operatorEscala = [
            CosfireCircularGaborTuple(λ=tupla.λ * scale, θ=tupla.θ, ρ=tupla.ρ * scale, ϕ=tupla.ϕ) for tupla in operator]
        output = self.compute_response(response_bank, operatorEscala, **kwargs)
        maximum_output = np.maximum(output, maximum_output)
    return maximum_output


def i_scale_cosfire__Circular_DoG(self, image, operator, respBank, **kwargs):
    maximum_output = np.zeros_like(image)
    for scale in self.scale_invariant:
        operatorEscala = [CosfireCircularDoGTuple(ρ=tupla.ρ * scale, ϕ=tupla.ϕ, σ=tupla.σ * scale)
                          for tupla in operator]
        output = self.compute_response(respBank, operatorEscala, **kwargs)
        maximum_output = np.maximum(output, maximum_output)
    return maximum_output


def i_rotation_cosfire__Circular_Gabor(self, image, operator, response_bank, **kwargs):
    maximum_output = np.zeros_like(image)
    for rotation in self.rotation_invariant:
        operatorRotacion = [CosfireCircularGaborTuple(λ=tupla.λ, θ=tupla.θ + rotation, ρ=tupla.ρ, ϕ=tupla.ϕ + rotation)
                            for tupla in operator]
        output = self.i_scale_cosfire(image, operatorRotacion, response_bank, **kwargs)
        maximum_output = np.maximum(output, maximum_output)
    return maximum_output


def i_rotation_cosfire__Circular_DoG(self, image, operator, response_bank, **kwargs):
    maximum_output = np.zeros_like(image)
    for rotation in self.rotation_invariant:
        operatorRotacion = [CosfireCircularDoGTuple(ρ=tupla.ρ, ϕ=tupla.ϕ + rotation, σ=tupla.σ) for tupla in operator]
        output = self.i_scale_cosfire(image, operatorRotacion, response_bank, **kwargs)
        maximum_output = np.maximum(output, maximum_output)
    return maximum_output


def i_reflection_cosfire__Circular_Gabor(self, image, operator, response_bank, **kwargs):
    maximum_output = self.i_rotation_cosfire(image, operator, response_bank, **kwargs)
    if self.reflection_invariant == 1:
        operatorI = [CosfireCircularGaborTuple(λ=tupla.λ, θ=π - tupla.θ, ρ=tupla.ρ, ϕ=π - tupla.ϕ)
                     for tupla in operator]
        output = self.i_rotation_cosfire(image, operatorI, response_bank, **kwargs)
        maximum_output = np.maximum(output, maximum_output)
    return maximum_output


def i_reflection_cosfire__Circular_DoG(self, image, operator, response_bank, **kwargs):  # pure imagination
    return self.i_rotation_cosfire(image, operator, response_bank, **kwargs)


def blur_gaussian__Circular_Gabor(self, bankFilters, **kwargs):
    dic = {}
    for tupla in self._cosfire_tuples_invariant:
        σ = self.alpha * tupla.ρ + self.σ0
        if not tupla in dic:
            dic[tupla] = cv2.GaussianBlur(bankFilters[(tupla.θ, tupla.λ)], (9, 9), σ, σ)
    return dic


def blur_gaussian__Circular_DoG(self, bankFilters, **kwargs):
    dic = {}
    for tupla in self._cosfire_tuples_invariant:
        σ = self.alpha * tupla.ρ + self.σ0
        dic[tupla] = cv2.GaussianBlur(bankFilters[tupla.σ], (9, 9), σ, σ)
    return dic


def compute_tuples__Circular_Gabor(self, inputImage, **kwargs):
    # Aqui llamar funcion que rellena parametros para invarianzas
    operator = self._cosfire_tuples[:]
    if self.reflection_invariant == 1:
        operator += [CosfireCircularGaborTuple(λ=tupla.λ, θ=π - tupla.θ, ρ=tupla.ρ, ϕ=π - tupla.ϕ)
                     for tupla in self._cosfire_tuples]

    for tupla, rotation, scale in itertools.product(operator, self.rotation_invariant, self.scale_invariant):
        new_tupla = CosfireCircularGaborTuple(λ=tupla.λ * scale,
                                              θ=tupla.θ + rotation,
                                              ρ=tupla.ρ * scale,
                                              ϕ=tupla.ϕ + rotation)
        self._cosfire_tuples_invariant.append(new_tupla)
    unicos = {}
    for tupla in self._cosfire_tuples_invariant:
        gabor_key = GaborKey(θ=tupla.θ, λ=tupla.λ)
        if not gabor_key in unicos:
            θ_array = np.array([gabor_key.θ])
            λ_array = np.array([gabor_key.λ])
            tt = self.compute_response_to_filters(inputImage, θ_array=θ_array, λ_array=λ_array, **kwargs)
            unicos[gabor_key] = tt[gabor_key]
    maximum = max([value.max() for key, value in unicos.items()])
    [cv2.threshold(src=image, dst=image, thresh=self.threshold_1 * maximum, maxval=maximum, type=cv2.THRESH_TOZERO)
     for key, image in unicos.items()]
    unicos = self.blur_gaussian(unicos)  ##2.1.1 Hacemos Blur
    return unicos


def compute_tuples__Circular_DoG(self, inputImage, **kwargs):
    # Aqui llamar funcion que rellena parametros para invarianzas
    operator = self._cosfire_tuples[:]
    if self.reflection_invariant == 1:
        operator += [CosfireCircularGaborTuple(λ=tupla.λ, θ=π - tupla.θ, ρ=tupla.ρ, ϕ=π - tupla.ϕ)
                     for tupla in self._cosfire_tuples]

    for tupla, rotation, scale in itertools.product(operator, self.rotation_invariant, self.scale_invariant):
        new_tupla = CosfireCircularDoGTuple(ρ=tupla.ρ * scale, ϕ=tupla.ϕ + rotation, σ=tupla.σ * scale)
        self._cosfire_tuples_invariant.append(new_tupla)
    unicos = {}
    for tupla in self._cosfire_tuples_invariant:
        if not tupla[2] in unicos:
            l1 = np.array(1 * [tupla[2]])
            tt = self.compute_response_to_filters(inputImage, self.filter_parameters, l1)
            unicos[tupla[2]] = tt[tupla[2]]
    maximum = max([value.max() for key, value in unicos.items()])
    [cv2.threshold(src=image, dst=image, thresh=self.threshold_1 * maximum,
                   maxval=maximum, type=cv2.THRESH_TOZERO)
     for key, image in unicos.items()]
    unicos = self.blur_gaussian(unicos)  ##2.1.1 Hacemos Blur
    return unicos


def get_cosfire_tuples__Circular_Gabor(self, **kwargs):
    operator = []
    ϕ_array = np.linspace(start=0, stop=2 * π, num=360, endpoint=False)
    for ρ in self.ρ_list:
        if ρ == 0:
            for θ, λ in itertools.product(self.filter_parameters.θ, self.filter_parameters.λ):
                response_at_ρ_ϕ = self._prototype_responses[GaborKey(θ=θ, λ=λ)][self.center_x][self.center_y]
                if response_at_ρ_ϕ > self._maximum_response * self.threshold_2:
                    operator.append(CosfireCircularGaborTuple(λ=λ, θ=θ, ρ=0, ϕ=0))
        elif ρ > 0:
            pixel_at_ρ_ϕ_list = [Pixel(row=floor(self.center_x - ρ * sin(ϕ)), column=floor(self.center_y + ρ * cos(ϕ)))
                                 for ϕ in ϕ_array]
            maximum_responses_array = np.array([pixel_at_ρ_ϕ.maximum_response(bank=self._prototype_responses)
                                                for pixel_at_ρ_ϕ in pixel_at_ρ_ϕ_list])
            if len(np.unique(maximum_responses_array)) == 1:
                continue
            maximum_responses_array_augmented = np.zeros(maximum_responses_array.size + 1)
            maximum_responses_array_augmented[0] = maximum_responses_array[-1]
            maximum_responses_array_augmented[1:] = maximum_responses_array.copy()
            response_peaks_index = - 1 + peakutils.peak.indexes(maximum_responses_array_augmented, thres=0.2,
                                                                min_dist=22)  # 22=360/16
            for peak_index in response_peaks_index:
                for θ in self.filter_parameters.θ:
                    maximum_response = 0
                    for λ in self.filter_parameters.λ:
                        peak_pixel = pixel_at_ρ_ϕ_list[peak_index]
                        response_at_peak = self._prototype_responses[GaborKey(θ=θ, λ=λ)][peak_pixel]
                        if (response_at_peak > self.threshold_2 * self._maximum_response) and (
                                response_at_peak > maximum_response):
                            maximum_response = response_at_peak
                            λ_maximum = λ
                    if maximum_response > 0:
                        operator.append(CosfireCircularGaborTuple(λ=λ_maximum, θ=θ, ρ=ρ, ϕ=peak_index * (π / 180.0)))

    return operator


def get_cosfire_tuples__Circular_DoG(self, **kwargs):
    operator = []
    ϕ_array = np.linspace(start=0, stop=2 * π, num=360, endpoint=False)
    for ρ in self.ρ_list:
        if ρ == 0:
            for σ in self.filter_parameters.σ:
                response_at_center = self._prototype_responses[σ][self.center_x][self.center_y]
                if response_at_center > self._maximum_response * self.threshold_2:
                    operator.append(CosfireCircularDoGTuple(ρ=0, ϕ=0, σ=σ))
        elif ρ > 0:
            listMax = np.zeros(360)
            direcciones = []
            for k, ϕ in enumerate(ϕ_array):
                yi = floor(self.center_y + ρ * cos(ϕ))
                xi = floor(self.center_x - ρ * sin(ϕ))
                response_at_center = 0
                rows, cols = self.prototype_image.shape
                if xi >= 0 and yi >= 0 and xi < rows and yi < cols:
                    for gabor_key, response in self.prototype_response_to_filters.items():
                        if response[xi][yi] > response_at_center:
                            response_at_center = response[xi][yi]
                listMax[k] = response_at_center
                direcciones.append((xi, yi))
            if len(np.unique(listMax)) == 1:
                continue
            listMax1 = np.zeros(listMax.size + 1)
            listMax1[1:] = listMax.copy()
            index = peakutils.peak.indexes(listMax1, thres=0.2, min_dist=22)  # 22=360/16
            index = list(index - 1)
            index = np.array(index)
            for k in range(index.size):
                for σ, response in self._prototype_responses.items():
                    response_at_bearing = response[direcciones[index[k]][0]][direcciones[index[k]][1]]
                    if response_at_bearing > self.threshold_2 * self._maximum_response:
                        operator.append(CosfireCircularDoGTuple(ρ=ρ, ϕ=index[k] * (π / 180.0), σ=σ))
    return operator


get_cosfire_tuples_dictionary = {
    ('Circular', 'Gabor'): get_cosfire_tuples__Circular_Gabor,
    ('Circular', 'DoG'): get_cosfire_tuples__Circular_DoG
}

compute_tuples_dictionary = {
    ('Circular', 'Gabor'): compute_tuples__Circular_Gabor,
    ('Circular', 'DoG'): compute_tuples__Circular_DoG
}

blur_gaussian_dictionary = {
    ('Circular', 'Gabor'): blur_gaussian__Circular_Gabor,
    ('Circular', 'DoG'): blur_gaussian__Circular_DoG
}

compute_responses_to_filters_dictionary = {
    ('Circular', 'Gabor'): compute_response_to_filters__Circular_Gabor,
    ('Circular', 'DoG'): compute_response_to_filters__Circular_DoG
}

i_scale_cosfire_dictionary = {
    ('Circular', 'Gabor'): i_scale_cosfire__Circular_Gabor,
    ('Circular', 'DoG'): i_scale_cosfire__Circular_DoG
}

i_rotation_cosfire_dictionary = {
    ('Circular', 'Gabor'): i_rotation_cosfire__Circular_Gabor,
    ('Circular', 'DoG'): i_rotation_cosfire__Circular_DoG
}

i_reflection_cosfire_dictionary = {
    ('Circular', 'Gabor'): i_reflection_cosfire__Circular_Gabor,
    ('Circular', 'DoG'): i_reflection_cosfire__Circular_DoG
}
