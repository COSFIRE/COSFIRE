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
from peakutils.peak import indexes
from typing import NamedTuple
import itertools

π = np.pi


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
    ρ: float
    ϕ: float
    λ: float
    θ: float


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
        self.center_y = center_y,
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
        self.responses_to_image = {}
        self._cosfire_tuples = []  # Struct of filter COSFIRE (S_f)
        self.prototype_response_to_filters = {}  # Bank of responses pattern Image
        self.maximum_reponse = 0
        self._cosfire_tuples_invariant = []  # operator invariant to rotation, escala and reflection

    # 1-Configuration COSFIRE Filter
    def fit(self, pattern_image):
        self.prototype_pattern_image = pattern_image
        self.prototype_response_to_filters = self.compute_response_to_filters(self.prototype_pattern_image)  # 1.1
        self.suppress_responses_threshold_1()  # 1.2
        self._cosfire_tuples = self.get_cosfire_tuples(self.prototype_response_to_filters, self.center_x,
                                                       self.center_y)  # 1.3

    # 2-Apply the COSFIRE filtler
    def transform(self, X):
        input_image = X
        self.responses_to_image = self.compute_tuples(input_image)  # 2.1
        self.responses_to_image = self.shift_responses(self.responses_to_image)  # 2.2
        output = self.i_reflection_cosfire(input_image, self._cosfire_tuples, self.responses_to_image)  # 2.3
        maximum_output = output.max()
        cv2.threshold(src=output, dst=output, thresh=self.threshold_3 * maximum_output, maxval=maximum_output,
                      type=cv2.THRESH_TOZERO)
        return output

    # (1.1) Get response filter
    def compute_response_to_filters(self, image, **kwargs):
        return self._compute_response_to_filters(self, image, **kwargs)

    # (1.2) Suppres Resposes
    def suppress_responses_threshold_1(self):
        self.maximum_response = max([value.max() for key, value in self.prototype_response_to_filters.items()])
        [cv2.threshold(src=image, dst=image, thresh=self.threshold_1 * self.maximum_response,
                       maxval=self.maximum_response, type=cv2.THRESH_TOZERO)
         for key, image in
         self.prototype_response_to_filters.items()]  # Desired collateral effect: thresholding self.responses_map
        return

    # (1.3) Get descriptor set (Sf)

    def get_cosfire_tuples(self, bank, center_x, center_y):
        operator = []
        ϕ_array = np.linspace(start=0, stop=2 * π, num=360, endpoint=False)
        for ρ in self.ρ_list:
            if ρ == 0:
                if self.filter_name == 'Gabor':
                    for θ, λ in itertools.product(self.filter_parameters.θ, self.filter_parameters.λ):
                        response_at_center = self.prototype_response_to_filters[GaborKey(θ=θ, λ=λ)][center_x][center_y]
                        if response_at_center > self.maximum_reponse * self.threshold_2:
                            operator.append(CosfireCircularGaborTuple(ρ=0, ϕ=0, λ=λ, θ=θ))
                elif self.filter_name == 'DoG':
                    for σ in self.filter_parameters.σ:
                        response_at_center = self.prototype_response_to_filters[σ][center_x][center_y]
                        if response_at_center > self.maximum_reponse * self.threshold_2:
                            operator.append(CosfireCircularDoGTuple(ρ=0, ϕ=0, σ=σ))
            elif ρ > 0:
                listMax = np.zeros(360)
                direcciones = []
                for k, ϕ in enumerate(ϕ_array):
                    yi = int(center_y + np.floor(ρ * np.cos(ϕ)))
                    xi = int(center_x - np.floor(ρ * np.sin(ϕ)))
                    response_at_center = 0
                    rows, cols = self.prototype_pattern_image.shape
                    if xi >= 0 and yi >= 0 and xi < rows and yi < cols:
                        for gabor_key, response in self.prototype_response_to_filters.items():
                            if response[xi][yi] > response_at_center:
                                response_at_center = response[xi][yi]
                    listMax[k] = response_at_center
                    direcciones.append((xi, yi))
                ss = int(360 / 16)
                if len(np.unique(listMax)) == 1:
                    continue
                listMax1 = np.zeros(listMax.size + 1)
                listMax1[1:] = listMax.copy()
                index = indexes(listMax1, thres=0.2, min_dist=ss)
                index = list(index - 1)
                index = np.array(index)
                for k in range(index.size):
                    if self.filter_name == 'Gabor':
                        for θ in self.filter_parameters.θ:
                            maximum_response = -1
                            for λ in self.filter_parameters.λ:
                                response_at_bearing = self.prototype_response_to_filters[
                                    GaborKey(θ=θ, λ=λ)][direcciones[index[k]][0]][direcciones[index[k]][1]]
                                if response_at_bearing > self.threshold_2 * self.maximum_reponse:
                                    if maximum_response < response_at_bearing:
                                        maximum_response = response_at_bearing
                                        λ_maximum_response = λ
                            if maximum_response != -1:
                                operator.append(CosfireCircularGaborTuple(ρ=ρ,
                                                                          ϕ=index[k] * (π / 180.0),
                                                                          λ=λ_maximum_response,
                                                                          θ=θ))
                    elif self.filter_name == 'DoG':
                        for l, response in self.prototype_response_to_filters.items():
                            response_at_bearing = response[direcciones[index[k]][0]][direcciones[index[k]][1]]
                            if response_at_bearing > self.threshold_2 * self.maximum_reponse:
                                tupla = CosfireCircularDoGTuple(ρ=ρ,
                                                                ϕ=index[k] * (π / 180.0),
                                                                σ=l)
                                operator.append(tupla)
        return operator

    # 2.1 For each tuple in the set Sf compute response
    def compute_tuples(self, inputImage):
        # Aqui llamar funcion que rellena parametros para invarianzas
        operator = self._cosfire_tuples[:]
        if self.reflection_invariant == 1:
            operator += [CosfireCircularGaborTuple(ρ=tupla.ρ,
                                                   ϕ=π - tupla.ϕ,
                                                   λ=tupla.λ,
                                                   θ=π - tupla.θ) for tupla in self._cosfire_tuples]

        for tupla, rotation, scale in itertools.product(operator, self.rotation_invariant, self.scale_invariant):
            if self.filter_name == 'Gabor':
                new_tupla = CosfireCircularGaborTuple(ρ=tupla.ρ * scale,
                                                      ϕ=tupla.ϕ + rotation,
                                                      λ=tupla.λ * scale,
                                                      θ=tupla.θ + rotation)
            elif self.filter_name == 'DoG':
                new_tupla = CosfireCircularDoGTuple(ρ=tupla.ρ * scale,
                                                    ϕ=tupla.ϕ + rotation,
                                                    σ=tupla.σ * scale)
            self._cosfire_tuples_invariant.append(new_tupla)
        unicos = {}
        for tupla in self._cosfire_tuples_invariant:
            if self.filter_name == 'Gabor':
                gabor_key = GaborKey(θ=tupla.θ, λ=tupla.λ)
                if not gabor_key in unicos:
                    θ_array = np.array([gabor_key.θ])
                    λ_array = np.array([gabor_key.λ])
                    tt = self.compute_response_to_filters(inputImage, θ_array=θ_array, λ_array=λ_array)
                    unicos[gabor_key] = tt[gabor_key]
            elif self.filter_name == 'DoG':
                if not tupla[2] in unicos:
                    l1 = np.array(1 * [tupla[2]])
                    tt = compute_response_to_filters__DoG(inputImage, self.filter_parameters, l1)
                    unicos[tupla[2]] = tt[tupla[2]]
        maximum = max([value.max() for key, value in unicos.items()])
        [cv2.threshold(src=image, dst=image, thresh=self.threshold_1 * maximum,
                       maxval=maximum, type=cv2.THRESH_TOZERO)
         for key, image in unicos.items()]
        unicos = self.blur_gaussian(unicos)  ##2.1.1 Hacemos Blur
        return unicos

    # (2.1.1)Blurr
    def blur_gaussian(self, bankFilters):
        dic = {}
        for tupla in self._cosfire_tuples_invariant:
            if self.filter_name == 'Gabor':
                σ = self.alpha * tupla.ρ + self.σ0
                if not tupla in dic:
                    dic[tupla] = cv2.GaussianBlur(bankFilters[(tupla.θ, tupla.λ)], (9, 9), σ, σ)
            elif self.filter_name == 'DoG':
                σ = self.alpha * tupla.ρ + self.σ0
                dic[tupla] = cv2.GaussianBlur(bankFilters[tupla.σ], (9, 9), σ, σ)
        return dic

    # (2.2) Shift
    def shift_responses(self, resp):
        response_maps = {}
        for tupla, response in resp.items():
            rows, cols = response.shape
            x = -tupla.ρ * np.cos(tupla.ϕ)
            y = tupla.ρ * np.sin(tupla.ϕ)
            M = np.float32([[1, 0, x],
                            [0, 1, y]])
            dst = cv2.warpAffine(response, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            response_maps[tupla] = dst
        return response_maps

    # (2.3) invariant under reflection
    def i_reflection_cosfire(self, image, operator, response_bank):
        return self._i_reflection_cosfire(self, image, operator, response_bank)

    # (2.4) invariant to rotation
    def i_rotation_cosfire(self, image, operator, response_bank):
        return self._i_rotation_cosfire(self, image, operator, response_bank)

    # (2.5) invariant to scale
    def i_scale_cosfire(self, image, operator, response_bank):
        return self._i_scale_cosfire(self, image, operator, response_bank)

    # Compute response
    def compute_response(self, conj, operator):
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
        GaborKey(θ=θ, λ=λ): cv2.filter2D(src=image,
                                         ddepth=self.ddepth,
                                         kernel=cv2.getGaborKernel(ksize=ksize,
                                                                   sigma=σ,
                                                                   theta=θ,
                                                                   lambd=λ,
                                                                   gamma=γ,
                                                                   psi=ψ,
                                                                   ktype=ktype))
        for θ, λ in itertools.product(θ_array, λ_array)}
    # TODO: DANIEL ACOSTA COMMENT: Faltaria normalizar cada respuesta de Gabor
    return response_dict


def compute_response_to_filters__Circular_DoG(self, image):
    response = {}
    for σ in self.filter_parameters.σ:
        g1 = cv2.GaussianBlur(image, (3, 3), σ)
        g2 = cv2.GaussianBlur(image, (3, 3), 0.5 * σ)
        r = g1 - g2
        response[σ] = r
    return response


def i_scale_cosfire__Circular_Gabor(self, image, operator, response_bank):
    maximum_output = np.zeros_like(image)
    for value in self.scale_invariant:
        operatorEscala = [
            CosfireCircularGaborTuple(ρ=tupla.ρ * value,
                                      ϕ=tupla.ϕ,
                                      λ=tupla.λ * value,
                                      θ=tupla.θ) for tupla in operator]
        output = self.compute_response(response_bank, operatorEscala)
        maximum_output = np.maximum(output, maximum_output)
    return maximum_output


def i_scale_cosfire__Circular_DoG(self, image, operator, respBank):
    maximum_output = np.zeros_like(image)
    for i in range(len(self.scale_invariant)):
        operatorEscala = [
            CosfireCircularDoGTuple(ρ=tupla.ρ * self.scale_invariant[i],
                                    ϕ=tupla.ϕ,
                                    σ=tupla.σ * self.scale_invariant[i]) for tupla in operator]
        output = self.compute_response(respBank, operatorEscala)
        maximum_output = np.maximum(output, maximum_output)
    return maximum_output


def i_rotation_cosfire__Circular_Gabor(self, image, operator, response_bank):
    maximum_output = np.zeros_like(image)
    for value in self.rotation_invariant:
        operatorRotacion = [
            CosfireCircularGaborTuple(ρ=tupla.ρ,
                                      ϕ=tupla.ϕ + value,
                                      λ=tupla.λ,
                                      θ=tupla.θ + value) for tupla in operator]
        output = self.i_scale_cosfire(image, operatorRotacion, response_bank)
        maximum_output = np.maximum(output, maximum_output)
    return maximum_output


def i_rotation_cosfire__Circular_DoG(self, image, operator, response_bank):
    maximum_output = np.zeros_like(image)
    for value in self.rotation_invariant:
        operatorRotacion = [CosfireCircularDoGTuple(ρ=tupla.ρ, ϕ=tupla.ϕ + value, σ=tupla.σ) for tupla in operator]
        output = self.i_scale_cosfire(image, operatorRotacion, response_bank)
        maximum_output = np.maximum(output, maximum_output)
    return maximum_output


def i_reflection_cosfire__Circular_Gabor(self, image, operator, response_bank):
    maximum_output = self.i_rotation_cosfire(image, operator, response_bank)
    if self.reflection_invariant == 1:
        operatorI = [CosfireCircularGaborTuple(ρ=tupla.ρ,
                                               ϕ=π - tupla.ϕ,
                                               λ=tupla.λ,
                                               θ=π - tupla.θ) for tupla in operator]
        output = self.i_rotation_cosfire(image, operatorI, response_bank)
        maximum_output = np.maximum(output, maximum_output)
    return maximum_output


def i_reflection_cosfire__Circular_DoG(self, image, operator, response_bank):  # pure imagination
    maximum_output = self.i_rotation_cosfire(image, operator, response_bank)
    return maximum_output


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
