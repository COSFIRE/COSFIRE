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


class GaborParameters(NamedTuple):
    ksize: tuple
    σ: float
    θ: float
    λ: float
    γ: float
    ψ: float = np.pi * 0.5
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
        self.center_x = center_x
        self.center_y = center_y,
        self.rho_list = [] if rho_list is None else rho_list
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

        self._compute_response_to_filters = compute_responses_to_filters_dictionary[filter_name]
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
        output = self.i_reflexion_cosfire(input_image, self._cosfire_tuples, self.responses_to_image)  # 2.3
        maximum_output = output.max()
        cv2.threshold(src=output, dst=output, thresh=self.threshold_3 * maximum_output, maxval=maximum_output,
                      type=cv2.THRESH_TOZERO)
        return output

    # (1.1) Get response filter
    def compute_response_to_filters(self, image):
        return self._compute_response_to_filters(self, image)

    # (1.2) Suppres Resposes
    def suppress_responses_threshold_1(self):
        self.maximum_response = max([value.max() for key, value in self.prototype_response_to_filters.items()])
        [cv2.threshold(src=image, dst=image, thresh=self.threshold_1 * self.maximum_response,
                       maxval=self.maximum_response, type=cv2.THRESH_TOZERO)
         for key, image in
         self.prototype_response_to_filters.items()]  # Desired collateral effect: thresholding self.responses_map
        return

    # (1.3) Get descriptor set (Sf)

    def get_cosfire_tuples(self, bank, xc, yc):
        operator = []
        phiList = np.arange(360) * np.pi / 180.0  # Discretizacion del circulo
        for i in range(len(self.rho_list)):  # Iteramos en lista de radios
            if self.rho_list[i] == 0:  # Caso rho=0
                if self.filter_name == 'Gabor':
                    for k in range(self.filter_parameters.θ.size):
                        ind = 0
                        val = -1
                        for l in range(self.filter_parameters.λ.size):
                            par = (self.filter_parameters.θ[k], self.filter_parameters.λ[l])
                            if self.prototype_response_to_filters[par][xc][
                                yc] > self.maximum_reponse * self.threshold_2:
                                ind = l
                                val = self.prototype_response_to_filters[par][xc][yc]
                        if val > -1:
                            tupla = CosfireCircularGaborTuple(ρ=0,
                                                              ϕ=0,
                                                              λ=self.filter_parameters.λ[ind],
                                                              θ=self.filter_parameters.θ[k])
                            operator.append(tupla)
                elif self.filter_name == 'DoG':
                    for k in range(self.filter_parameters.σ.size):
                        if self.prototype_response_to_filters[self.filter_parameters.σ[k]][xc][
                            yc] > self.maximum_reponse * self.threshold_2:
                            tupla = CosfireCircularDoGTuple(ρ=0, ϕ=0, σ=self.filter_parameters.σ[k])
                            operator.append(tupla)
            elif self.rho_list[i] > 0:  # Caso rho>0
                listMax = np.zeros(360)
                direcciones = []
                for k in range(phiList.size):
                    yi = int(yc + np.floor(self.rho_list[i] * np.cos(phiList[k])))
                    xi = int(xc - np.floor(self.rho_list[i] * np.sin(phiList[k])))
                    val = 0
                    nr = self.prototype_pattern_image.shape[0]
                    nc = self.prototype_pattern_image.shape[1]
                    if xi >= 0 and yi >= 0 and xi < nr and yi < nc:
                        for l in self.prototype_response_to_filters:
                            if self.prototype_response_to_filters[l][xi][yi] > val:
                                val = self.prototype_response_to_filters[l][xi][yi]
                    listMax[k] = val
                    direcciones.append((xi, yi))
                ss = int(360 / 16)
                # nn=np.arange(360)
                # plt.plot(nn,listMax)
                if len(np.unique(listMax)) == 1:
                    continue
                listMax1 = np.zeros(listMax.size + 1)
                for p in range(listMax.size):
                    listMax1[p + 1] = listMax[p]
                index = indexes(listMax1, thres=0.2, min_dist=ss)
                index = list(index - 1)
                index = np.array(index)
                for k in range(index.size):
                    if self.filter_name == 'Gabor':
                        for l in range(self.filter_parameters.θ.size):
                            mx = -1
                            ind = 0
                            for m in range(self.filter_parameters.λ.size):
                                par = (self.filter_parameters.θ[l], self.filter_parameters.λ[m])
                                var = self.prototype_response_to_filters[par][direcciones[index[k]][0]][
                                    direcciones[index[k]][1]]
                                if var > self.threshold_2 * self.maximum_reponse:
                                    if mx < var:
                                        mx = var
                                        ind = m
                            if mx != -1:
                                tupla = CosfireCircularGaborTuple(ρ=self.rho_list[i],
                                                                  ϕ=index[k] * (np.pi / 180.0),
                                                                  λ=self.filter_parameters.λ[ind],
                                                                  θ=self.filter_parameters.θ[l])
                                operator.append(tupla)
                    elif self.filter_name == 'DoG':
                        for l in self.prototype_response_to_filters:
                            var = self.prototype_response_to_filters[l][direcciones[index[k]][0]][
                                direcciones[index[k]][1]]
                            if var > self.threshold_2 * self.maximum_reponse:
                                tupla = CosfireCircularDoGTuple(ρ=self.rho_list[i],
                                                                ϕ=index[k] * (np.pi / 180.0),
                                                                σ=l)
                                operator.append(tupla)
        return operator

    # 2.1 For each tuple in the set Sf compute response
    def compute_tuples(self, inputImage):
        # Aqui llamar funcion que rellena parametros para invarianzas
        ope = [self._cosfire_tuples[:]]
        if self.reflection_invariant == 1:
            reflection = [CosfireCircularGaborTuple(ρ=tupla.ρ,
                                                    ϕ=np.pi - tupla.ϕ,
                                                    λ=tupla.λ,
                                                    θ=np.pi - tupla.θ)
                          for tupla in self._cosfire_tuples]
            ope.append(reflection)
        for i in range(len(ope)):
            for tupla in ope[i]:
                for j in range(len(self.rotation_invariant)):
                    for k in range(len(self.scale_invariant)):
                        if self.filter_name == 'Gabor':
                            new_tupla = CosfireCircularGaborTuple(ρ=tupla.ρ * self.scale_invariant[k],
                                                                  ϕ=tupla.ϕ + self.rotation_invariant[j],
                                                                  λ=tupla.λ * self.scale_invariant[k],
                                                                  θ=tupla.θ + self.rotation_invariant[j])

                            self._cosfire_tuples_invariant.append(new_tupla)
                        elif self.filter_name == 'DoG':
                            new_tupla = CosfireCircularDoGTuple(ρ=tupla.ρ * self.scale_invariant[k],
                                                                ϕ=tupla.ϕ + self.rotation_invariant[j],
                                                                σ=tupla.σ * self.scale_invariant[k])
                            self._cosfire_tuples_invariant.append(new_tupla)
        unicos = {}
        for i in range(len(self._cosfire_tuples_invariant)):
            if self.filter_name == 'Gabor':
                new_tupla = (self._cosfire_tuples_invariant[i][3], self._cosfire_tuples_invariant[i][2])
                if not new_tupla in unicos:
                    l1 = np.array(1 * [new_tupla[0]])
                    l2 = np.array(1 * [new_tupla[1]])
                    tt = self._compute_response_to_filters(self, inputImage)
                    unicos[new_tupla] = tt[new_tupla]
            elif self.filter_name == 'DoG':
                if not self._cosfire_tuples_invariant[i][2] in unicos:
                    l1 = np.array(1 * [self._cosfire_tuples_invariant[i][2]])
                    tt = compute_response_to_filters__DoG(inputImage, self.filter_parameters, l1)
                    unicos[self._cosfire_tuples_invariant[i][2]] = tt[self._cosfire_tuples_invariant[i][2]]
        max = 0
        for i in unicos:
            t = unicos[i].shape
            for j in range(t[0]):
                for k in range(t[1]):
                    sig = unicos[i][j][k]
                    if sig > max:
                        max = unicos[i][j][k]
        for i in unicos:
            t = unicos[i].shape
            for j in range(t[0]):
                for k in range(t[1]):
                    if unicos[i][j][k] < max * self.threshold_1:
                        unicos[i][j][k] = 0
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
        Resp = {}
        for clave, img in resp.items():
            nr = img.shape[0]
            nc = img.shape[1]
            x = clave[0] * np.sin(np.pi + clave[1])
            y = clave[0] * np.cos(np.pi + clave[1])
            nw = np.copy(img)
            for k in range(nr):
                for l in range(nc):
                    xx = int(k + x)
                    yy = int(l - y)
                    if xx >= 0 and xx < nr and yy >= 0 and yy < nc:
                        nw[k][l] = img[xx][yy]
                    else:
                        nw[k][l] = 0
            Resp[clave] = nw
        return Resp

    # (2.3) invariant under reflection
    def i_reflexion_cosfire(self, imagen, operator, respBank):
        out1 = self.i_rotation_cosfire(imagen, operator, respBank)
        if self.reflection_invariant == 1:
            operatorI = [CosfireCircularGaborTuple(ρ=tupla.ρ,
                                                   ϕ=np.pi - tupla.ϕ,
                                                   λ=tupla.λ,
                                                   θ=np.pi - tupla.θ) for tupla in operator]
            out2 = self.i_rotation_cosfire(imagen, operatorI, respBank)
            out1 = np.maximum(out2, out1)
        return out1

    # (2.4) invariant to rotation
    def i_rotation_cosfire(self, imagen, operator, respBank):
        output = np.zeros((imagen.shape[0], imagen.shape[1]))
        for i in range(len(self.rotation_invariant)):
            operatorRotacion = []
            for tupla in operator:
                if self.filter_name == 'Gabor':
                    new_tupla = CosfireCircularGaborTuple(ρ=tupla.ρ,
                                                          ϕ=tupla.ϕ + self.rotation_invariant[i],
                                                          λ=tupla.λ,
                                                          θ=tupla.θ + self.rotation_invariant[i])
                elif self.filter_name == 'DoG':
                    new_tupla = (operator[j][0], operator[j][1] + self.rotation_invariant[i], operator[j][2])
                operatorRotacion.append(new_tupla)
            outR = self.i_scale_cosfire(imagen, operatorRotacion, respBank)
            output = np.maximum(outR, output)
        return output

    # (2.5) invariant to scale
    def i_scale_cosfire(self, imagen, operator, respBank):
        output = np.zeros((imagen.shape[0], imagen.shape[1]))
        for i in range(len(self.scale_invariant)):
            operatorEscala = []
            for tupla in operator:
                if self.filter_name == 'Gabor':
                    new_tupla = CosfireCircularGaborTuple(ρ=tupla.ρ * self.scale_invariant[i],
                                                          ϕ=tupla.ϕ,
                                                          λ=tupla.λ * self.scale_invariant[i],
                                                          θ=tupla.θ)
                elif self.filter_name == 'DoG':
                    new_tupla = (tupla.ρ * self.scale_invariant[i],
                                 tupla.ϕ,
                                 tupla.σ * self.scale_invariant[i])
                operatorEscala.append(new_tupla)
            outR = self.compute_response(respBank, operatorEscala)
            output = np.maximum(outR, output)
        return output

    # Compute response
    def compute_response(self, conj, operator):
        nr = conj[operator[0]].shape[0]
        nc = conj[operator[0]].shape[1]
        resp = np.zeros((nr, nc))
        rhoMax = 0
        for i in range(len(operator)):
            if operator[i][0] > rhoMax:
                rhoMax = operator[i][0]
        tupleweightσ = np.sqrt(-((rhoMax ** 2) / (2 * np.log(0.5))))
        for i in range(nr):
            for j in range(nc):
                val = 1
                suma = 0
                for k in range(len(operator)):
                    aux = np.exp(-(operator[k][0] ** 2) / (2 * tupleweightσ ** 2))
                    # if self.nameFilter=='DoG':
                    #    aux=1
                    suma += aux
                    wi = aux
                    val *= (conj[operator[k]][i][j] ** wi)
                val = np.power(val, 1.0 / suma)
                resp[i][j] = val
        return resp


def compute_response_to_filters__Gabor(self, image):
    ksize, σ, θ_array, λ_array, γ, ψ, ktype = self.filter_parameters
    response_dict = {GaborKey(θ=θ, λ=λ): cv2.filter2D(src=image,
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


def compute_response_to_filters__DoG(self, image):
    response = {}
    for σ in self.filter_parameters.σ:
        g1 = cv2.GaussianBlur(image, (3, 3), σ)
        g2 = cv2.GaussianBlur(image, (3, 3), 0.5 * σ)
        r = g1 - g2
        response[σ] = r
    return response


compute_responses_to_filters_dictionary = {
    'Gabor': compute_response_to_filters__Gabor,
    'DoG': compute_response_to_filters__DoG
}
