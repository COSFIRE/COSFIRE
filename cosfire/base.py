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
import cosfire.GaborFilter as GF
from peakutils.peak import indexes
import cosfire.DoGFilter as DoG


class Cosfire:
    def __init__(self,
                 filter_name='NN',
                 center_coordinate_x=0,
                 center_coordinate_y=0,
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
                 scale_invariant=None):
        self.filter_name = filter_name
        self.center_coordinate_x = center_coordinate_x
        self.center_coordinate_y = center_coordinate_y,
        self.rho_list = [] if rho_list is None else rho_list
        self.eta = eta
        self.t1 = t1
        self.filter_parameters = [] if filter_parameters is None else filter_parameters  # Parameters of filter
        self.t2 = t2
        self.alpha = alpha
        self.sigma0 = sigma0
        self.t3 = t3
        self.reflection_invariant = reflection_invariant
        self.scale_invariant = [] if scale_invariant is None else scale_invariant
        self.rotation_invariant = [] if rotation_invariant is None else rotation_invariant

        self.responses_to_image = {}
        self.operator = []  # Struct of filter COSFIRE (S_f)
        self.input = {}  # Bank of responses pattern Image
        self.maximum_reponse = 0
        self.operator1 = []  # operator invariant to rotation, escala and reflection

    # 1-Configuration COSFIRE Filter
    def fit(self, pattern_image):
        self.pattern_image = pattern_image
        self.input = self.getResponseFilter()  # 1.1
        self.suppresResponsesT1()  # 1.2
        self.operator = self.get_cosfire_tuples(self.input, self.center_coordinate_x, self.center_coordinate_y)  # 1.3

    # 2-Apply the COSFIRE filtler
    def transform(self, Image, ):
        inputImage = Image
        # print "calculando banco"
        self.responses_to_image = self.compute_tuples(inputImage)  # 2.1
        self.responses_to_image = self.shift_responses(self.responses_to_image)  # 2.2
        # print "calculando respuesta"
        output = self.i_reflexion_cosfire(inputImage, self.operator, self.responses_to_image)  # 2.3
        mx = 0
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                if output[i][j] > mx:
                    mx = output[i][j]
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                if output[i][j] < mx * self.t3:
                    output[i][j] = 0
        return output

    # (1.1) Get response filter
    def getResponseFilter(self):
        if self.filter_name == 'Gabor':
            return GF.get_gabor_response(self.pattern_image, self.filter_parameters,
                                         self.filter_parameters[2], self.filter_parameters[3])
        elif self.filter_name == 'DoG':
            return DoG.get_difference_of_gaussians_response(self.pattern_image, self.filter_parameters,
                                                            self.filter_parameters[0])

    # (1.2) Suppres Resposes
    def suppresResponsesT1(self):
        eMax = []
        for i in self.input:
            eMax.append(self.input[i].max())
        self.maximum_reponse = max(eMax)
        for clav in self.input:
            for k in range(self.input[clav].shape[0]):
                for l in range(self.input[clav].shape[1]):
                    if self.input[clav][k][l] < self.t1 * self.maximum_reponse:
                        self.input[clav][k][l] = 0
        return

        # (1.3) Get descriptor set (Sf)

    def get_cosfire_tuples(self, bank, xc, yc):
        operator = []
        phiList = np.arange(360) * np.pi / 180.0  # Discretizacion del circulo
        for i in range(len(self.rho_list)):  # Iteramos en lista de radios
            if self.rho_list[i] == 0:  # Caso rho=0
                if self.filter_name == 'Gabor':
                    for k in range(self.filter_parameters[2].size):
                        ind = 0
                        val = -1
                        tupla = np.zeros(4)
                        for l in range(self.filter_parameters[3].size):
                            par = (self.filter_parameters[2][k], self.filter_parameters[3][l])
                            if self.input[par][xc][yc] > self.maximum_reponse * self.t2:
                                ind = l
                                val = self.input[par][xc][yc]
                        if val > -1:
                            tupla[2] = self.filter_parameters[3][ind]
                            tupla[3] = self.filter_parameters[2][k]
                            operator.append(tupla)
                elif self.filter_name == 'DoG':
                    for k in range(self.filter_parameters[0].size):
                        if self.input[self.filter_parameters[0][k]][xc][yc] > self.maximum_reponse * self.t2:
                            tupla = np.zeros(3)
                            tupla[2] = self.filter_parameters[0][k]
                            operator.append(tupla)
            elif self.rho_list[i] > 0:  # Caso rho>0
                listMax = np.zeros(360)
                direcciones = []
                for k in range(phiList.size):
                    yi = int(yc + np.floor(self.rho_list[i] * np.cos(phiList[k])))
                    xi = int(xc - np.floor(self.rho_list[i] * np.sin(phiList[k])))
                    val = 0
                    nr = self.pattern_image.shape[0]
                    nc = self.pattern_image.shape[1]
                    if xi >= 0 and yi >= 0 and xi < nr and yi < nc:
                        for l in self.input:
                            if self.input[l][xi][yi] > val:
                                val = self.input[l][xi][yi]
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
                        for l in range(self.filter_parameters[2].size):
                            mx = -1
                            ind = 0
                            for m in range(self.filter_parameters[3].size):
                                par = (self.filter_parameters[2][l], self.filter_parameters[3][m])
                                var = self.input[par][direcciones[index[k]][0]][direcciones[index[k]][1]]
                                if var > self.t2 * self.maximum_reponse:
                                    if mx < var:
                                        mx = var
                                        ind = m
                            if mx != -1:
                                tupla = np.zeros(4)
                                tupla[0] = self.rho_list[i]
                                tupla[1] = index[k] * (np.pi / 180.0)
                                tupla[2] = self.filter_parameters[3][ind]
                                tupla[3] = self.filter_parameters[2][l]
                                operator.append(tupla)
                    elif self.filter_name == 'DoG':
                        for l in self.input:
                            var = self.input[l][direcciones[index[k]][0]][direcciones[index[k]][1]]
                            if var > self.t2 * self.maximum_reponse:
                                tupla = np.zeros(3)
                                tupla[0] = self.rho_list[i]
                                tupla[1] = index[k] * (np.pi / 180.0)
                                tupla[2] = l
                                operator.append(tupla)
        return operator

    # 2.1 For each tuple in the set Sf compute response
    def compute_tuples(self, inputImage):
        # Aqui llamar funcion que rellena parametros para invarianzas
        ope = []
        normal = []
        for i in range(len(self.operator)):
            normal.append(self.operator[i])
        ope.append(normal)
        if self.reflection_invariant == 1:
            refleccion = []
            for i in range(len(self.operator)):
                a = (self.operator[i][0], np.pi - self.operator[i][1], self.operator[i][2], np.pi - self.operator[i][3])
                refleccion.append(a)
            ope.append(refleccion)
        for i in range(len(ope)):
            for l in range(len(ope[i])):
                for j in range(len(self.rotation_invariant)):
                    for k in range(len(self.scale_invariant)):
                        if self.filter_name == 'Gabor':
                            aux = np.zeros(4)
                            aux[0] = ope[i][l][0] * self.scale_invariant[k]
                            aux[1] = ope[i][l][1] + self.rotation_invariant[j]
                            aux[2] = ope[i][l][2] * self.scale_invariant[k]
                            aux[3] = ope[i][l][3] + self.rotation_invariant[j]
                            self.operator1.append(aux)
                        elif self.filter_name == 'DoG':
                            aux = np.zeros(3)
                            aux[0] = ope[i][l][0] * self.scale_invariant[k]
                            aux[1] = ope[i][l][1] + self.rotation_invariant[j]
                            aux[2] = ope[i][l][2] * self.scale_invariant[k]
                            self.operator1.append(aux)
        unicos = {}
        for i in range(len(self.operator1)):
            if self.filter_name == 'Gabor':
                a = (self.operator1[i][3], self.operator1[i][2])
                if not a in unicos:
                    l1 = np.array(1 * [a[0]])
                    l2 = np.array(1 * [a[1]])
                    tt = GF.get_gabor_response(inputImage, self.filter_parameters, l1, l2)
                    unicos[a] = tt[a]
            elif self.filter_name == 'DoG':
                if not self.operator1[i][2] in unicos:
                    l1 = np.array(1 * [self.operator1[i][2]])
                    tt = DoG.get_difference_of_gaussians_response(inputImage, self.filter_parameters, l1)
                    unicos[self.operator1[i][2]] = tt[self.operator1[i][2]]
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
                    if unicos[i][j][k] < max * self.t1:
                        unicos[i][j][k] = 0
        unicos = self.blur_gaussian(unicos)  ##2.1.1 Hacemos Blur
        return unicos

    # (2.1.1)Blurr
    def blur_gaussian(self, bankFilters):
        dic = {}
        for i in range(len(self.operator1)):
            if self.filter_name == 'Gabor':
                a1 = (self.operator1[i][3], self.operator1[i][2])
                sigma = self.alpha * self.operator1[i][0] + self.sigma0
                # cv2.imshow("wsw",cv2.GaussianBlur(bankFilters[a1], (15,15),sigma, sigma))
                a2 = (self.operator1[i][0], self.operator1[i][1], self.operator1[i][2], self.operator1[i][3])
                if not a2 in dic:
                    dic[a2] = cv2.GaussianBlur(bankFilters[a1], (9, 9), sigma, sigma)
            elif self.filter_name == 'DoG':
                sigma = self.alpha * self.operator1[i][0] + self.sigma0
                a2 = (self.operator1[i][0], self.operator1[i][1], self.operator1[i][2])
                dic[a2] = cv2.GaussianBlur(bankFilters[self.operator1[i][2]], (9, 9), sigma, sigma)
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
        # cv2.imshow("dsds1111",out1)
        if self.reflection_invariant == 1:
            operatorI = []
            for i in range(len(operator)):
                a = (operator[i][0], np.pi - operator[i][1], operator[i][2], np.pi - operator[i][3])
                operatorI.append(a)
            out2 = self.i_rotation_cosfire(imagen, operatorI, respBank)
            # cv2.imshow("dsfsd222222",out2)
            self.return_maximum(out2, out1)  ##Definir maxi
        return out1

    # (2.4) invariant to rotation
    def i_rotation_cosfire(self, imagen, operator, respBank):
        output = np.zeros((imagen.shape[0], imagen.shape[1]))
        for i in range(len(self.rotation_invariant)):
            operatorRotacion = []
            for j in range(len(operator)):
                if self.filter_name == 'Gabor':
                    a = (operator[j][0], operator[j][1] + self.rotation_invariant[i], operator[j][2],
                         operator[j][3] + self.rotation_invariant[i])
                elif self.filter_name == 'DoG':
                    a = (operator[j][0], operator[j][1] + self.rotation_invariant[i], operator[j][2])
                operatorRotacion.append(a)
            outR = self.i_scale_cosfire(imagen, operatorRotacion, respBank)
            self.return_maximum(outR, output)
        return output

    # (2.5) invariant to scale
    def i_scale_cosfire(self, imagen, operator, respBank):
        output = np.zeros((imagen.shape[0], imagen.shape[1]))
        for i in range(len(self.scale_invariant)):
            operatorEscala = []
            for j in range(len(operator)):
                if self.filter_name == 'Gabor':
                    a = (
                        operator[j][0] * self.scale_invariant[i], operator[j][1],
                        operator[j][2] * self.scale_invariant[i],
                        operator[j][3])
                elif self.filter_name == 'DoG':
                    a = (
                        operator[j][0] * self.scale_invariant[i], operator[j][1],
                        operator[j][2] * self.scale_invariant[i])
                operatorEscala.append(a)
            outR = self.compute_response(respBank, operatorEscala)
            self.return_maximum(outR, output)
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
        tupleweightsigma = np.sqrt(-((rhoMax ** 2) / (2 * np.log(0.5))))
        for i in range(nr):
            for j in range(nc):
                val = 1
                suma = 0
                for k in range(len(operator)):
                    aux = np.exp(-(operator[k][0] ** 2) / (2 * tupleweightsigma ** 2))
                    # if self.nameFilter=='DoG':
                    #    aux=1
                    suma += aux
                    wi = aux
                    val *= (conj[operator[k]][i][j] ** wi)
                val = np.power(val, 1.0 / suma)
                resp[i][j] = val
        return resp

    # Select the maximum of two answers
    def return_maximum(self, mat1, mat2):
        for i in range(mat1.shape[0]):
            for j in range(mat1.shape[1]):
                mat2[i][j] = max(mat1[i][j], mat2[i][j])

