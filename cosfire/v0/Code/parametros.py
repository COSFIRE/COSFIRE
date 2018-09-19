import numpy as np


#filterParam Gabor:

def parametros1(filtro):
    filtro.nameFilter="Gabor"
    filtro.coorX=166
    filtro.coorY=96
    filtro.rhoList.append(0)
    filtro.rhoList.append(12)
    filtro.eta=np.pi/8
    filtro.t1=0.99
    filtro.filterParam.append((10,10))
    filtro.filterParam.append(5)
    filtro.t2=0.75
    filtro.sigma0=0.67
    filtro.alpha=0.04
    filtro.t3=0.9
    th=np.zeros(2)
    # for i in range(15):
    #     th[i+1]=th[i]+np.pi/(8.0)
    th[0]=0
    th[1]=np.pi/2
    filtro.filterParam.append(th)
    lambd=np.zeros(1)
    lambd[0]=12
    filtro.filterParam.append(lambd)
    filtro.filterParam.append(0.5)
    filtro.filterParam.append(np.pi)
    filtro.invarianteReflexion=0 #Parametro de invarianza a reflexion Si=1 No=0
    filtro.invarianteEscala.append(1)#Lista de parametros a invarianza a escala (default=1)
    filtro.invarianteRotacion.append(0)

def parametros2(filtro):
    filtro.nameFilter="Gabor"
    filtro.coorX=166
    filtro.coorY=96
    filtro.rhoList.append(0)
    filtro.rhoList.append(12)
    filtro.eta=np.pi/8
    filtro.t1=0.99
    filtro.filterParam.append((10,10))
    filtro.filterParam.append(5)
    filtro.t2=0.75
    filtro.sigma0=0.67
    filtro.alpha=0.04
    filtro.t3=0.9
    th=np.zeros(2)
    # for i in range(15):
    #     th[i+1]=th[i]+np.pi/(8.0)
    th[0]=0
    th[1]=np.pi/2
    filtro.filterParam.append(th)
    lambd=np.zeros(1)
    lambd[0]=12
    filtro.filterParam.append(lambd)
    filtro.filterParam.append(0.5)
    filtro.filterParam.append(np.pi)
    filtro.invarianteReflexion=1 #Parametro de invarianza a reflexion Si=1 No=0
    filtro.invarianteEscala.append(1)#Lista de parametros a invarianza a escala (default=1)
    filtro.invarianteRotacion.append(0)

    
def parametros3(filtro):
    filtro.nameFilter="Gabor"
    filtro.coorX=24
    filtro.coorY=28
    filtro.rhoList.append(0)
    filtro.rhoList.append(2)
    filtro.rhoList.append(4)
    filtro.rhoList.append(7)
    filtro.rhoList.append(10)
    filtro.rhoList.append(13)
    filtro.rhoList.append(16)
    filtro.rhoList.append(20)
    filtro.rhoList.append(25)
    filtro.eta=np.pi/8
    filtro.t1=0.1
    filtro.filterParam.append((10,10))
    filtro.filterParam.append(0.8)
    filtro.t2=0.75
    filtro.sigma0=0.67
    filtro.alpha=0.04
    filtro.t3=0.99
    th=np.zeros(16)
    for i in range(16):
        th[i]=(i*np.pi)/8.0
    filtro.filterParam.append(th)
    lambd=np.zeros(1)
    for i in range(1):
        lambd[i]=4
    filtro.filterParam.append(lambd)
    filtro.filterParam.append(0.5)
    filtro.filterParam.append(np.pi/2)
    filtro.invarianteReflexion=0 #Parametro de invarianza a reflexion Si=1 No=0
    filtro.invarianteEscala.append(1)#Lista de parametros a invarianza a escala (default=1)
    filtro.invarianteRotacion.append(0)
    
def parametros6(filtro):
    filtro.nameFilter="Gabor"
    filtro.coorX=17
    filtro.coorY=17
    filtro.rhoList.append(0)
    filtro.rhoList.append(3)
    filtro.rhoList.append(8)
    filtro.eta=np.pi/8
    filtro.t1= 0.05

    filtro.t2=0.75
    filtro.sigma0=0.83
    filtro.alpha=0.1
    filtro.t3=0.3
    th=np.zeros(16)
    for i in range(16):
         th[i]=(i*np.pi)/(8.0)
    filtro.filterParam.append(th)
    lambd=np.zeros(1)
    lambd[0]=2*np.sqrt(2)
    filtro.filterParam.append((5,5))
    filtro.filterParam.append(0.8)
    filtro.filterParam.append(lambd)
    filtro.filterParam.append(0.5)
    filtro.filterParam.append(np.pi/2)
    filtro.invarianteReflexion=0 #Parametro de invarianza a reflexion Si=1 No=0
    filtro.invarianteEscala.append(1)#L
    filtro.invarianteRotacion.append(0)