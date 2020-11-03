import sympy as sp
import sympy
import numpy as np
from math import *

from CLPT_sympy import CLPT_symbol
sp.init_printing(use_latex="mathjax")

class CLPT_valeur(CLPT_symbol):
    def __init__(self, Delta,E1,E2,ν12,ν21,G12,t,𝛼,to,Dt):

        ## expresion (symbol)
        CLPT_symbol.__init__(self, Delta)


        ## valeur
        self.E1, self.E2       = E1,E2
        self.ν12,self.ν21      = ν12,ν21
        self.G12               = G12
        self.t                 = t
        self.𝛼                 = 𝛼
        self.to                = to
        self.Dt                = Dt

        self.h                 = len(self.Delta)*self.t

        ## veteur des 𝛼k
        self.vec_𝛼 = [sp.symbols("𝛼_"+str(i)) for i in range(1,self.N + 1)]

        ## pour substitution:
        self.subs_E=True;self.subs_nu=True;self.subs_delta=True;self.subs_alpha=True
        self.subs_t=True;self.subs_to=True;self.subs_Dt=True;self.subs_G=True;self.subs_h=True


        ## Solutions
        self.Esp_0,self.K=self.calcul_esp_k()



    def remplace_sym_par_valeur(self,tenseur):
        if self.subs_E :
            tenseur = tenseur.subs([(self.smbl_E1,self.E1),(self.smbl_E2,self.E2)])
        if self.subs_nu :
            tenseur = tenseur.subs([(self.smbl_ν12,self.ν12),(self.smbl_ν21,self.ν21)])
        if self.subs_delta :
            vect_delta_subs=[(δi,self.Delta[i]) for i,δi in enumerate(self.vec_δ)  ]
            tenseur = tenseur.subs(vect_delta_subs)
        if self.subs_alpha :
            vect_alpha_subs=[(δi,self.𝛼[i]) for i,δi in enumerate(self.vec_𝛼)  ]
            tenseur = tenseur.subs(vect_alpha_subs)
        if self.subs_t :
            tenseur = tenseur.subs(self.smbl_t,self.t)
        if self.subs_to:
            tenseur = tenseur.subs(self.smbl_to,self.to)
        if self.subs_Dt:
            tenseur = tenseur.subs(self.smbl_Dt,self.Dt)
        if self.subs_G:
            tenseur = tenseur.subs(self.smbl_G12,self.G12)
        if self.subs_h:
            tenseur = tenseur.subs(self.smbl_h,self.h)




        return tenseur

    def calcul_esp_k(self):
        Matrix_ABBD_sym=self.Matrix_ABBD()
        Vecteur_NMth_sym= self.Vecteur_NMth()
        #valeur
        Matrix_ABBD_val = self.remplace_sym_par_valeur(Matrix_ABBD_sym)
        Vecteur_NMth_val = self.remplace_sym_par_valeur(Vecteur_NMth_sym)

        Matrix_inv= Matrix_ABBD_val.inv()

        vect_esp_k =Matrix_inv*Vecteur_NMth_val

        Esp_0 = sp.Matrix(vect_esp_k[0:3])
        K     = sp.Matrix(vect_esp_k[3:6])

        return Esp_0,K
    def deformation(self,z):
        Esp_0,K=self.calcul_esp_k()
        defr = Esp_0 + z*K
        return defr
    def contrainte_sym_k(self,z):
        Qk = self.remplace_sym_par_valeur(self.tens_Q(self.smbl_δk))
        sig =Qk*self.deformation(z)
        return sig
    def contrainte(self,z):
        sym=sp.symbols("tst")
        if type(z)== type(sym): #si z est un symbole
            sig = self.contrainte_sym_k(z)
        else :
            #trouveer k  à  partir de z
            k = ceil((z-self.h/2)/self.t)

            #on calcule Qk
            Qk = self.remplace_sym_par_valeur(self.tens_Q(self.vec_δ[k]))


            sig=Qk*self.deformation(z)

        return sig
    def contrainte_plaque(self,z,k):
        sig = self.contrainte_sym_k(z)
        sig=sig.subs(self.smbl_δk,self.Delta[k])
        return sig



    def deplacement(self,x,y,z):
        t=sp.symbols("t")
        eps = self.deformation(z)

        u = sp.integrate(eps[0],(t,0,x))
        v = sp.integrate(eps[1],(t,0,y))
        w = eps[2]

        disp = sp.Matrix([u,v,w])

        return disp.expand()
