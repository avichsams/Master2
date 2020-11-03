import sympy as sp
import sympy
import numpy as np
sp.init_printing(use_latex="mathjax")
class CLPT_symbol:
    def __init__(self, Delta):

        ## valeur (float)
        self.Delta = Delta # vector Orientation des plaque [0,90] exple


        ## expresion (symbol)
        self.smbl_E1, self.smbl_E2 = sp.symbols("E_1 E_2")
        self.smbl_ν12,self.smbl_ν21 = sp.symbols("ν_{12} ν_{21}")
        self.smbl_G12 = sp.symbols("G_{12}")
        self.smbl_δk = sp.symbols("δ_k")
        self.smbl_k = sp.symbols("k")
        self.smbl_t=sp.symbols("t")
        self.smbl_h=sp.symbols("t")
        self.smbl_𝛼k = sp.symbols("𝛼_k")
        self.smbl_to = sp.symbols("t_0")
        self.smbl_Dt = sp.symbols("Δt")

        self.smbl_zk= -self.smbl_h/2 +self.smbl_k*self.smbl_t
        self.smbl_zk_1= -self.smbl_h/2 +(self.smbl_k-1)*self.smbl_t

        ## Q materiaux
        self.Q_ = sympy.Matrix([[self.smbl_E1/(1-self.smbl_ν12*self.smbl_ν21),(self.smbl_ν12*self.smbl_E2)/(1-self.smbl_ν12*self.smbl_ν21),0 ],
                       [(self.smbl_ν12*self.smbl_E2)/(1-self.smbl_ν12*self.smbl_ν21), self.smbl_E2/(1-self.smbl_ν12*self.smbl_ν21),0],[0,0,self.smbl_G12]])

        ## N nombre Couch

        self.N= len(self.Delta)

        ## veteur des δk
        self.vec_δ =[sp.symbols("δ_"+str(i)) for i in range(1,self.N + 1)]

        ## alpha k



    def tens_Q(self,δ):
        Q_=self.Q_

        cos = sympy.cos(δ)
        sin = sympy.sin(δ)

        Qxx = Q_[0,0]*(cos**4) +  (2*(sin**2)*cos**2)*Q_[0,1] +  (4*(sin**2)*cos**2)*Q_[2,2] + sin**4 * Q_[1,1]
        Qxs = Q_[0,0]*(-sin*cos**3) + (sin*cos**3-cos*sin**3)*Q_[0,1] +2*(sin*cos**3-cos*sin**3)*Q_[2,2] +cos*sin**3 *Q_[1,1]
        Qxy = Q_[0,0]*(sin**2 * cos**2) +(sin**4 + cos**4)*Q_[0,1] -4*sin**2 * cos**2 *Q_[2,2] +sin**2 * cos**2 * Q_[1,1]
        Qss = Q_[0,0]*(sin**2 * cos**2) -2*sin**2 * cos**2 *Q_[0,1]  +(sin**2 - cos**2)**2 *Q_[2,2] + sin**2 * cos**2 * Q_[1,1]
        Qys = Q_[0,0]*(-cos*sin**3) + (-sin*cos**3+cos*sin**3)*Q_[0,1] +2*(-sin*cos**3+cos*sin**3)*Q_[2,2] +sin*cos**3 *Q_[1,1]
        Qyy =Q_[0,0]* sin**4 + (2*(sin**2)*cos**2)*Q_[0,1] +  (4*(sin**2)*cos**2)*Q_[2,2] +cos**4 * Q_[1,1]
        Q= sympy.Matrix([[Qxx,Qxy,Qxs],
                               [Qxy, Qyy,Qys],
                         [Qxs,Qys,Qss]])
        return Q
    def alpha(self,k):
        ak=sp.symbols("𝛼_"+str(k))
        return ak*sp.Matrix([1,1,1])
    def tens_Gamma(self,δ,k):
        return self.tens_Q(δ)*self.alpha(k)

    def remplace_delta(self,tenseur):
        vect_delta_subs=[(δi,self.Delta[i]) for i,δi in enumerate(self.vec_δ)  ]
        tenseur=tenseur.subs(vect_delta_subs)
        tenseur= tenseur.evalf()
        tenseur=tenseur.applyfunc(sympy.simplify)
        return tenseur

    def tens_A(self,expression_remplace_delta=True):
        Q_=self.Q_

        Mat = 0*Q_

        for k in range(self.N):
            DZ=sp.simplify(self.smbl_zk-self.smbl_zk_1)
            DZ = DZ.subs(self.smbl_k,k+1)
            Mat+=DZ*self.tens_Q(self.vec_δ[k])

        if expression_remplace_delta:
            Mat=self.remplace_delta(Mat)
        return Mat

    def tens_B(self,expression_remplace_delta=True):
        Q_=self.Q_

        Mat = 0*Q_

        for k in range(self.N):
            DZ=sp.simplify(sp.Rational(1,2)*(self.smbl_zk**2-self.smbl_zk_1**2))
            DZ = DZ.subs(self.smbl_k,k+1)
            Mat+=DZ*self.tens_Q(self.vec_δ[k])

        if expression_remplace_delta:
            Mat=self.remplace_delta(Mat)
        return Mat

    def tens_D(self,expression_remplace_delta=True):
        Q_=self.Q_

        Mat = 0*Q_

        for k in range(self.N):
            fac =sp.symbols("1/3")
            DZ=sp.simplify((self.smbl_zk**3-self.smbl_zk_1**3))
            DZ = DZ.subs(self.smbl_k,k+1)
            Mat+=DZ*self.tens_Q(self.vec_δ[k])

        if expression_remplace_delta:
            Mat=self.remplace_delta(Mat)
        Mat = fac*Mat
        return Mat.subs(fac,sp.Rational(1,3))

    def tens_U(self,expression_remplace_delta=True):


        Mat= 0*self.tens_Gamma(self.smbl_δk,self.smbl_k)


        for k in range(self.N):
            DZ=sp.simplify((self.smbl_zk-self.smbl_zk_1))
            DZ = DZ.subs(self.smbl_k,k+1)
            Mat+=DZ*self.tens_Gamma(self.vec_δ[k],k+1)

        if expression_remplace_delta:
            Mat=self.remplace_delta(Mat)
        return Mat

    def tens_V(self,expression_remplace_delta=True):
        Mat= 0*self.tens_Gamma(self.smbl_δk,self.smbl_k)



        for k in range(self.N):
            DZ=sp.simplify(sp.Rational(1,2)*(self.smbl_zk**2-self.smbl_zk_1**2))
            DZ = DZ.subs(self.smbl_k,k+1)
            Mat+=DZ*self.tens_Gamma(self.vec_δ[k],k+1)

        if expression_remplace_delta:
            Mat=self.remplace_delta(Mat)
        return Mat

    def tens_W(self,expression_remplace_delta=True):
        Mat= 0*self.tens_Gamma(self.smbl_δk,self.smbl_k)



        for k in range(self.N):
            fac =sp.symbols("1/3",float=True)
            DZ=sp.simplify((self.smbl_zk**3-self.smbl_zk_1**3))
            DZ = DZ.subs(self.smbl_k,k+1)
            Mat+=DZ*self.tens_Gamma(self.vec_δ[k],k+1)

        if expression_remplace_delta:
            Mat=self.remplace_delta(Mat)
        Mat = fac*Mat
        return Mat.subs(fac,sp.Rational(1,3))

    def Matrix_ABBD(self,expression_remplace_delta=True):
        Ak = self.tens_A(expression_remplace_delta)
        Bk = self.tens_B(expression_remplace_delta)
        Dk = self.tens_D(expression_remplace_delta)

        Matrix = sp.eye(6)
        Matrix[0:3,0:3]=Ak
        Matrix[0:3,3:6]=Bk
        Matrix[3:6,0:3]=Bk
        Matrix[3:6,3:6]=Dk
        return Matrix
    def Vecteur_NMth(self,expression_remplace_delta=True):
        Uk=self.tens_U(expression_remplace_delta)
        Vk=self.tens_V(expression_remplace_delta)
        Wk=self.tens_W(expression_remplace_delta)

        Nth = -self.smbl_to*Uk -(self.smbl_Dt/self.smbl_h)*Vk
        Mth = -self.smbl_to*Vk -(self.smbl_Dt/self.smbl_h)*Wk
        return sp.Matrix([Nth,Mth])
