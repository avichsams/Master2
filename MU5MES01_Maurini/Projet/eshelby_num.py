import ufl
import dolfin as df
import numpy  as np

class EshelbyDisk_num:
    def __init__(self,mesh,keys,dx,ds,V,u,v,rho,g,R_in,R_out,E_m,E_i, ν_i, ν_m,ud):
        
        self.mesh   = mesh
        self.keys   = keys
        self.dx     = dx
        self.ds     = ds
        self.V      = V
        self.u      = u
        self.v      = v
        self.rho    = rho
        self.g      = g
        self.R_in   = R_in
        self.R_out  = R_out
        self.E_i    = E_i
        self.E_m    = E_m
        self.ν_i    = ν_i
        self.ν_m    = ν_m
        self.ud     = ud
    def face_externe(self,tol = 1E-1):
        return df.CompiledSubDomain("near(sqrt(x[1]*x[1] +x[0]*x[0]), R, tol) && on_boundary",\
                                 R=self.R_out,tol=tol)

    def bondary_conditions(self):
        return df.DirichletBC(self.V, self.ud, self.face_externe())


    def strain(self, u):
        return df.sym(df.grad(u))

    def stress(self,epsilon,E,nu):
        ndim = self.u.geometric_dimension()
        mu, lmbda = E/(2*(1.0 + nu)), E*nu/((1.0 + nu)*(1.0 -2.0*nu))
        return  2*mu*epsilon + lmbda*df.tr(epsilon)*df.Identity(ndim)

    def linear_form(self):
        poids = df.Constant((0.,-self.rho*self.g))
        return ufl.inner(poids,self.v)*self.dx(self.keys[0]) 

    def bilinear_form(self):
        a_i = df.inner(self.stress(self.strain(self.u),self.E_i,self.ν_i), self.strain(self.v))*self.dx(self.keys[1])
        a_m = df.inner(self.stress(self.strain(self.u),self.E_m,self.ν_m), self.strain(self.v))*self.dx(self.keys[2]) 
        return a_i + a_m

    def resolution(self):
        a  = self.bilinear_form()
        L  = self.linear_form()
        bc = self.bondary_conditions()

        usol = df.Function(self.V)
        
        df.solve(a == L, usol, bc)

        return usol
    

   

