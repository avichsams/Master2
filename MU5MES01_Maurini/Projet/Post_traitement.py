import ufl
import dolfin as df
import dolfin
import numpy  as np
import matplotlib.pyplot as plt

class Eshelby_Post_treat:
    def __init__(self,mesh,a,b,usol,E_m,E_i, ν_i, ν_m):

        self.mesh   = mesh
        self.a      = a
        self.b      = b
        self.usol   = usol
        self.E_i    = E_i
        self.E_m    = E_m
        self.ν_i    = ν_i
        self.ν_m    = ν_m
        self.ndim   = self.usol.geometric_dimension()




    def strain(self):
        return df.sym(df.grad(self.usol))

    def stress(self):
        tol = 1E-8
        # on défini E(x) = E_i si x € Inclusion; et  E(x) = E_m si non
        E = df.Expression('((x[0]*x[0])/(a*a))+((x[1]*x[1])/(b*b)) <= 1 + tol ? E_i : E_m'\
                  , degree=0,a=self.a,b=self.b ,tol=tol, E_i=self.E_i, E_m=self.E_m )

        # on défini E(x) = E_i si x € Inclusion; et  E(x) = E_m si non
        nu = df.Expression('((x[0]*x[0])/(a*a))+((x[1]*x[1])/(b*b)) <= 1 + tol ? nu_i : nu_m'\
                  , degree=0,a=self.a,b=self.b,tol=tol, nu_i=self.ν_i, nu_m=self.ν_m)

        ndim = self.ndim
        mu, lmbda = E/(2*(1.0 + nu)), E*nu/((1.0 + nu)*(1.0 -2.0*nu))
        epsilon=self.strain()
        return  2*mu*epsilon + lmbda*df.tr(epsilon)*df.Identity(ndim)

    def von_mises(self):

        sigma_sol=self.stress()
        ndim = self.usol.geometric_dimension()

        s=sigma_sol -(1/ndim)*ufl.tr(sigma_sol)*df.Identity(ndim)

        von_mises = ufl.sqrt(ndim*ufl.inner(s, s))

        return von_mises

    def transforme_champs_en_array(self,champs):
        '''
            une fonction qui transforme un champs en array
            qui na finalement pas beaucoup servi, appart calculé
            les contrainte max
        '''


        # projection du champs dans le ùaillage
        strain_element = dolfin.FiniteElement('DG', dolfin.triangle, 1)
        S = dolfin.FunctionSpace(self.mesh, strain_element)
        champs_p = dolfin.project(champs, S)

        #tout les point du maillage
        mesh_points=self.mesh.coordinates()

        # calcul du vecteur
        A_array=np.array([champs_p(x,y) for x,y in mesh_points])

        return A_array

    def plot_all_strain(self):

        #calcul des deformations
        champs=self.strain()

        #projection des deformation dans le maillage
        strain_element = dolfin.FiniteElement('DG', dolfin.triangle, 1)
        S = dolfin.TensorFunctionSpace(self.mesh, strain_element,degree=2)
        champs_p = dolfin.project(champs, S)

        ndim = self.ndim

        X=np.array(['x','y'])   # pour le tritre des plot
        n_fig = 0               # pour l'affichage des plot
        for i in range(ndim):
            for j in range(ndim):
                if i!=0 and j!=1 :
                    rien=0 #car symetrie
                else :

                    titre = r" $ε_{"+str(X[i])+""+str(X[j])+"} $"
                    n_fig+=1
                    plt.subplot(220+n_fig)

                    plt.colorbar(dolfin.plot(champs_p[i,j],title=titre,\
                                         wireframe = True),orientation="vertical")

                    #calcul de la deformation maximale suivant chaque composantes
                    print('la valeur maximale de la deformation '+titre + '= ',np.max(self.transforme_champs_en_array(champs[i,j])) )
        plt.show()

    def plot_all_stress(self,with_von_mises=False):
        # si with_von_mises == True plot aussi sigma von mises

        #calcul des contraintes
        champs=self.stress()

        #projection des contraintes dans le maillage
        stress_element = dolfin.FiniteElement('DG', dolfin.triangle, 1)
        S = dolfin.TensorFunctionSpace(self.mesh, stress_element,degree=2)

        champs_p = dolfin.project(champs, S)

        ndim = self.usol.geometric_dimension()
        X=np.array(['x','y'])   # pour le tritre des plot
        n_fig = 0               # pour l'affichage des plot
        for i in range(ndim):
            for j in range(ndim):
                if i!=0 and j!=1 :
                    rien =0 #car symetrie affiche que xy pas besoin de yx
                else :
                    titre = r"$σ_{"+str(X[i])+""+str(X[j])+"}$"
                    n_fig +=1
                    plt.subplot(220+n_fig)

                    plt.colorbar(dolfin.plot(champs_p[i,j],title=titre,\
                                         wireframe = True),orientation="vertical")
                    #calcul des contraintes maximale suivant chaque composantes
                    print('la valeur maximale de la contrainte '+titre + '= ',np.max(self.transforme_champs_en_array(champs[i,j])) )

        if with_von_mises :
            stress_element = dolfin.FiniteElement('DG', dolfin.triangle, 0)
            von_mises = self.von_mises()
            S = dolfin.FunctionSpace(self.mesh, stress_element)

            von_Mises_p = dolfin.project(von_mises, S)

            titre=r"$ σ_{vm} $ "
            plt.subplot(220+4)
            plt.colorbar(dolfin.plot(von_Mises_p,title=titre,\
                                 wireframe = True),orientation="vertical")

            #print von mises max
            print('la valeur maximale de la contrainte equivalente '+titre + '  =',np.max(self.transforme_champs_en_array(von_mises)) )


        plt.show()



    def plot_von_mises(self):
        # permet de plot von mises uniquement
        stress_element = dolfin.FiniteElement('DG', dolfin.triangle, 0)
        S = dolfin.FunctionSpace(self.mesh, stress_element)

        von_mises = self.von_mises()
        von_Mises_p = dolfin.project(von_mises, S)

        titre=r"$ σ_{vm} $ "

        plt.colorbar(dolfin.plot(von_Mises_p,title=titre,\
                             wireframe = True),orientation="vertical")

        #print von mises max
        print('la valeur maximale de la contrainte equivalente '+titre + '  =',np.max(self.transforme_champs_en_array(von_mises)) )


    def average_strain(self,dx,choix_print=True):
        '''
        calcul des déformations moyennes
            Parameters
            ----------
            dx : TYPE dolfin.mesure
                 domaine d'intégration

            choix_print: TYPE booléen
                si True : return des valeurs et print des phrases pour chaque composantes
                si False: return juste les valeurs sans phrases

            Returns
            -------
            un array contenant  une valeur  moyenne pour chaque  composantes

        '''
        # calcul des déformation
        eps = self.strain()

        # projection
        strain_element = dolfin.FiniteElement('DG', dolfin.triangle, 1)
        S = dolfin.TensorFunctionSpace(self.mesh, strain_element,degree=2)
        eps_p = dolfin.project(eps, S)


        ndim =self.ndim
        average_strains=np.array([])
        X=np.array(['x','y'])  # pour les prints

        for i in range(ndim):
            for j in range(ndim):
                if i!=0 and j!=1 :
                    rien=0 #car symetrie
                else :
                    average_strain =(dolfin.assemble(eps_p[i,j]*dx))/dolfin.assemble(1*dx)
                    if choix_print:
                        titre = r"ε_{"+str(X[i])+""+str(X[j])+"}"
                        print("  la deformation moyenne suivant  "+titre+ " est < e"+str(i+1)+str(j+1)+">=",average_strain)

                    average_strains = np.append(average_strains,average_strain)

        return average_strains

    def average_stress(self,dx,choix_print=True):
        '''
        calcul des contraintes moyennes
            Parameters
            ----------
            dx : TYPE dolfin.mesure
                 domaine d'intégration

            choix_print: TYPE booléen
                si True : return des valeurs et print des phrases pour chaque composantes
                si False: return juste les valeurs sans phrases

            Returns
            -------
            un array contenant  une valeur  moyenne pour chaque  composantes

        '''
        sig = self.stress()
        stress_element = dolfin.FiniteElement('DG', dolfin.triangle, 1)
        S = dolfin.TensorFunctionSpace(self.mesh, stress_element,degree=2)

        sig_p = dolfin.project(sig, S)
        ndim =self.ndim
        average_stress=np.array([])
        X=np.array(['x','y'])     # pour le print
        for i in range(ndim):
            for j in range(ndim):
                if i!=0 and j!=1 :
                    rien=0 #car symetrie
                else :
                    average_stre =(dolfin.assemble(sig_p[i,j]*dx))/dolfin.assemble(1*dx)
                    if choix_print:
                        titre = r"σ_{"+str(X[i])+""+str(X[j])+"}"
                        print('  la contrainte moyenne suivant '+titre+ " est < sig"+str(i+1)+str(j+1)+">=",average_stre)
                    average_stress = np.append(average_stress,average_stre)

        return average_stress

    def diviation_strains(self,dx):

        eps = self.strain()
        #projection :
        strain_element = dolfin.FiniteElement('DG', dolfin.triangle, 1)
        S = dolfin.TensorFunctionSpace(self.mesh, strain_element,degree=2)
        eps_p = dolfin.project(eps, S)

        average_strains = self.average_strain(dx,False)
        deviation=np.array([])
        k=0
        X=np.array(['x','y'])
        ndim=self.ndim
        for i in range(ndim):
            for j in range(ndim):
                if i!=0 and j!=1 :
                    rien=0 #car symetrie
                else :
                    titre = r"ε_{"+str(X[i])+""+str(X[j])+"}"
                    dev =(dolfin.assemble(abs(eps_p[i,j]-average_strains[k])*dx))/dolfin.assemble(average_strains[k]*dx)
                    deviation = np.append(deviation,dev)
                    print('  la deviation  suivant '+titre+ " est =",dev)
                    k=k+1

        return deviation

    def diviation_stress(self,dx):
        sig = self.stress()
        #projection :
        stress_element = dolfin.FiniteElement('DG', dolfin.triangle, 1)
        S = dolfin.TensorFunctionSpace(self.mesh, stress_element,degree=2)
        sig_p = dolfin.project(sig, S)

        average_stress = self.average_stress(dx,False)

        deviation=np.array([])
        k=0
        X=np.array(['x','y'])
        ndim=self.ndim
        for i in range(ndim):
            for j in range(ndim):
                if i!=0 and j!=1 :
                    rien=0 #car symetrie
                else :
                    titre = r"σ_{"+str(X[i])+""+str(X[j])+"}"
                    dev =(dolfin.assemble(abs(sig_p[i,j]-average_stress[k])*dx))/dolfin.assemble(average_stress[k]*dx)
                    deviation = np.append(deviation,dev)
                    print('  la deviation  suivant '+titre+ " est =",dev)
                    k=k+1

        return deviation

    def energie_interne(self,dx):
        return df.assemble((ufl.inner(self.strain(),self.stress()))*dx)
