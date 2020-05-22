import scipy.linalg as nla
import scipy.interpolate #Contient des splines 
import numpy as np
import matplotlib.pyplot as plt

x= [0., 1.7, 3.3, 5.0, 6.7, 8.4, 10.1]
y= [1.7, 3.4, 0.8, 2.5, 4.3, 4.3, 0.8]

xnew = np.linspace(0, 10, num=100, endpoint=True)
s=scipy.interpolate.interp1d(x, y)
s2=scipy.interpolate.interp1d(x, y, kind="cubic")

#plt.plot(x, y, 'o', xnew, s(xnew), '-', xnew, s2(xnew), "--")
#plt.show()


#On considère que si on a n+1 points on a n sous intervalles. 
# ici on aura donc 6 sous intervalles.

# on créée donc nos 4 vecteurs de taille n
n=6
p=1

#Création de la matrice Epsilon. Par défaut c'est une diagonale de 1.
E = np.zeros((n+1,n+1))
for i in range (0,n+1) :
    E[i][i]=1
print("E=",E)

#Création du vecteur h (
h = np.zeros(n)
for i in range (0,n) :
    h[i]=x[i+1]-x[i]
print("h=",h)
print(h.T)
#Création de la matrice T
T = np.zeros((n-1,n-1))
for i in range (0,n-2) : 
    T[i][i] = 2*(h[i]+h[i+1])
    T[i+1][i]=h[i+1]
    T[i][i+1]=h[i+1]
T[n-2][n-2]=2*(h[n-2]+h[n-1])
T=1/3*T
print("T=",T)

#Création du vecteur g
g = np.zeros(6)
for i in range (0,n) :
    g[i]=1/h[i]

#Création de la matrice Q
Q = np.zeros((n+1,n-1))
for i in range (0,n-1) :
    Q[i][i]=g[i]
    Q[i+1][i]=-g[i]-g[i+1]
    Q[i+2][i]=g[i+1]
print("Q=",Q)

#calcul de c    
b=np.matmul(p*(Q.T),y)
c2=np.matmul(np.matmul(E,E), Q)
A= np.matmul(Q.T,c2) + p*T

print ( nla.solve(A,b, sym_pos=True) )

L = np.linalg.cholesky(A)

y_temp = nla.solve_triangular(L,b,lower=True)

c_temp = nla.solve_triangular(L.T, y_temp, lower=False)

print("c calcule=",c_temp)

c = np.zeros(n+1)
c[0]=0
c[n]=0
for i in range (1,n) : 
    c[i]=c_temp[i-1]

print("c=",c)

#calcul de a 
a1=np.matmul(E,E)
a1=np.matmul(a1, Q)
a=y-p**(-1)*np.matmul(a1,c_temp)
print("a=",a)

    
#calcul de d 
d = np.zeros(n)
d[0] = 0 ; 
for i in range (0,n) :
    d[i]=(c[i+1]-c[i]) / (3*h[i]) #n'ira pas, revoir condition c0 = cn = 0 et conditions sur les indices

print("d=",d)

#calcul de b
b = np.zeros(n)
for i in range (0,n):
    b[i]= (a[i+1]-a[i])/h[i] - c[i]*h[i] - d[i]*h[i]**2 
    
print("b=",b)    

y_new=np.zeros(n+1)


for i in range(0,n):
    X_PLOT= np.linspace(x[i], x[i+1], 10000) 
    Y_PLOT = ( a[i] + b[i] * (X_PLOT- x[i]) 
                    + c[i] * (X_PLOT- x[i])**2 
                    + d[i] * (X_PLOT- x[i])**3 )
                  
    plt.plot(X_PLOT, Y_PLOT) 

plt.show()

def f(x,x_vect,n,a,b,c,d):
    xi=x_vect[0]
    i=0
    while i<n+1 and x_vect[i+1]<x :
        i = i+1
        xi=x_vect[i]
    return a[i] + b[i]*(x-xi) + c[i]*(x-xi)**2 + d[i]*(x-xi)**3 


x_axis = np.linspace(0,12,num=1000,endpoint=True)
l = [ f(toto, x, n, a, b, c, d) for toto in x_axis ]
y_axis=np.asarray(l)
print("y_axis=",y_axis)

#plt.plot(x_axis, f(x_axis,x,n,a,b,c,d))
#plt.show()
    
