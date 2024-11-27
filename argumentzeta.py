#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 04:03:23 2024

@author: chiarabellotti
"""

import mpmath as mpm
import numpy as np

from mpmath import *
from numpy import inf
#from scipy.optimize import differential_evolution, NonlinearConstraint
from scipy.integrate import quad
from scipy.special import polygamma, factorial
from scipy.optimize import optimize, minimize, NonlinearConstraint, differential_evolution
from sympy import Symbol, solveset, S, erf, div, solve

##############

mpm.mp.dps = 10 #50

##############

def latex_float(num, we):
    strng = format(float(num),we)
    if "E" in strng:
        a, b = strng.split("E")
        return "$" + str(float(a)) + "\cdot 10^{" + str(int(b)) + "}" + "$"
    
    
    
eta=0.000158 
c=1.000225
r=1.000605
n=5

T=exp(106)
T0=30610046000
J1=64
J2=39

Q0=1
Q1=1.18
Q2=1.18
Q3=3.9
Q4=1
Q5=1
Q6=1
Q7=1
Q8=1
Q9=1
Q10=2.3
Q11=3.9

#def theta_y

def theta(y):
    if c+r<=y:
        return 0
    elif c-r< y and y<c+r:
        return acos(fdiv(y-c,r))
    elif y<=c-r:
        return pi
      
#def sigma

def sigma(z):
    return c+r*cos(z)
 
#def k1

def k1(T):
    if 0<=T and T<=3:
        return 1.461
    elif T>3 and T<= exp(105):
        return 0.618
    elif T>exp(105):
        return 66.7
    
#def k2

def k2(T):
    if 0<=T and T<=3:
        return 0
    elif T>3 and T<= exp(105):
        return fdiv(1,6)
    elif T>exp(105):
        return fdiv(27,164) 
    
#def k3

def k3(T):
    if 0<=T and T<=3:
         return 0
    elif T>3 and T<= exp(105):
         return 1
    elif T>exp(105):
         return 0   

#def sigma1 and delta Lemma 2.4

sigma1=c+fdiv(power(c-0.5,2),r)

delta= 2*c-sigma1-0.5

#def c1

def c1(T):
    if 3<=T and T<=exp(3070):
        return 1
    elif exp(3070)<T and T<= exp(3.69 *pow(10,8)):
        return 0.25
    elif T>exp(3.69 *pow(10,8)):
        return 58.096

#def c2
    
def c2(T):
    if 3<=T and T<=exp(3070):
        return 1
    elif exp(3070)<T and T<= exp(3.69 *pow(10,8)):
        return 1
    elif T>exp(3.69 *pow(10,8)):
        return fdiv(2,3)
           

#def sigma_{4+h}

sigmah=[]

for h in range(n+1):
    sigmah.append(1-fdiv((4+h),(pow(2,4+h)-2)))
print()


#DEF C1

#def integral from sigma_{4+h} to sigma_{5+h}, 0<=h<= n-1

integralh=[]

for h in range(n):
    integrallht= lambda w: (pow(2,h+5)-2)*(pow(2,h+4)-1)*(sigmah[1+h]-sigma(w))+(pow(2,h+4)-2)*(pow(2,h+5)-1)*(sigma(w)-sigmah[h])
    integrallhtt = quad(integrallht, theta(sigmah[1+h]), theta(sigmah[h]))[0]
    singleinth=fdiv(integrallhtt, (pow(2,h+5)-2)*(pow(2,h+4)-2)*(sigmah[1+h]-sigmah[h]))
    integralh.append(singleinth)

#Remaining terms

integral1t = lambda w: (pow(2,n+4)-1)*(1-sigma(w))+(pow(2,n+4)-2)*(sigma(w)-sigmah[n])
integral1 = quad(integral1t, theta(1), theta(sigmah[n]))[0]  

integral2t = lambda w: 14*(k2(T)+1)*(sigmah[0]-sigma(w))+15*(sigma(w)-0.5)
integral2 = quad(integral2t, theta(sigmah[0]), theta(0.5))[0]

integral3t = lambda w: 1-2*sigma(w)+4*k2(T)*sigma(w)
integral3 = quad(integral3t, theta(0.5), theta(0))[0]  

integral4t = lambda w: 1-2*sigma(w)
integral4 = quad(integral4t, theta(-eta), pi)[0]   

integral5t = lambda w: sigma(w)*fdiv(1+2*eta, -2*eta)+fdiv(sigma(w)+eta,2*eta)
integral5 = quad(integral5t, theta(0), theta(-eta))[0]

def C1bar(T):
    return fdiv(integral1, (pow(2,n+4)-2)*(1-sigmah[n]))+sum(integralh)+fdiv(integral2, 14*(sigmah[0]-0.5))+0.5*(integral3+integral4)+integral5+theta(1)-theta(0.5)

def finalC1(T):
    return fdiv(C1bar(T),2*pi*log(fdiv(r,c-0.5)))

print('final C1', finalC1(T)) 


#DEF C2 

int1t = lambda w: 1-sigma(w)+ c2(T)*(sigma(w)-sigmah[n])
int1 = quad(int1t, theta(1), theta(sigmah[n]))[0]

int2t = lambda w: k3(T)*(sigmah[0]-sigma(w))+(sigma(w)-0.5)
int2 = quad(int2t, theta(sigmah[0]), theta(0.5))[0]

int3t = lambda w: 1+eta-sigma(w)
int3 = quad(int3t, theta(1+eta), theta(1))[0]

int4t = lambda w: c2(T)*(1-2*sigma(w))+2*k3(T)*sigma(w)
int4 = quad(int4t, theta(0.5), theta(0))[0]

int5t = lambda w: c2(T)*fdiv(sigma(w)+eta,eta)
int5 = quad(int5t, theta(0), theta(-eta))[0]


def C2bar(T):
    return fdiv(int1, 1-sigmah[n])+theta(sigmah[0])-theta(sigmah[n])+fdiv(int2,sigmah[0]-0.5)+int3*fdiv(c2(T),eta)+int4+int5

def finalC2(T):
    return fdiv(C2bar(T),2*pi*log(fdiv(r,c-0.5)))

print('final C2', finalC2(T)) 


#DEF C2prime

def finalC2prime(T):
    return fdiv(C2bar(T)+1.2322*pi,2*pi*log(fdiv(r,c-0.5)))

print('final C2prime', finalC2prime(T)) 


#def Lstar

def Lstar(z,j):
    return fdiv(power(j+c+r*cos(z),2), T0)+fdiv(power(r*sin(z),2), T0)+2*r*sin(z)


#DEF C3

#first row C3

def r1C3(T):
    return fdiv(7,8)+fdiv(1,4)+fdiv(1,50*T0)+fdiv(log(zeta(sigma1)),pi)+fdiv(1,2*log(fdiv(r,c-0.5)))*log(fdiv(zeta(c),zeta(2*c)))+0.5*(fdiv(640*delta-112,1536*(3*T0-1))+fdiv(1,power(2,10)))
 
 #D3
  
int1r1 = lambda w: 1+eta-sigma(w)
int11r = quad(int1r1, theta(1+eta), theta(1))[0]  

int1r2 = lambda w: sigma(w)-1
int12r = quad(int1r2, theta(1+eta), theta(1))[0]  

def r1D3(T):
    return fdiv(int11r*log(c1(T)), eta)+fdiv(int12r*log(zeta(1+eta)),eta)


int3r1 = lambda w: 1-sigma(w)
int31r = quad(int3r1, theta(1), theta(sigmah[n]))[0]

int3r2 = lambda w: sigma(w)-sigmah[n]
int32r = quad(int3r2, theta(1), theta(sigmah[n]))[0]
 
def r2D3(T):

    return fdiv(log(1.546)*int31r,1-sigmah[n])+fdiv(log(c1(T))*int32r,1-sigmah[n])


int5r1 = lambda w: sigmah[0]-sigma(w)
int51r = quad(int5r1, theta(sigmah[0]), theta(0.5))[0]

int5r2 = lambda w: sigma(w)-0.5
int52r = quad(int5r2, theta(sigmah[0]), theta(0.5))[0]

def r3D3(T):
    return log(1.546)*(theta(sigmah[0])-theta(sigmah[n]))+fdiv(log(k1(T))*int51r,sigmah[0]-0.5)+fdiv(log(1.546)*int52r,sigmah[0]-0.5)


int6r1 = lambda w: 1-2*sigma(w)
int61r = quad(int6r1, theta(0.5), theta(0))[0]

int6r2 = lambda w: sigma(w)
int62r = quad(int6r2, theta(0.5), theta(0))[0]

def r4D3(T):
    return log(fdiv(c1(T), power(2*pi,0.5)))*int61r+int62r*2*log(k1(T))


int7r1 = lambda w: -fdiv(sigma(w),eta)*log(fdiv(1+eta,c1(T)*power(2*pi,eta)))+log(fdiv(c1(T), power(2*pi,0.5)))
int71r = quad(int7r1, theta(0), theta(-eta))[0]

int7r2 = lambda w: 1-2*sigma(w)
int72r = quad(int7r2,  theta(-eta), pi)[0]

def r5D3(T):
    return int71r-fdiv(log(2*pi),2)*int72r

def r6D3(T):
    return fdiv(log(zeta(1+eta))+log(zeta(c)),2)*(theta(1+eta)-fdiv(pi,2))+fdiv(pi,4*J1)*log(zeta(c))

def r7D3(T):
    return fdiv(log(zeta(1+eta))+log(zeta(c)),2)*(theta(1-c)-theta(-eta))+fdiv(pi-theta(1-c),2*J2)*log(zeta(c))

def D3(T):
    return r1D3(T)+r2D3(T)+r3D3(T)+r4D3(T)+r5D3(T)+r6D3(T)+r7D3(T)


#def kappa1

_sum = 0
for j in range(1, J1 ):
    _sum += log(zeta(c+r*cos(fdiv(pi*j,2*J1))))
    
kappa1= fdiv(pi,4*J1)*(log(zeta(c+r))+2*_sum)


#def kappa2

_sum2 = 0
for j in range(1, J2 ):
    _sum2 += log(zeta(1-c-r*cos(fdiv(pi*j,J2) +theta(1-c)*(1-fdiv(j,J2)))))
    
kappa2= fdiv(pi-theta(1-c),2*J2)*(log(zeta(1-c+r))+2*_sum2)


#def kappa3

int1k3 = lambda w: Lstar(w,-1)
int1k3f= quad(int1k3, 0, theta(1+eta))[0]

int2k3 = lambda w: Lstar(w,Q0)
int2k3f= quad(int2k3, theta(1+eta), theta(1))[0]

int3k3 = lambda w: ((power(2,n+4)-1)*(1-sigma(w))+(power(2,n+4)-2)*(sigma(w)-sigmah[n]))*Lstar(w,max(Q0,Q9))
int3k3f= quad(int3k3, theta(1), theta(sigmah[n]))[0]

integralhk3=[]

for h in range(n):
    integrallhtk3= lambda w: ((pow(2,h+5)-2)*(pow(2,h+4)-1)*(sigmah[1+h]-sigma(w))+(pow(2,h+4)-2)*(pow(2,h+5)-1)*(sigma(w)-sigmah[h]))*Lstar(w,1)
    integrallhtk3f = quad(integrallhtk3, theta(sigmah[1+h]), theta(sigmah[h]))[0]
    singleinthk3=fdiv(integrallhtk3f, (pow(2,h+5)-2)*(pow(2,h+4)-2)*(sigmah[1+h]-sigmah[h]))
    integralhk3.append(singleinthk3)
 
int4k3 = lambda w: (14*(k2(T)+1)*(sigmah[0]-sigma(w))+15*(sigma(w)-0.5))*Lstar(w,Q2)
int4k3f= quad(int4k3, theta(sigmah[0]), theta(0.5))[0]

int5k3 = lambda w: Lstar(w,-1)
int5k3f= quad(int5k3, theta(0.5), theta(0))[0]

int6k3 = lambda w: (1-2*sigma(w)+4*k2(T)*sigma(w))*Lstar(w,Q11)
int6k3f= quad(int6k3, theta(0.5), theta(0))[0]

int7k3 = lambda w: Lstar(w,-1)
int7k3f= quad(int7k3, theta(0), theta(-eta))[0]

int8k3 = lambda w: (-sigma(w)+0.5)*Lstar(w,Q10)
int8k3f= quad(int8k3, theta(0), theta(-eta))[0]

int9k3 = lambda w: Lstar(w,-1)
int9k3f= quad(int9k3, theta(-eta), pi)[0]

int10k3 = lambda w: (0.5*(1-2*sigma(w)))*Lstar(w,1)
int10k3f= quad(int10k3, theta(-eta), pi)[0]

M1=int1k3f + int2k3f + fdiv(1,(power(2,n+4)-2)+(1-sigmah[n])) * int3k3f + sum(integralhk3) + fdiv(int4k3f,14*(sigmah[0]-0.5))+ int5k3f + 0.5*int6k3f + int7k3f + int8k3f + int9k3f + int10k3f

kappa31=fdiv(1,2*T0)*max(0,M1)


int1k32 = lambda w: fdiv(c2(T),eta)*(1+eta-sigma(w))*Lstar(w,Q0)
int1k32f= quad(int1k32, theta(1+eta), theta(1))[0]

int2k32 = lambda w: ((1-sigma(w))+c2(T)*(sigma(w)-sigmah[n]))*Lstar(w,max(Q0,Q9))
int2k32f= quad(int2k32, theta(1), theta(sigmah[n]))[0]

integralhk32=[]

for h in range(n):
    integrallhtk32= lambda w: Lstar(w,1)
    singleinthk32 = quad(integrallhtk32, theta(sigmah[1+h]), theta(sigmah[h]))[0]
    integralhk32.append(singleinthk32)
 
int3k32 = lambda w: (k3(T)*(sigmah[0]-sigma(w))+(sigma(w)-0.5))*Lstar(w,Q2)
int3k32f= quad(int3k32, theta(sigmah[0]), theta(0.5))[0]

int4k32 = lambda w: (c2(T)*(1-2*sigma(w))+2*k3(T)*sigma(w))*Lstar(w,Q11)
int4k32f= quad(int4k32, theta(0.5), theta(0))[0]

int5k32 = lambda w: c2(T)*fdiv(sigma(w)+eta,eta)*Lstar(w,Q10)
int5k32f= quad(int5k32, theta(0), theta(-eta))[0]

M2=int1k32f + fdiv(int2k32f,1-sigmah[n]) + sum(integralhk32) + fdiv(int3k32f, sigmah[0]-0.5) + int4k32f + int5k32f

kappa32=fdiv(1,2*T0*log(T0))*max(0,M2)

kappa3=kappa31+kappa32


divfin=2*pi*log(fdiv(r,c-0.5))


def finalC3(T):
    return r1C3(T)+fdiv(D3(T)+kappa1+kappa2+kappa3,divfin)

print('final C3', finalC3(T))


#DEF C3prime

def r1C3prime(T):
    return fdiv(7,8)+fdiv(1,4)+fdiv(1,50*T0)+fdiv(log(zeta(sigma1)),pi)+0.5*(fdiv(640*delta-112,1536*(3*T0-1))+fdiv(1,power(2,10)))


def finalC3prime(T):
    return r1C3prime(T)+fdiv(D3(T)+kappa1+kappa2+kappa3,divfin)

print('final C3prime', finalC3prime(T))


#DEF C3tilde

def C3T(T):
    return finalC3(T)-fdiv(7,8)+fdiv(1,pi)*(atan(fdiv(sigma1-1,T0))+atan(fdiv(1,2*T0)))

print('C3tilde: ', C3T(T))


#DEF C3tildeprime

def C3Tprime(T):
    return finalC3prime(T)-fdiv(7,8)+fdiv(1,pi)*(atan(fdiv(sigma1-1,T0))+atan(fdiv(1,2*T0)))

print('C3tildeprime: ', C3Tprime(T))


#DEF mathcalC3

def mathcalC3(T):
    
    return 2*C3T(T)+fdiv(3,4*pi)-fdiv(log(2*pi*e),2*pi)

print('mathcalC3: ', mathcalC3(T))


#DEF mathcalC3prime

def mathcalC3p(T):
    
    return 2*C3Tprime(T)+fdiv(3,4*pi)-fdiv(log(2*pi*e),2*pi)

print('mathcalC3prime: ', mathcalC3p(T))


#DEF mathcalD3

def mathcalD3(T):
    return 2*C3T(T)+fdiv(1,4*pi)+fdiv(log(3)-log(2*pi*e),pi)

print('mathcalD3: ', mathcalD3(T))


#DEF mathcalD3prime

def mathcalD3p(T):
    return 2*C3Tprime(T)+fdiv(1,4*pi)+fdiv(log(3)-log(2*pi*e),pi)

print('mathcalD3prime: ', mathcalD3p(T))



#Check conditions

def check(c,r,eta):
  if c-r <= -0.5:
    return print('false1')
 
  if c-r >= 1-c:
     return print('false2')
 
  if 1-c >= -eta:
    return print('false3')
    

  if -eta>=0:
     return print('false4')

  if -eta>=1/4:
     return print('false5')

  if 2*c-sigma1-0.5 < 1/4:
     return print('false6')

  if 2*c-sigma1-0.5 >= 0.5:
     return print('false7')
 
  if 1+eta <= 0.5:
     return print('false8')

  if 1+eta >= c:
     return print('false9')

  if sigma1 <= c:
     return print('false10')

  if sigma1 >= c+r:
     return print('false11')

  if theta(1+eta)>2.1:
     return print('false12')

check(c,r,eta) 




        
        