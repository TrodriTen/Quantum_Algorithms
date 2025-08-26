import numpy as np
from sympy.physics.quantum import TensorProduct
from sympy import Matrix

## Ejercicio 1

SN_1 = (1/2) * np.array([[complex(1,1),complex(1,1).conjugate()], [complex(1,1).conjugate(), complex(1,1)]])
SN_2 = (1/np.sqrt(complex(0,2))) * np.array([[complex(0,1), 1],[1,complex(0,1)]])

cuadrado_SN_1 = np.pow(SN_1,2)
cuadrado_SN_2 = np.pow(SN_2,2)

print(cuadrado_SN_1)
print(cuadrado_SN_2)

## Ejercicio 2

X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1*complex(0,1)],[complex(0,1),0]])
Z = np.array([[1,0],[0,-1]])

XY = np.dot(X,Y)
XYZ = np.dot(XY,Z)

if XYZ.all() == np.dot(complex(0,1), np.array([[1,0],[0,1]])).all():
    print('La multiplicacion de las 3 matrices X,Y,Z es igual a i por I')

## Ejercicio 3

cuadrado_X = np.pow(X,2)
cuadrado_Y = np.pow(Y,2)
cuadrado_Z = np.pow(Z,2)

counter = 0
similitud = ''

if cuadrado_X.all() == np.array([[1,0],[0,1]]).all():
    counter += 1
    similitud += 'X es involutoria '
if cuadrado_Y.all() == np.array([[1,0],[0,1]]).all():
    counter +=1
    similitud += 'Y es involutoria '
if cuadrado_Z.all() == np.array([[1,0],[0,1]]).all():
    counter += 1
    similitud += 'Z es involutoria '

print(similitud)

## Ejercicio 4

rta = ''

### Matriz original

H = (1/np.sqrt(2)) * np.array([[complex(1,0),complex(1,0)],[complex(1,0),complex(-1,0)]])

### Matriz Conjugada

H_conjugada = np.copy(H)
for i in range(len(H)):
    for j in range(len(H)):
        new_pos = H[i][j].conjugate()
        H_conjugada[i][j] = new_pos

### Matriz inversa

H_inversa = np.linalg.inv(H)

### Matriz elevada al cuadrado

cuadrado_H = np.pow(H,2)

if H.all() == H_conjugada.all():
    rta += 'La matriz es Hermite \n'
if np.dot(H,H_conjugada).all() == np.dot(H_conjugada,H).all():
    rta += 'La matriz es Normal \n'
if cuadrado_H.all() == np.array([[1,0],[0,1]]).all():
    rta += 'La matriz es Involutoria \n'
if H_conjugada.all() == H_inversa.all():
    rta += 'La matriz es Unitaria'

print(rta)

## Ejercicio 5
H = Matrix([[1/np.sqrt(2),1/np.sqrt(2)], [1/np.sqrt(2),-1/np.sqrt(2)]])
I = Matrix([[1,0],[0,1]])
print(TensorProduct(H,I))


