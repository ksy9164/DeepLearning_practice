#  We assume that it is Quadratic Equation => y^ = ax^2 + bx + c
#  however it is not Quadratic Equation => y = ax
import numpy as np
import matplotlib.pyplot as plt
import random

x_data = np.arange(1, 10, 0.1)
y_data = 2 * x_data

#  print(y_data)

#we guess y = a*x^2 + b*x + c

def forward(x) :
    return a * x * x + b * x + c

def loss(y, y_pre) :
    return (y_pre - y) * (y_pre - y)

def grad_a(a, b, c, x, y) :
    return 2 * x * x * ((a * x * x) + (b * x) + c - y)

def grad_b(a, b, c, x, y) :
    return 2 * x * ((a * x * x) + (b * x) + c - y)

def grad_c(a, b, c, x, y) :
    return 2 * ((a * x * x) + (b * x) + c - y)

a = random.randrange(1,10)
b = random.randrange(1,10)
c = random.randrange(1,10)

#1000 epoch
q = forward(3)
ql = loss(7,q)
for i in range(0,10000) :
    for x,y in zip(x_data, y_data) :
        ga = grad_a(a,b,c,x,y)
        gb = grad_b(a,b,c,x,y)
        gc = grad_c(a,b,c,x,y)
        a = a - 0.0001 * ga
        b = b - 0.0001 * gb
        c = c - 0.0001 * gc
    if (i % 500 == 0) :
        y_h = forward(x)
        l = loss(y, y_h)
        print("x = ",x,"y = ",y,"y_h = ",y_h,"\n")
        print("a = ",a,"b = ",b,"c = ",c,"loss = ",l,"\n")
print("first l is ",ql)
