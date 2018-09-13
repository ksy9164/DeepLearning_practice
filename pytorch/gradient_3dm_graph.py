#  We assume that it is Quadratic Equation => y^ = ax^3 + bx^2 + cx + d
#  however it is not Quadratic Equation => y = ax
import numpy as np
import matplotlib.pyplot as plt
import random
import time
x_data = np.arange(1, 10, 1)
y_data = 2 * x_data + 1

#  print(y_data)

#we guess y = a*x^3 + b*x^2 + cx +d

def forward(x) :
    return a*(x**3) + b*(x**2) + c*x + d

def loss(y, y_pre) :
    return (y_pre - y) * (y_pre - y)

def grad_norm(a, b, c, d, x, y) :
    return a*x**3 + b*x**2 + c*x + d - y

def grad_a(a, b, c, d, x, y) :
    return 2*(x**3)*grad_norm(a, b, c, d, x, y)
def grad_b(a, b, c, d, x, y) :
    return 2*(x**2)*grad_norm(a, b, c, d, x, y)
def grad_c(a, b, c, d, x, y) :
    return 2*x*grad_norm(a, b, c, d, x, y)
def grad_d(a, b, c, d, x, y) :
    return 2*grad_norm(a, b, c, d, x, y)

a = random.randrange(0,5)
b = random.randrange(0,5)
c = random.randrange(0,5)
d = random.randrange(0,5)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()
fig.show()
fig.canvas.draw()

for i in range(0,100000) :
    for x,y in zip(x_data, y_data) :
        ga = grad_a(a,b,c,d,x,y)
        gb = grad_b(a,b,c,d,x,y)
        gc = grad_c(a,b,c,d,x,y)
        gd = grad_d(a,b,c,d,x,y)
        a = a - 0.000001 * ga
        b = b - 0.000001 * gb
        c = c - 0.000001 * gc
        d = d - 0.000001 * gd
    
    if (i % 100 == 0) :
        guess_y = []
        for j in x_data :
            guess_y.append(forward(j))
        ax.clear()
        ax.plot(x_data,y_data)
        ax.plot(x_data,guess_y)
        fig.canvas.draw()

    if (i % 1000 == 0) :
        y_h = forward(x)
        l = loss(y, y_h)
        print("x = ",x,"y = ",y,"y_h = ",y_h,"\n")
        print("a = ",a,"b = ",b,"c = ",c, "d = ",d ,"loss = ",l,"\n")
fig.canvas.draw_idle()

