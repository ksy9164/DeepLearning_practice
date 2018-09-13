# y = ax + b 
# gradient descent

import random
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = random.randrange(1,10000)

t = len(x_data)
def gradient (x,y,w):
    return 2 * x * (w * x - y)

for i in range(10000) :
    for x, y in zip(x_data, y_data) :
        g = gradient(x,y,w)
        w -= w * 0.00001 * g
    if (i % 100 == 0) :
        print(" w = ",w," g = ",g)

