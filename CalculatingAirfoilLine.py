import numpy as np
import matplotlib.pyplot as plt

student_number = 5533449
last_three_digits = student_number % 1000
sum_of_digits = sum([int(digit) for digit in str(last_three_digits)])
print(sum_of_digits)

airfoil = 'NACA2421'
m = airfoil[4]
m = float(m)
m/=100
p = airfoil[5]
p=float(p)
p/=10
x = np.linspace(0, 1, 101)
print(x[:int(p*100)+1])
print(x[int(p*100)+1:   ])
z1 = m/p**2*(2*p*x[:int(p*100)+1]-x[:int(p*100)+1]**2)
z2 = m/((1-p)**2)*((1-2*p)+2*p*x[int(p*100)+1:]-x[int(p*100)+1:]**2)
z = np.concatenate([z1, z2])

plt.plot(x, z)
plt.axis('equal')
plt.show()
