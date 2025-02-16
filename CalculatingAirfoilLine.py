import numpy as np
import matplotlib.pyplot as plt

student_number = 5533449   #based on the student number we can know the airfoil
last_three_digits = student_number % 1000
sum_of_digits = sum([int(digit) for digit in str(last_three_digits)])

if sum_of_digits == 0 or sum_of_digits == 1:
    airfoil = 'NACA2424'
elif sum_of_digits == 2 or sum_of_digits == 3:
    airfoil = 'NACA2415'
elif sum_of_digits == 4 or sum_of_digits == 5:
    airfoil = 'NACA4415'
elif sum_of_digits == 6 or sum_of_digits == 7:
    airfoil = 'NACA2408'
elif sum_of_digits == 8 or sum_of_digits == 9:
    airfoil = 'NACA4421'
elif sum_of_digits == 10 or sum_of_digits == 11:
    airfoil = 'NACA2412'
elif sum_of_digits == 12 or sum_of_digits == 13:
    airfoil = 'NACA1412'
elif sum_of_digits == 14 or sum_of_digits == 15:
    airfoil = 'NACA4418'
elif sum_of_digits == 16 or sum_of_digits == 17:
    airfoil = 'NACA2421'
elif sum_of_digits == 18 or sum_of_digits == 19:
    airfoil = 'NACA1408'
elif sum_of_digits == 20 or sum_of_digits == 21:
    airfoil = 'NACA4412'
elif sum_of_digits == 22 or sum_of_digits == 23:
    airfoil = 'NACA1410'
elif sum_of_digits == 24 or sum_of_digits == 25:
    airfoil = 'NACA2418'
elif sum_of_digits == 26 or sum_of_digits == 27:
    airfoil = 'NACA2410'
else:
    airfoil = 'Unknown Airfoil'


m = airfoil[4] # gets the max camber from the airfoil name
m = float(m)
m/=100 
# m *= 0.5  # changing the max camber
p = airfoil[5] #gets the position of max camber from airfoil name
p=float(p)
p/=10
# p*=2 # changing the position of max camber
chord = 1
x = np.linspace(0, chord, 101) #define an array of the x coordinate

z1 = m/p**2*(2*p*x[:int(p*100)+1]-x[:int(p*100)+1]**2) #first part of the given equation
z2 = m/((1-p)**2)*((1-2*p)+2*p*x[int(p*100)+1:]-x[int(p*100)+1:]**2) #second part of the given equation
z = np.concatenate([z1, z2])

if __name__ == '__main__':
    print("sum of three last digits",sum_of_digits)
    print(x[:int(p*100)+1])
    print(x[int(p*100)+1:   ])
    plt.plot(x, z)
    plt.axis('equal')
    plt.show()
