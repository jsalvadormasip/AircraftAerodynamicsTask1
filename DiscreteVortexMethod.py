import numpy as np
import matplotlib.pyplot as plt
from CalculatingAirfoilLine import z as airfoil_line, x as airfoil_x, chord as airfoil_chord
from scipy.stats import linregress
def vor2d(Gammaj, x, z, xj, zj):
    # Calculate the velocity induced by a vortex element
    # at a point (x, z)
    # Gammaj: strength of the vortex element
    # x, z: coordinates of the point where the velocity is calculated
    # xj, zj: coordinates of the vortex element
    # Returns: the velocity induced by the vortex element at (x, z)
    dx = x - xj
    dz = z - zj
    r = np.sqrt(dx**2 + dz**2)
    u = Gammaj/(2*np.pi)*dz/r**2
    w = -Gammaj/(2*np.pi)*dx/r**2
    return u, w
def centered_difference(x,y):
    dydx = np.zeros_like(y)
    h = x[1]-x[0]

    # Use centered difference for interior points 1:-1 bc it goes from second point until last point but not including the last point. 
    dydx[1:-1] = (y[2:] - y[:-2]) / (2 * h)
    
    # Use forward/backward difference for the first/last point
    dydx[0] = (y[1] - y[0]) / h
    dydx[-1] = (y[-1] - y[-2]) / h
    return dydx
def point_in_between_points(x1,y1,x2,y2,ratio):
    x = x1 + ratio*(x2-x1)
    y = y1 + ratio*(y2-y1)
    return x, y
def closest_point(array, point):
    # Calculate the distance between each point in the array and the target point
    distances = np.linalg.norm(array - point, axis=1)
    
    # Find the index of the minimum distance
    min_index = np.argmin(distances)
    
    # Get the closest point
    closest = array[min_index]
    
    return min_index
#parameters

# numberofpanels = np.linspace(2,100,18, dtype=int)
numberofpanels = np.array([1.8**i for i in range(1,8)])
numberofpanels = np.hstack((numberofpanels, 1))
# numberofpanels = np.hstack((numberofpanels, 100))
# numberofpanels = np.hstack((numberofpanels, 150))
# numberofpanels = np.array([100])
numberofpanels = np.round(numberofpanels)
numberofpanels = numberofpanels.astype(int)
# numberofpanels = np.array([100])
print(numberofpanels)
rho = 1.225 # kg/m^3 air density
Q_inf = 100 # m/s free stream velocity
slopes = []
alpha_array = np.linspace(-10*np.pi/180,10*np.pi/180,21)
# alpha_array = np.array([0])
for ix in numberofpanels:
    npanels = ix
    Cl_array = []
    Cm_array = []
    for i in alpha_array:
        alpha = i # angle of attack
        free_stream_velocity = np.array([Q_inf*np.cos(alpha), Q_inf*np.sin(alpha)])

        airfoil_points = np.column_stack((airfoil_x, airfoil_line))
        discretized_indices = np.linspace(0, 100, npanels+1, dtype=int)
        # print(discretized_indices)
        discretized_airfoil_line = airfoil_line[discretized_indices]
        discretized_airfoil_x = airfoil_x[discretized_indices]
        # print(discretized_airfoil_line)
        # print(discretized_airfoil_x)



        vortex_points_array = point_in_between_points(discretized_airfoil_x[:-1], discretized_airfoil_line[:-1], discretized_airfoil_x[1:], discretized_airfoil_line[1:], 0.25)
        control_points_array = point_in_between_points(discretized_airfoil_x[:-1], discretized_airfoil_line[:-1], discretized_airfoil_x[1:], discretized_airfoil_line[1:], 0.75)
        vortex_points = np.column_stack((vortex_points_array[0], vortex_points_array[1]))
        control_points = np.column_stack((control_points_array[0], control_points_array[1]))

        control_points_indices = []
        for iy in range(npanels):
            control_points_indices.append(closest_point(airfoil_points,control_points[iy]))
        control_points_indices = np.array(control_points_indices)
        # control_points_indicess = closest_point(airfoil_points,control_points)
        # print(control_points_indices)
        # print(control_points_indicess)
        detadx = centered_difference(airfoil_x, airfoil_line)
        detadx = detadx[control_points_indices]
        normal_vectors_0 = -detadx/np.sqrt(1+detadx**2)
        normal_vectors_1 = 1/np.sqrt(1+detadx**2)
        normal_vectors = np.column_stack((normal_vectors_0, normal_vectors_1))
        tangential_vectors0 = normal_vectors_1
        tangential_vectors1 = -normal_vectors_0
        tangential_vectors = np.column_stack((tangential_vectors0, tangential_vectors1))

        a_matrix = np.zeros((control_points.shape[0], vortex_points.shape[0]))
        RHS_matrix = np.zeros((control_points.shape[0], 1))
        for ix in range(control_points.shape[0]):
            for j in range(vortex_points.shape[0]):
                a_matrix[ix,j] = vor2d(1, control_points[ix,0], control_points[ix,1], vortex_points[j,0], vortex_points[j,1])[0]*normal_vectors[ix,0] + vor2d(1, control_points[ix,0], control_points[ix,1], vortex_points[j,0], vortex_points[j,1])[1]*normal_vectors[ix,1]
            RHS_matrix[ix] = -np.dot(free_stream_velocity, normal_vectors[ix])


        vortex_strengths = np.linalg.solve(a_matrix, RHS_matrix)
        # print("vortex strengths")
        # print(vortex_strengths)

        delta_lift_matrix = rho*Q_inf*vortex_strengths
        # delta_p_matrix = delta_lift_matrix/np.linalg.norm(control_points[1]-control_points[0])
        lift = np.sum(delta_lift_matrix)
        # Moment_LE = 0
        # for ixx in range(delta_lift_matrix.shape[0]):
        #     Moment_LE += delta_lift_matrix[ixx]*(vortex_points[ixx,0]-0.25*airfoil_chord)*np.cos(alpha)
        # Moment_LE = -Moment_LE
        Moment_LE = 0
        Moment_LE = delta_lift_matrix.reshape(delta_lift_matrix.shape[0])*(vortex_points[:,0]-0.25*airfoil_chord)*np.cos(alpha)
        Moment_LE = np.sum(Moment_LE)
        Moment_LE = -Moment_LE
        print(delta_lift_matrix.reshape(delta_lift_matrix.shape[0]).shape)
        print(vortex_points[:,0].shape)
        # print(delta_lift_matrix)
        # print(vortex_points[:,0]-0.25*airfoil_chord)
        Cl = lift/(0.5*rho*airfoil_chord*Q_inf**2)
        Cl_array.append(Cl)
        Cm_le = Moment_LE/(0.5*rho*airfoil_chord**2*Q_inf**2)
        Cm_array.append(Cm_le)
        Cp = 2*vortex_strengths/(Q_inf)
        print(i)
        if i == 0:
            Cp_array= Cp
            print("Cp_array", Cp_array)

        # print(Cl)
    Cl_array = np.array(Cl_array)
    Cm_array = np.array(Cm_array)
    slope, intercept, r_value, p_value, std_err = linregress(alpha_array, Cl_array)
    slopes.append(slope)
    print("For number of panels: ", ix)
    print("Slope: ", slope)
# plt.plot( vortex_points[:,0], Cp)
# print(control_points[:,0])
# plt.show()
plt.plot(alpha_array*180/np.pi, Cl_array)
print(alpha_array*180/np.pi)
print(Cl_array)
plt.show()
Cl_array_alpha = np.column_stack((alpha_array*180/np.pi, Cl_array))
Cl_array_alpha_moment = np.column_stack((Cl_array_alpha, Cm_array))
np.savetxt("Cl_array.txt", Cl_array_alpha_moment)
Cp_array = np.column_stack((vortex_points[:,0], Cp_array))
np.savetxt("Cp_0", Cp_array)
plt.plot(alpha_array*180/np.pi, Cm_array)
print(alpha_array*180/np.pi)
print(Cm_array)
plt.show()

# plt.plot(airfoil_x, airfoil_line)
plt.plot(discretized_airfoil_x, discretized_airfoil_line, 'ro-')
plt.plot(vortex_points[:,0], vortex_points[:,1], 'bo')
plt.plot(control_points[:,0], control_points[:,1], 'go')
# Plot normal vectors at control points
plt.quiver(control_points[:,0], control_points[:,1], normal_vectors[:,0], normal_vectors[:,1], color='r', scale=10)

# Plot tangential vectors at control points
# plt.quiver(control_points[:,0], control_points[:,1], tangential_vectors[:,0], tangential_vectors[:,1], color='b', scale=10)
twopi = np.ones(numberofpanels.shape[0])*2*np.pi
plt.axis('equal')
plt.show()
plt.scatter(numberofpanels, slopes)
# plt.plot(numberofpanels, twopi, 'r--')
plt.xlabel("Number of panels", fontsize = 18)
plt.ylabel(r"$\frac{dC_l}{d\alpha}$", fontsize = 20)

plt.show()

print("The convergence rate of the number of panels is quite fast, as the slope of the Cl vs alpha curve is converging to a value of slightly less than 2pi as soon as 10 panels. That being said, it could be argued that even with 2 panels, the results are satisfactory enough, as the result will only vary by 0.0005 if 30 times more panels are used.  ")