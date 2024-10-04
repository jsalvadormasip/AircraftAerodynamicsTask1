import numpy as np
import matplotlib.pyplot as plt
from CalculatingAirfoilLine import z as airfoil_line, x as airfoil_x, chord as airfoil_chord

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
npanels = 10 
rho = 1.225 # kg/m^3 air density
Q_inf = 100 # m/s free stream velocity
Cl_array = []
alpha_array = np.linspace(-10*np.pi/180,10*np.pi/180,21)
for i in alpha_array:
    alpha = i # angle of attack
    free_stream_velocity = np.array([Q_inf*np.cos(alpha), Q_inf*np.sin(alpha)])

    airfoil_points = np.column_stack((airfoil_x, airfoil_line))
    discretized_indices = np.linspace(0, 100, npanels+1, dtype=int)
    print(discretized_indices)
    discretized_airfoil_line = airfoil_line[discretized_indices]
    discretized_airfoil_x = airfoil_x[discretized_indices]
    print(discretized_airfoil_line)
    print(discretized_airfoil_x)



    vortex_points_array = point_in_between_points(discretized_airfoil_x[:-1], discretized_airfoil_line[:-1], discretized_airfoil_x[1:], discretized_airfoil_line[1:], 0.25)
    control_points_array = point_in_between_points(discretized_airfoil_x[:-1], discretized_airfoil_line[:-1], discretized_airfoil_x[1:], discretized_airfoil_line[1:], 0.75)
    vortex_points = np.column_stack((vortex_points_array[0], vortex_points_array[1]))
    control_points = np.column_stack((control_points_array[0], control_points_array[1]))

    control_points_indices = []
    for i in range(npanels):
        control_points_indices.append(closest_point(airfoil_points,control_points[i]))
    control_points_indices = np.array(control_points_indices)

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
    for i in range(control_points.shape[0]):
        for j in range(vortex_points.shape[0]):
            a_matrix[i,j] = vor2d(1, control_points[i,0], control_points[i,1], vortex_points[j,0], vortex_points[j,1])[0]*normal_vectors[i,0] + vor2d(1, control_points[i,0], control_points[i,1], vortex_points[j,0], vortex_points[j,1])[1]*normal_vectors[i,1]
        RHS_matrix[i] = -np.dot(free_stream_velocity, normal_vectors[i])


    vortex_strengths = np.linalg.solve(a_matrix, RHS_matrix)
    print("vortex strengths")
    print(vortex_strengths)

    delta_lift_matrix = rho*Q_inf*vortex_strengths
    delta_p_matrix = delta_lift_matrix/np.linalg.norm(control_points[1]-control_points[0])
    lift = np.sum(delta_lift_matrix)
    Moment_LE = np.sum(delta_lift_matrix*vortex_points[:,0]*np.cos(alpha))
    Cl = lift/(0.5*rho*airfoil_chord*Q_inf**2)
    Cl_array.append(Cl)
    Cm0 = Moment_LE/(0.5*rho*airfoil_chord**2*Q_inf**2)
    print(Cl)
Cl_alpha = (Cl_array[1]-Cl_array[0])/(alpha_array[1]-alpha_array[0])
print(Cl_alpha)
plt.plot(alpha_array, Cl_array)

plt.show()

# plt.plot(airfoil_x, airfoil_line)
plt.plot(discretized_airfoil_x, discretized_airfoil_line, 'ro-')
plt.plot(vortex_points[:,0], vortex_points[:,1], 'bo')
plt.plot(control_points[:,0], control_points[:,1], 'go')
# Plot normal vectors at control points
plt.quiver(control_points[:,0], control_points[:,1], normal_vectors[:,0], normal_vectors[:,1], color='r', scale=10)

# Plot tangential vectors at control points
# plt.quiver(control_points[:,0], control_points[:,1], tangential_vectors[:,0], tangential_vectors[:,1], color='b', scale=10)

plt.axis('equal')
plt.show()
