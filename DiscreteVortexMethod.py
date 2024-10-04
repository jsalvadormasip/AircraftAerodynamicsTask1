import numpy as np
import matplotlib.pyplot as plt
from CalculatingAirfoilLine import z as airfoil_line, x as airfoil_x

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
npanels = 10 
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



print(vortex_points)
print("vortexpointsarray")
print(vortex_points_array)
print(control_points)


plt.plot(airfoil_x, airfoil_line)
plt.plot(discretized_airfoil_x, discretized_airfoil_line, 'ro')
plt.plot(vortex_points[:,0], vortex_points[:,1], 'bo')
plt.plot(control_points[:,0], control_points[:,1], 'go')
# Plot normal vectors at control points
plt.quiver(control_points[:,0], control_points[:,1], normal_vectors[:,0], normal_vectors[:,1], color='r', scale=10)

# Plot tangential vectors at control points
plt.quiver(control_points[:,0], control_points[:,1], tangential_vectors[:,0], tangential_vectors[:,1], color='b', scale=10)

plt.axis('equal')
plt.show()
