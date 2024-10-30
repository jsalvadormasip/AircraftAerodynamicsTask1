import numpy as np
import matplotlib.pyplot as plt
#now, from the previous code lets get the chord length, aifoil ilne and x coordinates
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
def centered_difference(x,y): #this is used to get the derivative of the airfoil line
    dydx = np.zeros_like(y)
    h = x[1]-x[0]

    # Use centered difference for interior points 1:-1 bc it goes from second point until last point but not including the last point. 
    dydx[1:-1] = (y[2:] - y[:-2]) / (2 * h)
    
    # Use forward/backward difference for the first/last point
    dydx[0] = (y[1] - y[0]) / h
    dydx[-1] = (y[-1] - y[-2]) / h
    return dydx
def point_in_between_points(x1,y1,x2,y2,ratio): #this is used to get the vortex and control points positions
    x = x1 + ratio*(x2-x1)
    y = y1 + ratio*(y2-y1)
    return x, y
def closest_point(array, point): #this is used to get the closest point in the aifoil line to the control point
    #used to calculate the normal at each control point. 
    # Calculate the distance between each point in the array and the target point
    distances = np.linalg.norm(array - point, axis=1)
    
    # Find the index of the minimum distance
    min_index = np.argmin(distances)
    
    # Get the closest point
    closest = array[min_index]
    
    return min_index


#parameters
numberofpanels = np.array([1.8**i for i in range(1,8)]) # when we want to plot the convergence of the number of panels we use this array
# numberofpanels = np.array([100])  #otherwise we use this one.
numberofpanels = np.round(numberofpanels)
numberofpanels = numberofpanels.astype(int)

rho = 1.225 # kg/m^3 air density
Q_inf = 100 # m/s free stream velocity
slopes = [] #initiate an array to store the slopes of the Cl vs alpha curve
alpha_array = np.linspace(-10*np.pi/180,10*np.pi/180,21) # angle of attack array

for ix in numberofpanels: #loop over the different number of panels
    npanels = ix
    Cl_array = [] #initiate CL array
    Cm_array = [] #initiate Cm array
    for i in alpha_array: #loop over the different angles of attack
        alpha = i # angle of attack
        free_stream_velocity = np.array([Q_inf*np.cos(alpha), Q_inf*np.sin(alpha)]) #the free stream velocity vector

        airfoil_points = np.column_stack((airfoil_x, airfoil_line)) #the airfoil points imported from the other code. 
        
        discretized_indices = np.linspace(0, 100, npanels+1, dtype=int) #discretize the airfoil by selecting equally spaced indices of the airfoil line array. the 100 is set bc the original airfoil_line has 101 points. 
     
        discretized_airfoil_line = airfoil_line[discretized_indices] # get the z coordinates of the discretized airfoil
        discretized_airfoil_x = airfoil_x[discretized_indices] # same for the x coordinates
        



        vortex_points_array = point_in_between_points(discretized_airfoil_x[:-1], discretized_airfoil_line[:-1], discretized_airfoil_x[1:], discretized_airfoil_line[1:], 0.25) #get the vortex points position
        control_points_array = point_in_between_points(discretized_airfoil_x[:-1], discretized_airfoil_line[:-1], discretized_airfoil_x[1:], discretized_airfoil_line[1:], 0.75) #get the control points position
        vortex_points = np.column_stack((vortex_points_array[0], vortex_points_array[1])) #reformating arrays
        control_points = np.column_stack((control_points_array[0], control_points_array[1]))


        #the normal vectors will be needed to calculate the normal component of the flow in each control point
        control_points_indices = []
        for iy in range(npanels): #get the indices in the original airfoil line of the points closest to the control points 
            control_points_indices.append(closest_point(airfoil_points,control_points[iy]))
        control_points_indices = np.array(control_points_indices)

        #get the normal vectors at each control point
        detadx = centered_difference(airfoil_x, airfoil_line) #get derivative of the airfoil line
        detadx = detadx[control_points_indices]#get it only at the control points

        normal_vectors_0 = -detadx/np.sqrt(1+detadx**2) #calculate the normal vectors
        normal_vectors_1 = 1/np.sqrt(1+detadx**2)
        normal_vectors = np.column_stack((normal_vectors_0, normal_vectors_1))

        #calculate the tangential vectors
        tangential_vectors0 = normal_vectors_1
        tangential_vectors1 = -normal_vectors_0
        tangential_vectors = np.column_stack((tangential_vectors0, tangential_vectors1))

        #now, we will define the matrix and the right hand side of the equation to solve for the vortex strengths
        a_matrix = np.zeros((control_points.shape[0], vortex_points.shape[0]))
        RHS_matrix = np.zeros((control_points.shape[0], 1))
        for ix in range(control_points.shape[0]):
            for j in range(vortex_points.shape[0]):
                a_matrix[ix,j] = vor2d(1, control_points[ix,0], control_points[ix,1], vortex_points[j,0], vortex_points[j,1])[0]*normal_vectors[ix,0] + vor2d(1, control_points[ix,0], control_points[ix,1], vortex_points[j,0], vortex_points[j,1])[1]*normal_vectors[ix,1]
            RHS_matrix[ix] = -np.dot(free_stream_velocity, normal_vectors[ix])


        vortex_strengths = np.linalg.solve(a_matrix, RHS_matrix) #solve the system of equations to get the vortex strengths

        delta_lift_matrix = rho*Q_inf*vortex_strengths 
        # delta_p_matrix = delta_lift_matrix/np.linalg.norm(control_points[1]-control_points[0])
        lift = np.sum(delta_lift_matrix) #calculate the lift with the contributions from all vortices
        
        #now we will calculate the moment around the quarter chord (sorry the variable name is incorrect)
        Moment_LE = 0
        Moment_LE = delta_lift_matrix.reshape(delta_lift_matrix.shape[0])*(vortex_points[:,0]-0.25*airfoil_chord)*np.cos(alpha)
        Moment_LE = np.sum(Moment_LE)
        Moment_LE = -Moment_LE
        
        Cl = lift/(0.5*rho*airfoil_chord*Q_inf**2) #calculate lift coefficient and store it
        Cl_array.append(Cl)

        Cm_le = Moment_LE/(0.5*rho*airfoil_chord**2*Q_inf**2) #calculate the pitching moment coefficient at quarter chord, and store it. 
        Cm_array.append(Cm_le)

        Cp = 2*vortex_strengths/(Q_inf) #get the cp distribution
        
        if i == 0: #store the Cp distribution for alpha = 0
            Cp_array= Cp
            print("Cp_array", Cp_array)

        
    Cl_array = np.array(Cl_array)
    Cm_array = np.array(Cm_array)
    slope, intercept, r_value, p_value, std_err = linregress(alpha_array, Cl_array) #get the slope of the Cl vs alpha curve for all the number of panels
    slopes.append(slope) #store the slope
    
#now, let's plot everything. 
plt.plot(alpha_array*180/np.pi, Cl_array)
print(alpha_array*180/np.pi)
print(Cl_array)
plt.title("Cl vs alpha")
plt.show()

#saving it to a txt file for further analysis in excel
Cl_array_alpha = np.column_stack((alpha_array*180/np.pi, Cl_array))
Cl_array_alpha_moment = np.column_stack((Cl_array_alpha, Cm_array))
np.savetxt("Cl_array.txt", Cl_array_alpha_moment)
Cp_array = np.column_stack((vortex_points[:,0], Cp_array))
np.savetxt("Cp_0", Cp_array)


plt.plot(alpha_array*180/np.pi, Cm_array)
print(alpha_array*180/np.pi)
print(Cm_array)
plt.title("Cm vs alpha")
plt.show()

plt.plot(Cp_array[:,0], Cp_array[:,1])
plt.title('Cp vs x/c')
plt.show()


#plot the airfoil, vortex and control points
plt.plot(discretized_airfoil_x, discretized_airfoil_line, 'ro-')
plt.plot(vortex_points[:,0], vortex_points[:,1], 'bo')
plt.plot(control_points[:,0], control_points[:,1], 'go')
# Plot normal vectors at control points
plt.quiver(control_points[:,0], control_points[:,1], normal_vectors[:,0], normal_vectors[:,1], color='r', scale=10)

# Plot tangential vectors at control points
twopi = np.ones(numberofpanels.shape[0])*2*np.pi #plot the 2pi line, which is the ideal slope of CLa. 
plt.axis('equal')
plt.title('Vortex and Control Points')
plt.show()
plt.plot(numberofpanels, slopes)
plt.plot(numberofpanels, twopi, 'r--')
plt.xlabel("Number of panels", fontsize = 18)
plt.ylabel(r"$\frac{dC_l}{d\alpha}$", fontsize = 20)
plt.title("Convergence of the lift slope with increasing number of panels")
plt.show()

print("The convergence rate of the number of panels is quite fast, as the slope of the Cl vs alpha curve is converging to a value of slightly less than 2pi as soon as 10 panels. That being said, it could be argued that even with 2 panels, the results are satisfactory enough, as the result will only vary by 0.0005 if 30 times more panels are used.  ")