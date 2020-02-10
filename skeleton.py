# Skeleton code starting point for Yr 1 Theory Group Project

from vpython import *
import math
import numpy as np
import copy as copy
dt = 1/5856 # timestep

#  Define the star, planets and constants
M = 1 # mass of star (G == 1)
m1 = 3e-6 # mass of planet 1
G = 4*(math.pi)**2 #Value of G, assuming radius is 1AU, time is 1 year, and the mass of the planet is 1
monthlength = 20 #Sets the amount of time between areas between calculated 
initpos1 = vector(0,0.5,0) # initial position vector of Planet1
Planet1 = sphere(pos=initpos1,radius=0.05*m1,color=color.blue)#Initialises the planet
Star = sphere(pos=vector(0,0,0),radius=0.1,color=color.yellow)#Initialises the star
vel1 = -vector(2*(math.pi), 0, 0) # initial velocity of planet 1
trace = curve(radius = 0.005, color = color.white)#Creates a curve which can be drawn
posrad = initpos1 #sets the initial vector as the starting position
areatotal = 0
count = 0
standarddeviation = list()

def angle(r1,r2):
    angle1 = acos(dot(r1,r2) / ( (mag(r1)) * (mag(r2)) ) ) #Calculates the angle between the two position vectors
    return angle1
    
def area(r1, r2, theta):
    area1 = (r1*r2*theta)/2 #Calculates the area
    #print(area1) 
    return area1
    

for step in range(2928):

    # slow down the animation
    #print (Planet1.pos)
    #rate(100)
    # calculate changes in velocities
    denom1M = mag(Planet1.pos) ** 3 #Calculates the denominator for change in velocity
    dv1M = G * dt * Planet1.pos * M / denom1M #Calculates the change in velocity
    
    # update velocities
    vel1 = vel1 - dv1M 
    
    # update positions
    Planet1.pos = Planet1.pos + vel1 * dt 
    
    trace.append(Planet1.pos) #Traces the planet and draws the curve
       
    if step%(monthlength) == 0 and step != 0: #Ignores the first position
        count = count + 1
        angle2 = angle(posrad, Planet1.pos) #Calls the function to calculate the angle between the vectors
        areaelement = area(mag(posrad), mag(Planet1.pos),angle2)#Calculates the area between the vectors
        lineplanet = curve(vector(Planet1.pos), vector(Star.pos)) #Draws lines from the star to the planet
        posrad = copy.copy(Planet1.pos) #Changes the value of the planet vector
        areatotal = areatotal + areaelement
        standarddeviation.append(areaelement)#Adds on the area elements to a total


print(areatotal/count)#Prints the average area
print(np.std(np.array(standarddeviation))) #Calculates the average area of the sectors

