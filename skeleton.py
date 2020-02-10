# Skeleton code starting point for Yr 1 Theory Group Project

from vpython import *

dt = 0.001 # timestep
step = 1 # loop counter
maxstep = 1000

#  Define the star, planets and constants
M = 1000 # mass of star (G == 1)
m1 = 1 # mass of planet 1
initpos1 = vector(0,1,0) # initial position vector of Planet1
Planet1 = sphere(pos=initpos1,radius=0.05*m1,color=color.blue)
Star = sphere(pos=vector(0,0,0),radius=0.1,color=color.yellow)
vel1 = -vector(25, 0, 0) # initial velocity of planet 1
trace = curve(radius = 0.005, color = color.white)

while step <= maxstep:

    rate(100)  # slow down the animation
    print (Planet1.pos)

    # calculate changes in velocities
    denom1M = mag(Planet1.pos) ** 3 
    dv1M = dt * Planet1.pos * M / denom1M
    
    # update velocities
    vel1 = vel1 - dv1M
    
    # update positions
    Planet1.pos = Planet1.pos + vel1 * dt

    step = step + 1
    trace.append(Planet1.pos)

print("end of program")
