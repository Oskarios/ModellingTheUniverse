# Skeleton code starting point for Yr 1 Theory Group Project

from vpython import *

dt = 0.001 # timestep
step = 1 # loop counter
maxstep = 100000

#  Define the star, planets and constants
M = 1000 # mass of star (G == 1)
m1 = 1 # mass of planet 1
initpos1 = vector(0,1,0) # initial position vector of Planet1
Planet1 = sphere(pos=initpos1,radius=0.05*m1,color=color.blue)
Star = sphere(pos=vector(0,0,0),radius=0.1,color=color.yellow)
vel1 = -vector(25, 0, 0) # initial velocity of planet 1


#Set up rough secondary planet
m2 = 2
initpos2 = vector(0,1.5,0)
Planet2 = sphere(pos=initpos2, radius=0.05*m2, color=color.green)
vel2 = -vector(15, 0, 0)

# Added trace for planet
trace1 = curve(radius = 0.005, color = color.white)

#Added trace for planet 2
trace2 = curve(radius = 0.005, color = color.green)

while step <= maxstep:

    rate(100)  # slow down the animation
    print (Planet1.pos)

    # calculate changes in velocities

    #Planet1 Star
    denom1M = mag(Planet1.pos) ** 3 
    dv1M = dt * Planet1.pos * M / denom1M

    #Planet2 Star
    denom2M = mag(Planet2.pos)** 3
    dv2M = dt * Planet2.pos * M / denom2M

    #Planet1 on Planet 2
    denom12 = mag(Planet2.pos - Planet1.pos) ** 3
    dv12 = dt * Planet1.pos * m2 / denom12

    
    #Planet2 on Planet1
    denom21 = mag(Planet1.pos - Planet2.pos) ** 3
    dv21 =  dt * Planet2.pos * m2 / denom21
    
    

    #update velocities
    vel1 = vel1 - dv1M - dv21 + dv12
    vel2 = vel2 - dv2M - dv12 + dv21
    
    # update positions
    Planet1.pos = Planet1.pos + vel1 * dt
    Planet2.pos = Planet2.pos + vel2 * dt

    step = step + 1
    trace1.append(Planet1.pos)
    trace2.append(Planet2.pos)

print("end of program")
