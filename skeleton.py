# Skeleton code starting point for Yr 1 Theory Group Project
from vpython import *

dt = 0.001 # timestep
step = 1 # loop counter
maxstep = 1000

#  Define the star, planets and constants
M = 1000 # mass of star (G == 1)
Star = sphere(pos=vector(0,0,0),radius=0.1,color=color.yellow) #This produces the star at the centre of the system

m1 = 1 # mass of planet 1
initpos1 = vector(0,1,0) # initial position vector of Planet1
Planet1 = sphere(pos=initpos1,radius=0.05*m1,color=color.blue) #This produces a sphere of designated radius, colour and centre.
vel1 = -vector(25, 0, 0) # initial velocity of planet 1

trace1 = curve(radius=0.005, color=color.blue)  #This creates a trace on the planet to track its path
trace2= curve(radius=0.005, color=color.cyan)  #the previous line was repeated to produce a trace for each planet
trace3=curve(radius=0.005, color=color.white)

m2 = 1                                  #lines 12-15 have been repeated to produce two exta planets with different orbits, masses and colours 
initpos2= vector(0,1.5,0)
Planet2= sphere(pos=initpos2,radius=0.05*m2, color=color.cyan)
vel2= -vector(30,0,0)

m3= 2
initpos3= vector(0,2,0)
Planet3=sphere(pos=initpos3, radius=0.025*m3, color=color.white)
vel3= -vector(10,0,0)



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
    
    print (Planet2.pos)                    #Lines 37-47 have been repeated to produce the orbits of the differnet planets using keplers law.
    denom2M = mag(Planet2.pos)**3
    dv2M = dt * Planet2.pos * M/denom2M
    vel2 = vel2 - dv2M
    Planet2.pos = Planet2.pos + vel2 * dt
    
    print (Planet3.pos)
    denom3M= mag(Planet3.pos)**3
    dv3M= dt * Planet3.pos* M/denom3M
    vel3= vel3 - dv3M
    Planet3.pos = Planet3.pos +vel3 *dt
    
    trace1.append(Planet1.pos)    # The trace has been added after every step to produce the path of orbit.
    trace2.append(Planet2.pos)      #Previous line has been repeated for each planet.
    trace3.append(Planet3.pos)
    step = step + 1
    
print("end of program")

# This is a long way of doing this, it can be done more simpily by created a class which will reduce the amount of code needed. 
