# Skeleton code starting point for Yr 1 Theory Group Project

from vpython import *
import numpy as np
import copy as copy

dt = 0.001 # timestep
global step
step = 1 # loop counter
maxstep = 1000

#Create class for star so that it can be used as a target object later on
class Star:
    def __init__(self, mass, initpos, radius):
        #Currently taking stars as not having velocities so it has no such property
        self.mass = mass
        self.initpos = initpos
        self.pos = self.initpos
        self.radius = radius
        #Main reason for having separate class is line below -- should create superclass for which subclasses override this line
        self.sphere=sphere(pos=self.initpos, radius=self.radius, color=color.yellow)


#Created a reusable class for 
class Planet:
    def __init__(self, mass, initpos, vel, radius, colour):
        self.mass = mass
        self.initpos = initpos
        self.vel = vel
        self.radius = radius
        self.pos = self.initpos
        self.sphere = sphere(pos=self.initpos, radius=self.radius*self.mass,color=colour)
        self.trace = curve(radius = 0.005, color = colour)

    def updateSphere(self):
        self.sphere.pos = self.pos

    #Performs necessary positional updates for the planet
    def updatePos(self):
        self.pos = self.pos + self.vel * dt
        self.updateSphere()
        self.updateTrace()

    def updateTrace(self):
        self.trace.append(self.pos)

    #Returns kinetic energy of the planet
    def getKE(self):
        ke = 0.5 * self.mass * mag(self.vel)**2
        return ke

    #Returns the potential energy of the planet with respect to target celestial body object
    def getPE(self, target):
        pe = (self.mass * target.mass)/mag(self.pos - target.pos)
        return pe
        
        
    def angle(self, r1,r2):
        angle1 = acos(dot(r1,r2) / ( (mag(r1)) * (mag(r2)) ) ) #Calculates the angle between the two position vectors
        return angle1
    
    def area(self, r1, r2, theta):
        area1 = (r1*r2*theta)/2 #Calculates the area 
        print(f"area = {area1}")
        return area1





#This class will control the forces present between the bodies within the solar system
class SolarSystem:
    def __init__(self, Star, planets):
        self.Star = Star
        self.planets = planets

        #Create some planets and add them to the planets array
        self.planets.append(Planet(1,vector(0,1,0), -vector(25,0,0), 0.05, colour=color.green))
        self.planets.append(Planet(2,vector(0,1.5,0), -vector(15,0,0), 0.05, colour=color.blue))
        self.planets.append(Planet(0.4,vector(0,2,0), -vector(15,0,0), 0.05, colour=color.blue))

        self.F = self.createPlanetMatrix()

        #Do we do this here or do we call it separately -- do it in the loop
        self.forces = self.getForces_IP() #Going to be a huge matrix linking the various forces to the required bodies 
        print(self.forces)
        print(self.planets)


    def addPlanet(self, planet):
        self.planets.append(planet)


    ##This uses dictionaries which isn't going to help -- ##it also straight up doesn't work
    def getForces_IP_1(self):
        forces = {}
        for i in range(0,len(self.planets)):
            forces[i] = {}
            for j in range(i+1, len(self.planets)):
                forces[i][j] = str(i)+str(j) ## this was a debugging thing
        return forces

    def createPlanetMatrix(self):
        M = [[0 for  i in range(len(self.planets))]for j in range(len(self.planets))]
        return M

    def getForces_IP(self):
        #USE NUMPY ARRAYS TO FORM A MATRIX ??

        #Figure out how to do it in regular python first
        #F = [[0 for i in range(len(self.planets))]]* len(self.planets)

        for i in range(0,len(self.planets)):
            for j in range(i+1, len(self.planets)):
                #force = self.calcForce(self.planets[i],self.planets[j])
                self.F[i][j] = self.calcForce(self.planets[i],self.planets[j]) #pretty sure it's not specifically a force but hey-ho

                self.F[j][i] = self.calcForce(self.planets[j],self.planets[i])

    def updatePlanetVelocities(self):
        for i in range(len(self.planets)):
            vel = self.planets[i].vel
            dviM = self.calcForce(self.Star, self.planets[i])
            vel -= dviM
            for j in range(len(self.F[i])):
                if self.F[i][j] != 0:
                    vel += self.F[i][j]
            self.planets[i].vel = vel

    def updatePlanetPositions(self):
        for i in range(len(self.planets)):
            self.planets[i].updatePos()

    def getKineticEnergies(self):
        ke = 0
        for i in range(len(self.planets)):
            ke += self.planets[i].getKE()
        return ke

    def getPotentialEnergies(self):
        pe = 0
        for i in range(0,len(self.planets)):
            for j in range(i+1,len(self.planets)):
                if self.F[i][j] != 0:
                    pe -= self.planets[i].getPE(self.planets[j])
                    pe -= self.planets[j].getPE(self.planets[i])
        return pe


    def getTotalEnergy(self, ke, pe):
        E = ke - pe
        return E
                    
    
    #This updates position and velocities automatically
    def updatePos_EulerCromer(self):
        vels = []
        for i in range(len(self.planets)):
            vel = self.planets[i].vel
            dv = -1 * self.calcForce(self.Star, self.planets[i])
            print(dv)
            
            
            for j in range(len(self.F[i])):
                if self.F[i][j]!=0:
                    dv += self.F[i][j]
                
                    vel += dv
                    vels.append(vel)
            
            
        self.updatePlanetPositions()
            
        for i in range(len(self.planets)):
            self.planets[i].vel = vels[i]

    ## REQUIRED METHODS ##

        # CALCULATE FORCES BETWEEN BODIES IN SYSTEM
    # I DON'T THINK THIS IS GOING TO WORK
    def calcForce(self, fieldBody, body):
        denom = mag(body.pos-fieldBody.pos) ** 3
        dv = dt * body.pos * fieldBody.mass / denom
        return dv
        # CALCULATE TOTAL KINETIC ENERGIES OF BODIES IN SYSTEM
        # CALCULATE TOTAL POTENTIAL ENERGIES OF BODIES IN SYSTEM
        # CALCULATE TOTAL ENERGY OF SYSTEM
        
    def KeplerForPlanets(self):
        for i in range(len(self.planets)):
            self.Kepler(self.planets[i],step)
        
    def Kepler(self, Planet,step):
        areatotal = 0
        monthlength = 100
        pos1 = Planet.initpos
        angle2 = Planet.angle(pos1, Planet.pos) #Calls the function to calculate the angle between the vectors
        areaelement = Planet.area(mag(pos1), mag(Planet.pos),angle2)#Calculates the area between the vectors
        lineplanet = curve(vector(Planet.pos), vector(self.Star.pos)) #Draws lines from the star to the planet
        pos1 = copy.copy(Planet.pos) #Changes the value of the planet vector
        areatotal = areatotal + areaelement

                
            

#  Define the star, planets and constants
M = 1000 # mass of star (G == 1)
#Star = sphere(pos=vector(0,0,0),radius=0.1,color=color.yellow)

Star = Star(M, vector(0,0,0), radius=0.1)
monthlength = 20
system = SolarSystem(Star,[])
energyPlot = []
while step <= maxstep:

    rate(100)  # slow down the animation
    #print (Planet1.pos)

    #Calculate Changes in velocities
    system.getForces_IP()
    
    if step%monthlength == 0 and step !=0:
        areas = system.KeplerForPlanets()
    else:
        areas = np.array([0 for i in range(1,self.nbodies)])

    #Update Velocities
    system.updatePlanetVelocities()

    #Update Positions
    system.updatePlanetPositions()   
    
    #system.updatePos_EulerCromer()

    #Push total energy
    energy = system.getTotalEnergy(system.getKineticEnergies(),system.getPotentialEnergies())
    #print(energy)

    #Instead of just printing energy need to plot it

    #Average a reading over 1 or 2 seconds?
    
    step += 1
    
print("end of program")
print(areatotal)
