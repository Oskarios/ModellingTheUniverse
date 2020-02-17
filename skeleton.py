# Skeleton code starting point for Yr 1 Theory Group Project

from vpython import *
import numpy as np

dt = 0.001 # timestep
step = 1 # loop counter
maxstep = 100000

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
    def __init__(self, mass, initpos, vel, radius, colour, moons=[]):
        self.mass = mass
        self.initpos = initpos
        self.vel = vel
        self.radius = radius
        self.pos = self.initpos
        #Array of moons planet has
        self.moons = moons
        self.sphere = sphere(pos=self.initpos, radius=self.radius*self.mass,color=colour)
        self.trace = curve(radius = 0.005, color = colour)
        if len(moons) > 0:
            M = self.createMoonMatrix()
            

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




class Moon:
    def __init__(self, parent, mass, vel, radius, orbit_rad):
        self.parent = parent
        self.mass = mass
        self.vel = vel
        self.radius = radius
        #please make sure that the orbit radius is very small
        self.pos = self.parent.pos + vector(0,-orbit_rad,0)
        self.sphere = sphere(pos=self.pos, radius=self.radius, color=color.white)
        self.trace = curve(radius=0.001, color=color.white)


    #REQUIRED METHODS
        # - NEED TO CALCULATE DV BETWEEN  ITSELF AND MOON

        def updateSphere(self):
            self.sphere.pos = self.pos

        def updatePos(self):
            self.pos = self.pos+self.vel * dt
            self.updateSphere()
            self.updateTrace

        def updateTrace(self):
            self.trace.append(self.pos)

        


#This class will control the forces present between the bodies within the solar system
class SolarSystem:
    def __init__(self, Star, planets):
        self.Star = Star
        self.planets = planets

        #Create some planets and add them to the planets array
        self.planets.append(Planet(1,vector(0,1,0), -vector(25,0,0), 0.05, colour=color.green))
        self.planets.append(Planet(2,vector(0,1.5,0), -vector(15,0,0), 0.05, colour=color.blue))
        self.planets.append(Planet(0.4,vector(0,4,0), -vector(15,0,0), 0.05, colour=color.blue))

        self.F = self.createPlanetMatrix()

        #Do we do this here or do we call it separately -- do it in the loop
        self.forces = self.getForces_IP() #Going to be a huge matrix linking the various forces to the required bodies 
        print(self.forces)
        print(self.planets)

    #Adds a planet to the solar system after initialisation
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

    #Basically creates N X N square adjacency matrix for N number of planets
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
        
        '''
        for j in range(1,len(self.planets)):
            for i in range(j-1,len(self.planets)-1):
                self.F[i][j]=1
                self.F[j][i] = -1
        '''
        '''
        for i in range(0,len(self.planets)-1):
            for j in range(i+1, len(self.planets)):
                
                self.F[i][j] = str(i)+str(j)
                #self.F[j][i] = - 1 * self.F[i][j]

            print(i,j)
            
        for x in range(0,len(self.planets)-1):
            for y in range(x+1, len(self.planets)):
                print(F[x][y])
                F[y][x] = -1 * F[x][y]
                #F[j][i] = -F[i][j]

        '''
        #print(self.F)

        
    #Updates all the velocities of all planets in solar system
    def updatePlanetVelocities(self):
        for i in range(len(self.planets)):
            vel = self.planets[i].vel
            dviM = self.calcForce(self.Star, self.planets[i])
            vel -= dviM
            for j in range(len(self.F[i])):
                if self.F[i][j] != 0:
                    #Don't know whether should be + or - but + gives constant energy
                    vel += self.F[i][j]
            self.planets[i].vel = vel

    #updates the position of all planets in solar system 
    def updatePlanetPositions(self):
        for i in range(len(self.planets)):
            self.planets[i].updatePos()
            #Also want to update the positions of moons if the planet has them
            
    #Gets the kinetic energy of all planets in self.planets
    def getKineticEnergies(self):
        ke = 0
        for i in range(len(self.planets)):
            ke += self.planets[i].getKE()
        return ke
    
    #Gets the potential energies of all planets in system
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


#  Define the star, planets and constants
M = 1000 # mass of star (G == 1)
#Star = sphere(pos=vector(0,0,0),radius=0.1,color=color.yellow)

Star = Star(M, vector(0,0,0), radius=0.1)



#Create first planet with OO approach
'Planet1 = Planet(1,vector(0,1,0), -vector(25,0,0), 0.05, colour=color.green)'
#Create second planet with OO approach
'Planet2 = Planet(2,vector(0,1.5,0), -vector(15,0,0), 0.05, colour=color.blue)'

system = SolarSystem(Star,[])
energyPlot = []
while step <= maxstep:

    rate(100)  # slow down the animation
    #print (Planet1.pos)

    #Calculate Changes in velocities
    system.getForces_IP()
    

    #Update Velocities
    system.updatePlanetVelocities()

    #Update Positions
    system.updatePlanetPositions()

    #Push total energy
    energy = system.getTotalEnergy(system.getKineticEnergies(),system.getPotentialEnergies())
    print(energy)

    #Instead of just printing energy need to plot it

    #Average a reading over 1 or 2 seconds?
    
    '''
    # calculate changes in velocities

    #Planet1 Star
    denom1M = mag(Planet1.pos) ** 3 
    dv1M = dt * Planet1.pos * M / denom1M

    #Planet2 Star
    denom2M = mag(Planet2.pos)** 3
    dv2M = dt * Planet2.pos * M / denom2M

    #Planet1 on Planet 2
    denom12 = mag(Planet2.pos - Planet1.pos) ** 3
    dv12 = dt * Planet1.pos * Planet2.mass / denom12

    
    #Planet2 on Planet1
    denom21 = mag(Planet1.pos - Planet2.pos) ** 3
    dv21 =  dt * Planet2.pos * Planet1.mass / denom21
    
    

    #update velocities
    Planet1.vel += - dv1M - dv21 + dv12
    Planet2.vel += - dv2M - dv12 + dv21
    
    # update positions
    Planet1.updatePos()
    Planet2.updatePos()
    
    #Calculate total KE
    Total_KE = Planet1.getKE() + Planet2.getKE()

    #Calculate total potential energy
    Total_PE = Planet1.getPE(Planet2) + Planet2.getPE(Planet1) + Planet1.getPE(Star) + Planet2.getPE(Star)

    #Calculate the total energy of the solar system (the Hamiltonian)
    Total_Energy = Total_KE - Total_PE

    print(Total_Energy)
    '''

    step += 1
    
print("end of program")
