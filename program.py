# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:50:34 2020

@author: Oskar
"""


from vpython import * #For rendering
import numpy as np     #For optimisation
import matplotlib as mp #For plots
from scipy import signal #For resampling data to mean

#Increase the pixel density of figures created
mp.rcParams['figure.dpi'] = 300

'''
dt = 0.001
step = 1
maxstep = 10000
'''

#Pairs are permutation pairs of celestial bodies(including or excluding Sun?)
#This is difficult because for forces we don't move the sun but still need to consider sun for PE

'''
A Solar System is a collection of celestial bodies (i.e. a sun and a few planets)

The solar system controls the interactions between all celestial bodies (objects)
present within it

'''

class SolarSystem:
    def __init__(self,bodies):
        self.bodies = bodies #This will be a numpy array of celestial bodies
        self.numBodies = bodies.size # Prevents calling the .size attribute too many times (probably has little impact on performance)
        self.dt = 0 #Time step for particular solar system -- required for dv calculations (as passed from "parent" simulation)
        #print(permutations())
        self.G = 1.6233e-4 #Value of G, assuming radius is 1AU, time is 1 year, and the mass of the planet is 1
        
        
        #x = np.array([0,1,2,3])
        '''print(bodies.size)
        for i in range(bodies.size):
            np.append(body_indices,i)
            print(i)'''
        
        
        x = self.bodies # Found some code -- easier if array is x
        
        #print(x)
        #self.pairs = np.array(permutations(body_indices,2))
        #print(self.pairs)


        '''
            The next few lines calculate the permutation pairs (2d array) of c-bodies
        '''        
        ix = np.vstack([x for _ in range(x.shape[0])]) 
       # print(ix)
        #print(np.vstack(np.vstack(([ix],[ix.T])).T))
        '''
        def createDistMat(self, dim):
            M = np.array([[vector(0,0,0)]*dim]*dim)
            return M
        
        def getDV(self, matrix):
        '''
        self.pairs = np.vstack(np.vstack(([ix],[ix.T])).T)
        
        
        #self.pairs= np.vstack((self.pairs,np.flip(self.pairs,1)))
        
        #print("PAIRS")
        #print(self.pairs)
        
        
        self.correctPairs()     # Remove Sun-dependent pairs
        
        #print("CORRECTED PAIRS")
        #print(self.pairs.shape)
        
        #This removes any permutation/pair whose final element is the singular star
    '''   
    def correctPairs(self):
        print("Correcting pairs")
        print(self.pairs.shape)
        for i in range(self.pairs.shape[0]):
            if self.pairs[i][1] = = self.bodies[0] or self.pairs[i][0]==self.pairs[i][1]: #The sun is at 0 and we don't want to change its pos
                np.delete(self.pairs,i)
    '''
    '''
    def updateBodyVelocities(self):
        for i in range(self.pairs.shape[0]):
            self.pairs[i][1].vel -= self.calcDV(self.pairs[i][0],self.pairs[i][1])
    '''
    
    def correctPairs(self):
        toDelete = []
        for i in range(self.pairs.shape[0]):
            if self.pairs[i][1] == self.bodies[0] or self.pairs[i][0] == self.pairs[i][1]:
                # Will delete a pair if it represents interaction between something and itself, or the influence of a planet on the sun
                toDelete.append(i)
        self.pairs = np.delete(self.pairs,toDelete,0)
    
    '''
    def correctPairs(self):
        toDelete = []
        print("Correcting pairs")
        for i in range(self.pairs.shape[0]):
            if self.pairs[i][1] == 0 or self.pairs[i][0] == self.pairs[i][1]:
                print(self.pairs[i])
                toDelete.append(i)
        print(toDelete)
        self.pairs=np.delete(self.pairs,toDelete,0)
        
        #self.pairs = np.unique(self.pairs,)
        print("Unique Pairs")
        print(self.pairs)
        
    '''     
    
    #Calculates a*dt (dv) based on Newton's Law of Gravitation
    def calcDV(self, fieldBody, body):
        denom = mag(body.pos - fieldBody.pos) ** 3
        # -1 used instead of -G (G normalised as 1 in this model)
        dv = -self.G * self.dt * body.pos * fieldBody.mass / denom
        return dv
    
    #Only require kinetic energies of planets and not the Sun's? Though its KE should be zero? But efficiency
    def getKineticEnergies(self):
        #print("Get Kinetic Energies CALLED")
        ke = 0
        for i in range(self.numBodies):
            #print("i: " + str(i))
            ke += self.bodies[i].getKE()
        return ke

    #The target is the first object in the pair so index set to 0 (field of sun -> potential)
    def getPotentialEnergies(self):
        pe = 0
        for i in range(self.pairs.shape[0]):
            pe -= self.pairs[i][1].getPE(self.pairs[i][0],self.G)
        return pe
    
    #Sums total energy of system
    def getTotalEnergy(self,ke,pe):
        E = ke + pe
        return E
    
    #For velocity verlet method need to track both dv_i and dv_i+1
    def getDvArr(self):
        dv_arr = np.array([])
        for i in range(self.pairs.shape[0]):
            dv_arr = np.append(dv_arr,self.calcDV(self.pairs[i][0],self.pairs[i][1]))
        return dv_arr ## corresponds to the celestial body pairs dv of pairs[i][1]
    
    def VelocityVerlet(self):
        dvi_arr = self.getDvArr()
        
        self.updatePositions(dvi_arr)
        
        #Calculate the next accelerations
        
        dvs_arr = np.stack((dvi_arr,self.getDvArr()),1)
        
        #Update the velocities
        
        self.updateVelocities(dvs_arr)
    
    def Euler(self):
        for i in range(self.pairs.shape[0]):
            #For every permutatiuon incrementally update velocity and position of each planet in turn
            self.pairs[i][1].vel += self.calcDV(self.pairs[i][0],self.pairs[i][1])
            self.pairs[i][1].pos += self.pairs[i][1].vel * self.dt     
        
    #Updates posistion for all planets when using velocity verlet method
    def updatePositions(self,dv_arr):
        #print("ATTEMPTING POSITION UPDATE")
        #print(self.pairs.shape)
        #print(dv_arr)
        for i in range(self.pairs.shape[0]):
            self.pairs[i][1].pos += self.pairs[i][1].vel*self.dt + 0.5 * dv_arr[i] * self.dt
    
    #Updates velocities of all planets when using velocity verlet method
    def updateVelocities(self,dvs_arr):
        for i in range(self.pairs.shape[0]):
            self.pairs[i][1].vel += 0.5 * (dvs_arr[i][0] + dvs_arr[i][1])
                
        
    
        

                
'''
    Stars,planets,moons etc. would all be celestial bodies in this case
'''        
class CelestialBody:
    def __init__(self, mass, pos, vel, radius):
        self.mass = mass
        self.initpos = pos
        self.pos = pos
        self.vel = vel
        self.radius = radius
        
    #Updates its own position for use using Euler Methodology (I think? might be unused)
    def updatePos(self, dt):
        self.pos = self.pos + self.vel * dt
    
    #Returns its own kinetic energy    
    def getKE(self):
        ke = 0.5 * self.mass * mag(self.vel)**2
        #print(ke)
        return ke
    
    #Returns the potential energy the c-body has with respect to the G field created
    #by the target body (target body creates field)
    def getPE(self, target,G):
        pe = (G*self.mass*target.mass)/mag(self.pos - target.pos)
        return pe
    
    #Returns angle subtended by two subsequent planet positions    
    def angle(self, r1,r2):
        angle1 = acos(dot(r1,r2) / ( (mag(r1)) * (mag(r2)) ) ) #Calculates the angle between the two position vectors
        return angle1
    
    #Returns the area swept out by planet in 1 arbitrary "month"
    def area(self, r1, r2, theta):
        area1 = (r1*r2*theta)/2 #Calculates the area 
        #print(f"area = {area1}")
        return area1

'''
Body Renderer objects are used when rendering the pre-baked simulation
'''
class BodyRenderer:
    def __init__(self,mass,radius,colour):
        self.mass = mass # Here mass is only used to scale the visible radius of the rendered sphere
        self.radius = radius
        self.sphere =  sphere(pos=vector(0,0,0), radius=self.radius,color=colour)
        self.trace = curve(radius = 0.0025, color = colour)
        
    
    # When called will simultaneously update the posistion of the sphere (vpython) and the trace
    def updateBody(self,pos,area):
        self.sphere.pos = pos
        self.trace.append(pos)
        #if area:
            #curve(pos, vector(0,0,0), radius = 0.001)
                
'''
The simulation object controls everything relating to a specific simulation
- Simulation will control the baking of a simulation
- Simulation will then render said "baked" simulation
- Pre-determined plots will be created
'''
class Simulation:
    def __init__(self,system,monthLength,dt,maxstep):        
        self.system = system #Passed will be a numpy array of CelestialBodies
        self.monthLength = monthLength #Arbitrary length defined for Kepler II
        self.system.dt = dt #Time step specific to simulation and determined at instantiation
        self.nbodies = self.system.numBodies
        #self.dt = dt
        self.maxstep = maxstep #Determines how long simulation will run for
        self.step = 1 #Set to 1 instead of zero to make reshaping easier
        
        #we bake a simulation so I call the big array that stores data "bake"
        self.bake = np.array([]) #Initialisation of 2D bake that holds time-slices as individual 1D arrays
        
    '''
    THE RUN METHOD CONTROLS ALL CALCULATIONS REGARDING INTERACTIONS BETWEEN BODIES
        - CONSTRUCTS THE BAKED SIMULATION (self.bake)
    '''
    def run(self):
        
        #print(self.system.pairs) - DEBUGGING
        #print(self.monthLength) - DEBUGGING
        while self.step <= self.maxstep:
            #rate(100) Don't need to limit this as this is running the simulation
            
            #self.bake will be formed up of many bake-steps -- initialise empty np array for this 
            bakeStep = np.array([])
            for i in range(self.nbodies): #Add positions of planets for bakestep
                bakeStep = np.append(bakeStep,self.system.bodies[i].pos)
                #bakeStep = np.append(bakeStep,)
            
            #To maintain shape of bakestep create 0 array for areas swept out for nbodies-1 (ignore star)
            areas = np.array([0 for i in range(1,self.nbodies)])
            
            #Controls whether measurement for Kepler II is made 
            if self.step % self.monthLength == 0:
                areas = self.KeplerForPlanets(self.step) #Overides array each "month"
                
            #Adds the areas swept out for each planet to the bake-step                           
            bakeStep = np.append(bakeStep,areas)
            
            #Calculates the total kinetic energies for the solar system
            ke = self.system.getKineticEnergies()
            #print(ke)
            #Calculates the total potential energies for the solar system
            pe = self.system.getPotentialEnergies()
            #Calculates the total energy of the system
            energy = self.system.getTotalEnergy(ke,pe)
            #Adds the energies to the bakestep (this array is still 1D)
            bakeStep = np.append(bakeStep,(ke,pe,energy))
            #print(energy)
            
            #print(bakeStep)
            
            #Uses numpy array methods in order to add bakestep to self.bake as an entire array
            #self.bake is then a 2D array made up of 'time-slices' of different properties of system
            self.bake = np.reshape(np.append(self.bake,bakeStep),(self.step,2*self.nbodies+2))
            
            
            
         
            
            #Calculates planetary updates based upon velocity-verlet numerical method
            #self.system.VelocityVerlet()
            
            #Calculates planetary updates based upon Euler numerical method
            self.system.Euler()
            
            
            #Increments steps
            self.step += 1
        '''print("FINAL BAKE")
        print(self.bake)
        print("/FINAL BAKE")
        #print(self.bake[:,-1])
        PURELY FOR DEBUGGING
        '''
        
        '''
        PURELY FOR DEBUGGING AGAIN HERE
        print(self.bake[0])
        print(self.bake[:,3])
        print(self.bake[:,1])
        print(self.bake[:,-1])
        '''
        
        #Using array slicing a single parameter (i.e. kinetic energy) can be tracked across the bake (as in through time)
        #signal.resample used to average out oscillating calculated values -- probably unnecessary tbh
        self.energies = self.bake[:,-1]
        self.pes = self.bake[:,-2]
        self.kes = self.bake[:, -3]
     
        
    
    def KeplerForPlanets(self,step):
        areas = np.array([]) #Empty area of areas to be populated by areas swept by each planet
        for i in range(1,self.nbodies): # The first body is a star soo...
            areas = np.append(areas,self.Kepler(self.system.bodies[i],step))
            #print("KEPLER")
        return areas
        
    #
    def Kepler(self, Planet,step):
        areatotal = 0
        #monthlength = 100
        pos1 = Planet.initpos
        angle2 = Planet.angle(pos1, Planet.pos) #Calls the function to calculate the angle between the vectors
        areaelement = Planet.area(mag(pos1), mag(Planet.pos),angle2)#Calculates the area between the vectors
        #lineplanet = curve(vector(Planet.pos), vector(self.Star.pos)) #Draws lines from the star to the planet
        #pos1 = copy.copy(Planet.pos) #Changes the value of the planet vector
        areatotal = areatotal + areaelement
        return areatotal 
        
        #np.savetxt("simulation.csv",self.bake)
        
    def render(self):
        #this renders the whole baked simulation in vpython
        
        #iterate through self.bake 
        #i.e.
        
        '''SOME stuff LIKE THIS
        for i in range(self.bake.shape[0]):
            sunpos = self.bake[i][0]
            
        '''
        
        renderers = np.array([])
        for i in range(self.nbodies):
            renderers = np.append(renderers,BodyRenderer(self.system.bodies[i].mass,self.system.bodies[i].radius,color.white))
        '''
        print(renderers)
        
        '''
        
        #self.energies = np.reshape(self.energies,(100,100))
        
        #Evals = np.mean(self.energies,axis=1)
        
        #Eerror = np.std(self.energies,axis=1,dtype=np.float64)/np.sqrt(self.energies.shape[1])
        #print(Evals)
       # print(Eerror)
        #mp.pyplot.errorbar(np.array([i*100 for i in range(Evals.size)]),Evals,Eerror,label="Total Energy",color='r',ls='-', marker='x',capsize=5,capthick=1,ecolor='r')
        '''
        relposition=np.subtract(self.bake[:,2],self.bake[:,1])[0:4000] # distance between planet one and planet two
        posrelsun2= np.subtract(self.bake[:,2], self.bake[:,0])[0:4000] # distance of planet 2 away from the sun
        posrelsun1= np.subtract(self.bake[:,1], self.bake[:,0])[0:4000] # distance between planet 1 and the sun
        print(relposition)

        relx = np.array([])
        rely = np.array([])
        for i in range(relposition.size):               # a graph is created to plot the magnitude of the distance between the two planets
            relx = np.append(relx,relposition[i].x)     # calculates the relative x displacement
            rely = np.append(rely,relposition[i].y)     # calculated the relative y displacement 
            c= np.sqrt((rely)**2+(relx)**2)             # calculates the magnitude of the displacement
        print("Subtract worked")
        mp.pyplot.xlabel("Time[samples]")
        mp.pyplot.ylabel ("Displacement")
        mp.pyplot.plot(c)                               # graph is plotted
        mp.pyplot.show()
        
        relx = np.array([])
        rely = np.array([])
        for i in range(posrelsun2.size):                # the same is repeated between planet 2 and the sun
            relx=np.append(relx,posrelsun2[i].x)
            rely=np.append(rely,posrelsun2[i].y)
            d= np.sqrt((rely)**2+(relx)**2)
        mp.pyplot.xlabel("Time[samples]")
        mp.pyplot.ylabel ("Displacement")
        mp.pyplot.plot(d)
        mp.pyplot.show()
        
        relx = np.array([])
        rely = np.array([])
        for i in range(posrelsun1.size):                # the same is reppeated between planet 1 and the sun 
            relx=np.append(relx,posrelsun1[i].x)
            rely=np.append(rely,posrelsun1[i].y)
            d= np.sqrt((rely)**2+(relx)**2)
        mp.pyplot.xlabel("Time[samples]")
        mp.pyplot.ylabel ("Displacement")
        mp.pyplot.plot()
        mp.pyplot.show()
        '''
        
        
        '''
        self.energyPlot(self.energies,"Total Energy","r")
        self.energyPlot(self.pes,"Potential Energy","g")
        self.energyPlot(self.kes,"Kinetic Energy","b")
        '''
        print(self.kes)
        
        self.plotEnergies()
        self.plotAreas()
            
        
        for i in range(self.bake.shape[0]):
            rate(100)
            for j in range(renderers.size):
                area = 0
                if j > 0:
                    area = self.bake[:,self.nbodies+j][i]
                renderers[j].updateBody(self.bake[i][j],area)
                
    def energyPlot(self,energies_raw,label,colour):
        energies = np.reshape(energies_raw,(100,100))
        values = np.mean(energies,axis=1)
        errors = np.std(energies,axis=1,dtype=np.float64)/np.sqrt(energies.shape[1])
        mp.pyplot.errorbar(np.array([i*100 for i in range(values.size)]),values,errors,label=label,color=colour,ls='-', marker='x',capsize=5,capthick=1,ecolor=colour)
        
    def plotEnergies(self):
        mp.pyplot.plot(self.energies,'r',label="Total Energy")
        mp.pyplot.plot(self.pes,'g',label="Potential Energy")
        mp.pyplot.plot(self.kes,'b',label="Kinetic Energy")
        mp.pyplot.xlabel("Time (samples)")
        mp.pyplot.ylabel("Energy")
        mp.pyplot.legend()
        mp.pyplot.show()
        
    def plotAreas(self):
        for i in range(self.nbodies-1):
            areas = self.bake[:,self.nbodies+i][0:4000]
            x = np.nonzero(areas)
            #np.savetxt("foo"+str(i)+".csv", areas, delimiter=",")
            mp.pyplot.plot(x[0],areas[x[0]],'o',label="Planet " + str(i+1), markersize=1)
        mp.pyplot.xlabel("Time (samples)")
        mp.pyplot.ylabel("Area swept out")
        mp.pyplot.legend()
        mp.pyplot.show()
        
        
#Creates the Sun -- Mass set to be 330 000 EARTH MASSES
STAR = CelestialBody(330000,vector(0,0,0),vector(0,0,0),0.1)       #Creates Star
'''
INITIAL CONDITIONS WITH G NORMALISED AS 1 AND HONESTLY IT NEVER WORKED

PLANET1 = CelestialBody(1, vector(0,3,0),-vector(250,0,0),0.06)       #Creates planet
PLANET2 = CelestialBody(0.05, vector(0,10,0),-vector(15,0,0),0.1)     #Creates planet
PLANET3 = CelestialBody(0.1, vector(0,4.5,0), -vector(3,0,0),0.1)   #Creates planet
'''

'''
Now let's take some inspo from our Solar System 
    - Planets commented out if they broke the simulation
    - Can't seem to get more than two planets to be stable
    - I don't know what's going on at this point
    
    Source for Mass: https://www.google.com/search?q=earth+mass+of+solar+system+planets&rlz=1C1CHBF_enGB885GB885&oq=earth+mass+of+solar+system+planets&aqs=chrome..69i57j33l6.5418j1j7&sourceid=chrome&ie=UTF-8
    Source for Speeds: https://www.google.com/search?q=earth+mass+of+solar+system+planets&rlz=1C1CHBF_enGB885GB885&oq=earth+mass+of+solar+system+planets&aqs=chrome..69i57j33l6.5418j1j7&sourceid=chrome&ie=UTF-8
    Source for Orbits: https://www.wolframalpha.com/

'''

MERCURY = CelestialBody(0.055,vector(0,0.3606,0),-vector(10.02,0,0),0.06)
VENUS = CelestialBody(0.815,vector(0,7.28,0),-vector(7.388,0,0),0.06)
EARTH = CelestialBody(1,vector(0,1,0),-vector(6.283,0,0),0.06)
#MARS = CelestialBody(0.107,vector(0,9.55,0),-vector(5.082,0,0),0.04)
#JUPITER = CelestialBody(317.8,vector(0,5.207,0),-vector(2.754,0,0),0.09)
#Creates numpy array of all celestial bodies -- makes it easier to pass as parameter to instantiate solar system
BODIES = np.array([STAR,EARTH,VENUS,MERCURY])#Creates solar system made up of celestial bodies found in np.array -- BODIES
SYSTEM = SolarSystem(BODIES)
#SYSTEM.correctPairs()


sim = Simulation(SYSTEM,20,0.001,10000)
sim.run()
sim.render()
