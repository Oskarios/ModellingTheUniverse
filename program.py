# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:50:34 2020

@author: Oskar
"""


from vpython import * #For rendering
import numpy as n     #For optimisation
import matplotlib as mp #For plots
from scipy import signal #For resampling data to mean


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
        dv = -1 * self.dt * body.pos * fieldBody.mass / denom
        return dv
    
    #Only require kinetic energies of planets and not the Sun's? Though its KE should be zero? But efficiency
    def getKineticEnergies(self):
        ke = 0
        for i in range(1,self.numBodies - 1):
            ke += self.bodies[i].getKE()
        return ke

    #The target is the first object in the pair so index set to 0 (field of sun -> potential)
    def getPotentialEnergies(self):
        pe = 0
        for i in range(self.pairs.shape[0]):
            pe -= self.pairs[i][1].getPE(self.pairs[i][0])
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
        
        '''for i in range(self.pairs.shape[0]):
            self.pairs[i][1].dv += self.calcDV(self.pairs[i][0],self.pairs[i][1])
            #self.pairs[i][1].pos += self.pairs[i][1].vel * self.dt
            #dv_arr = np.append(dv_arr,self.calcDV(self.pairs[i][0],self.pairs[i][1]))
            
            
            
        for i in range(self.bodies.size):
            self.bodies[i].vel += self.bodies[i].dv * self.dt
            self.bodies[i].pos += self.pairs[i][1].vel * self.dt'''
            
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
        return ke
    
    #Returns the potential energy the c-body has with respect to the G field created
    #by the target body (target body creates field)
    def getPE(self, target):
        pe = (self.mass*target.mass)/mag(self.pos - target.pos)
        return pe
    
    def angle(self, r1,r2):
        angle1 = acos(dot(r1,r2) / ( (mag(r1)) * (mag(r2)) ) ) #Calculates the angle between the two position vectors
        return angle1
    
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
        self.sphere =  sphere(pos=vector(0,0,0), radius=self.radius*self.mass,color=colour)
        self.trace = curve(radius = 0.0025, color = colour)
        
    # When called will simultaneously update the posistion of the sphere (vpython) and the trace
    def updateBody(self,pos):
        self.sphere.pos = pos
        self.trace.append(pos)
                

class Simulation:
    def __init__(self,system,monthLength,dt,maxstep):        
        self.system = system
        self.monthLength = monthLength
        self.system.dt = dt
        self.nbodies = self.system.numBodies
        #self.dt = dt
        self.maxstep = maxstep
        self.step = 1
        
        #we bake a simulation so I call the big array that stores data "bake"
        self.bake = np.array([])
        
    def run(self):
        
        print(self.system.pairs)
        print(self.monthLength)
        while self.step <= self.maxstep:
            #rate(100) Don't need to limit this as this is running the simulation
            
            bakeStep = np.array([])
            for i in range(self.nbodies):
                bakeStep = np.append(bakeStep,self.system.bodies[i].pos)
                #bakeStep = np.append(bakeStep,)
            
            areas = np.array([0 for i in range(1,self.nbodies)])
            
            if self.step % self.monthLength == 0:
                areas = self.KeplerForPlanets(self.step)
                
                            
            bakeStep = np.append(bakeStep,areas)
            
            ke = self.system.getKineticEnergies()
            pe = self.system.getPotentialEnergies()
            
            energy = self.system.getTotalEnergy(ke,pe)
            
            bakeStep = np.append(bakeStep,(ke,pe,energy))
            #print(energy)
            
            #print(bakeStep)
            
            self.bake = np.reshape(np.append(self.bake,bakeStep),(self.step,2*self.nbodies+2))
            
            
            
            #Add the new row to the PHAT array 
            #self.bake = np.vstack((bake,bakeStep))
            
            
            #self.system.VelocityVerlet()
            
            self.system.Euler()
            
            self.step += 1
        '''print("FINAL BAKE")
        print(self.bake)
        print("/FINAL BAKE")
        #print(self.bake[:,-1])
        '''
        
        print(self.bake[0])
        print(self.bake[:,3])
        print(self.bake[:,1])
        print(self.bake[:,-1])
        
        self.energies = signal.resample(self.bake[:,-1],100)
        self.pes = signal.resample(self.bake[:,-2],100)
        self.kes = signal.resample(self.bake[:, -3],100)
     
        
    
    def KeplerForPlanets(self,step):
        areas = np.array([])
        for i in range(1,self.nbodies): # The first body is a star soo...
            areas = np.append(areas,self.Kepler(self.system.bodies[i],step))
            #print("KEPLER")
        return areas
        
    def Kepler(self, Planet,step):
        areatotal = 0
        monthlength = 100
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
        
        '''SOME SHIT LIKE THIS
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
        self.energyPlot(self.energies,"Total Energy","r")
        self.energyPlot(self.pes,"Potential Energy","g")
        self.energyPlot(self.kes,"Kinetic Energy","b")
        '''
        
        self.plotEnergies()
        self.plotAreas()
            
        
        for i in range(self.bake.shape[0]):
            rate(100)
            for j in range(renderers.size):
                renderers[j].updateBody(self.bake[i][j])
                
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
            areas = self.bake[:,self.nbodies+i]
            mp.pyplot.plot(areas,label="Planet " + str(i+1))
        mp.pyplot.xlabel("Time (samples)")
        mp.pyplot.ylabel("Area swept out")
        mp.pyplot.legend()
        mp.pyplot.show()
        
        

STAR = CelestialBody(1000,vector(0,0,0),vector(0,0,0),0.0001)
PLANET1 = CelestialBody(1, vector(0,1,0),-vector(25,0,0),0.1)
PLANET2 = CelestialBody(0.5, vector(0,3,0),-vector(10,0,0),0.1)
PLANET3 = CelestialBody(0.1, vector(0,4.5,0), -vector(3,0,0),0.1)

BODIES = np.array([STAR,PLANET1,PLANET2])

SYSTEM = SolarSystem(BODIES)
#SYSTEM.correctPairs()

sim = Simulation(SYSTEM,20,0.001,10000)
sim.run()
sim.render()