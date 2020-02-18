# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:50:34 2020

@author: Oskar
"""

from vpython import *
import numpy as np
import matplotlib as mp
from itertools import permutations
'''
dt = 0.001
step = 1
maxstep = 10000
'''

#Pairs are permutation pairs of celestial bodies(including or excluding Sun?)
#This is difficult because for forces we don't move the sun but still need to consider sun for PE
class SolarSystem:
    def __init__(self,bodies):
        self.bodies = bodies #This will be a numpy array of celestial bodies
        self.numBodies = bodies.size
        self.dt = 0
        #print(permutations())
        #x = np.array([0,1,2,3])
        '''print(bodies.size)
        for i in range(bodies.size):
            np.append(body_indices,i)
            print(i)'''
        
        
        x = self.bodies
        
        #print(x)
        #self.pairs = np.array(permutations(body_indices,2))
        #print(self.pairs)
        
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
        
        self.correctPairs()
        
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
    def calcDV(self, fieldBody, body):
        denom = mag(body.pos - fieldBody.pos) ** 3
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
    
    def getTotalEnergy(self,ke,pe):
        E = ke + pe
        return E
    
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
        
    
    def updatePositions(self,dv_arr):
        #print("ATTEMPTING POSITION UPDATE")
        #print(self.pairs.shape)
        #print(dv_arr)
        for i in range(self.pairs.shape[0]):
            self.pairs[i][1].pos += self.pairs[i][1].vel*self.dt + 0.5 * dv_arr[i] * self.dt
    
    def updateVelocities(self,dvs_arr):
        for i in range(self.pairs.shape[0]):
            self.pairs[i][1].vel += 0.5 * (dvs_arr[i][0] + dvs_arr[i][1])
                
            

class CelestialBody:
    def __init__(self, mass, pos, vel, radius):
        self.mass = mass
        self.pos = pos
        self.vel = vel
        self.radius = radius
    
    def updatePos(self, dt):
        self.pos = self.pos + self.vel * dt
        
    def getKE(self):
        ke = 0.5 * self.mass * mag(self.vel)**2
        return ke
    
    def getPE(self, target):
        pe = (self.mass*target.mass)/mag(self.pos - target.pos)
        return pe

class Simulation:
    def __init__(self,system,dt,maxstep):        
        self.system = system
        self.system.dt = dt
        self.nbodies = self.system.numBodies
        #self.dt = dt
        self.maxstep = maxstep
        self.step = 0
        
        #we bake a simulation so I call the big array that stores data "bake"
        self.bake = np.array([])
        
    def run(self):
        while self.step <= self.maxstep:
            #rate(100) Don't need to limit this as this is running the simulation
            
            bakeStep = np.array([])
            for i in range(self.nbodies):
                bakeStep = np.append(bakeStep,(self.system.bodies[i].pos,self.system.bodies[i].vel))
                #bakeStep = np.append(bakeStep,)
            
            ke = self.system.getKineticEnergies()
            pe = self.system.getPotentialEnergies()
            
            energy = self.system.getTotalEnergy(ke,pe)
            
            bakeStep = np.append(bakeStep,(ke,pe,energy))
            #print(energy)
            
            #print(bakeStep)
            
            self.bake = np.reshape(np.append(self.bake,bakeStep),(self.step + 1,2*self.nbodies + 3))
            
            
            
            #Add the new row to the PHAT array 
            #self.bake = np.vstack((bake,bakeStep))
            self.system.VelocityVerlet()
            self.step += 1
        print(self.bake)
        
    def render(self):
        #this renders the whole baked simulation in vpython
        
        #iterate through self.bake 
        #i.e.
        
        '''SOME SHIT LIKE THIS
        for i in range(self.bake.shape[0]):
            sunpos = self.bake[i][0]
            
        '''
        return False

STAR = CelestialBody(1000,vector(0,0,0),vector(0,0,0),0.1)
PLANET1 = CelestialBody(1, vector(0,1,0),-vector(25,0,0),0.05)
PLANET2 = CelestialBody(2, vector(0,1.5,0),-vector(15,0,0),0.05)
PLANET3 = CelestialBody(0.4, vector(0,4,0), -vector(15,0,0),0.05)

BODIES = np.array([STAR,PLANET1,PLANET2,PLANET3])

SYSTEM = SolarSystem(BODIES)
#SYSTEM.correctPairs()

sim = Simulation(SYSTEM,0.001,10000)
sim.run()