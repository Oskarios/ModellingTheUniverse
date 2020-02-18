# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:50:34 2020

@author: Oskar
"""

from vpython import *
import numpy as np
import matplotlib as mp
from itertools import permutations

dt = 0.001
step = 1
maxstep = 10000


#Pairs are permutation pairs of celestial bodies(including or excluding Sun?)
#This is difficult because for forces we don't move the sun but still need to consider sun for PE
class SolarSystem:
    def __init__(self,bodies, dt):
        self.bodies = bodies #This will be a numpy array of celestial bodies
        self.numBodies = bodies.size
        self.dt = dt
        self.pairs = np.array(permutations(self.bodies),2)
        '''
        def createDistMat(self, dim):
            M = np.array([[vector(0,0,0)]*dim]*dim)
            return M
        
        def getDV(self, matrix):
        '''
        self.correctPairs()
        
        #This removes any permutation/pair whose final element is the singular star
        def correctPairs(self):
            for i in range(self.pairs.shape[0]):
                if self.pairs[i][1] != self.bodies[0]: #The sun is at 0 and we don't want to change its pos
                    np.delete(self.pairs,i)
    
        
        def updateBodyVelocities(self):
            for i in range(self.pairs.size):
                self.pairs[i][1].vel -= self.calcDV(self.pairs[i][0],self.pairs[i][1])

            
        def calcDV(self, fieldBody, body):
            denom = mag(body.pos - fieldBody.pos) ** 3
            dv = self.dt * body.pos * fieldBody.mass / denom
            return dv
        
        #Only require kinetic energies of planets and not the Sun's? Though its KE should be zero? But efficiency
        def getKineticEnergies(self):
            ke = 0
            for i in range(1,self.bodies.size - 1):
                ke += self.bodies[i].getKE()
            return ke
    
        #The target is the first object in the pair so index set to 0 (field of sun -> potential)
        def getPotentialEnergies(self):
            pe = 0
            for i in range(self.pairs.shape[0]):
                pe -= self.pairs[i][1].getPE(self.planets[0])
            return pe
    

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
        self.dt = dt
        self.maxstep = maxstep
        self.step = 0
        
        #we bake a simulation so I call the big array that stores data "bake"
        self.bake = np.array([])
        
    def run():
        while self.step <= self.maxstep:
            bakeStep = np.array([])
            
            
            
            
            #Add the new row to the PHAT array 
            self.bake = np.vstack((bake,bakeStep))
            
            self.step += 1
            
    def render():
        #this renders the whole baked simulation in vpython
        
        #iterate through self.bake 
        #ie
        
        '''SOME SHIT LIKE THIS
        for i in range(self.bake.shape[0]):
            sunpos = self.bake[0][0]
            
        '''
        return False


