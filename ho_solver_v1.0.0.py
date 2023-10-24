#!/usr/bin/env python
#Author: Willian Matioli Serenone
#Date: 2023-10-15
import numpy as np
from matplotlib import pyplot as plt

class HO():
    pars={'F0':100.,
         'omega':20.,
         'omega0':3.,
         'gamma':5.}
    
    def __init__(self,t0:float):
        """Create a instance of an harmonic oscillator
        Input:
            t0: Initial time of the simulation"""
        self.x = 3
        self.v = 0
        self.t = t0
        
    def get_second_derivative(self,t:float):
        """Returns d^2x/dt^2 = F_0 sin(omega*t) + gamma*v - omega0^2 x
        Input:
            t: The time in which we are evaluating the second derivative
        """
        return self.get_external_force(t) - self.pars['gamma']*self.v - self.x*self.pars['omega0']**2
    
    def get_external_force(self,t:float):
        """Evaluate the external force at time t
        Input:
            t: The time in which we are evaluating the second derivative
        """
        return self.pars['F0']*np.sin(self.pars['omega']*t)

    def get_energy(self, positions:np.array, velocities:np.array):
        """TO IMPLEMENT: Return total energy""" 
        return positions
    
    def get_kinectic(self, velocities:np.array):
        """TO IMPLEMENT: Return kinectic energy""" 
        return velocities
    
    def get_potential(self, positions:np.array):
        """TO IMPLEMENT: Return total energy""" 
        return positions
    
if __name__ == '__main__':
    t0=0
    tf=4
    dt=.01
    nt=int((tf-t0)/dt)
    positions=np.zeros(nt)
    velocities=np.zeros(nt)
    ho = HO(t0)

    for it, t in enumerate(np.arange(t0,tf,dt)):
        x0 = ho.x
        v0 = ho.v

        """RK4 method
        dv/dt = F(t) + gamma v(x,t)-x*omega^2 = f(x,t)
        dx/dt = v(x,t)
        """
        #First step
        k1=dt*ho.get_second_derivative(t)
        q1=dt*v0
        ho.x = x0+q1/2
        ho.v = v0+k1/2

        #Second step
        k2=dt*ho.get_second_derivative(t+dt/2)
        q2=(v0+k1/2)*dt
        ho.x = x0+q2/2
        ho.v = v0+k2/2

        #Third step
        k3=dt*ho.get_second_derivative(t+dt/2)
        q3=(v0+k2/2)*dt
        ho.x = ho.x+q3
        ho.v = ho.v+k3

        #Fourth step
        k4=dt*ho.get_second_derivative(t+dt)
        q4=(v0+k3)*dt

        ho.v = v0+(k1+2*k2+2*k3+k4)/6
        ho.x = x0+(q1+2*q2+2*q3+q4)/6

        positions[it] = ho.x
        velocities[it] = ho.v

fig, ax = plt.subplots(1,3,figsize=[15,5],gridspec_kw={'wspace':.4})
ax[0].plot(np.arange(t0,tf,dt),positions,label="Position")
ax_twin = ax[0].twinx()
ax_twin.plot(np.arange(t0,tf,dt),velocities,label="Velocity",color="tab:orange")

ax[1].plot(positions,velocities)

ax[2].plot(np.arange(t0,tf,dt),ho.get_energy(positions,velocities),label="Total")
ax[2].plot(np.arange(t0,tf,dt),ho.get_kinectic(velocities),label="Kinectic")
ax[2].plot(np.arange(t0,tf,dt),ho.get_potential(positions),label="Potential")

ax[0].set_xlabel("time (s)")
ax[0].set_ylabel("position (m)")
ax_twin.set_ylabel("velocity (m/s)")

ax[1].set_xlabel("position (m)")
ax[1].set_ylabel("velocity (m/s)")

ax[2].set_xlabel("time (s)")
ax[2].set_ylabel("Energy (J)")

ax[0].legend()
ax[2].legend()

fig.savefig("HO.png")