from Robot import Robot
from Obstacle import Circle
import numpy as np
import time
from matplotlib import pyplot as plt

r = Robot()
r.add_prismatic_link(10,0,0,150)
r.add_revolute_link(30,np.pi/8,0,0,np.pi)
r.add_revolute_link(30,-np.pi/8,0,-np.pi,np.pi)

c = Circle(np.array([30,30]),10)

q_ddot = np.array([10,1,1])
timestep = .01

plt.axis([-100,100,-100,100])
plt.ion()
plt.show()

for i in range(5000):
    poses,jacobians = r.evaluate_position()
    breakpoint()
    plt.clf()
    r.draw_self(poses)
    c.draw_self()
    plt.axis([-100,100,-100,100])
    plt.draw()
    plt.pause(.001)
    #time.sleep(timestep)
    r.update_positions(q_ddot,timestep)

    