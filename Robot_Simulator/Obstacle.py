import numpy as np
from matplotlib import pyplot as plt

class Obstacle():
    def __init__(self,position):
        self.position = position

    def dist_to_point(self,point):
        np.linalg.norm(point-self.position)

    def set_position(self,new_pos):
        self.position = new_pos
    
    def perturb_position(self,perturbation):
        self.position += perturbation

class Circle(Obstacle):
    def __init__(self,position,radius):
        self.position = position
        self.radius = radius
    
    def dist_to_point(self,point):
        return min(np.linalg.norm(point-self.position) - self.radius,0)

    def draw_self(self,color="r"):
        c = plt.Circle((self.position[0],self.position[1]),self.radius,color=color)
        ax = plt.gca()
        ax.add_patch(c)

    