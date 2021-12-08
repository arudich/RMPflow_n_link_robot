from Robot_Simulator.Robot import Robot
from Robot_Simulator.Obstacle import Circle

#nodes implemented in https://github.com/gtrll/multi-robot-rmpflow
from rmp import RMPRoot, RMPNode 

#nodes I added
from rmp import RevoluteJointCollisionSphere, PrismaticJointCollisionSphere, RMPTransformNode, Keep_Dims

#nodes implemented in https://github.com/gtrll/multi-robot-rmpflow
from rmp_leaf import CollisionAvoidance, GoalAttractorUni, Damper

import numpy as np
from matplotlib import pyplot as plt

class world1:
	def __init__(self):
		self.make_robot()
		self.make_obstacles()
		self.make_target()
	
	def make_robot(self):
		r = Robot()
		r.add_revolute_link(40,np.pi/4,0,-10*np.pi,10*np.pi)
		#r.add_prismatic_link(10,0,-150,150)
		r.add_revolute_link(30,np.pi/4,0,-10*np.pi,10*np.pi)
		r.add_revolute_link(30,-np.pi/4,0,-10*np.pi,10*np.pi)
		r.add_revolute_link(35,-np.pi/4,0,-10*np.pi,10*np.pi)
		#r.add_prismatic_link(10,0,-150,150)

		self.robot = r
		self.colors = ["b","b","b","b"]

	def make_obstacles(self):
		self.obstacles = []
		self.obstacles.append(Circle(np.array([50,50]),8))
		self.obstacles.append(Circle(np.array([80,60]),5))
	
	def make_target(self):
		self.target = Circle(np.array([80,40]),5)

class world2:
	def __init__(self):
		self.make_robot()
		self.make_obstacles()
		self.make_target()
	
	def make_robot(self):
		r = Robot()
		r.add_revolute_link(30,0,0,-10*np.pi,10*np.pi)
		r.add_revolute_link(30,np.pi/3,0,-10*np.pi,10*np.pi)
		r.add_revolute_link(30,-np.pi/8,0,-10*np.pi,10*np.pi)
		r.add_revolute_link(30,0,0,-10*np.pi,10*np.pi)

		self.colors = ["b","b","b","b"]

		self.robot = r

	def make_obstacles(self):
		self.obstacles = []
		self.obstacles.append(Circle(np.array([10,20]),5))
		self.obstacles.append(Circle(np.array([30,40]),5))
		self.obstacles.append(Circle(np.array([50,60]),5))

	
	def make_target(self):
		self.target = Circle(np.array([10,40]),5)

class world3:
	def __init__(self):
		self.make_robot()
		self.make_obstacles()
		self.make_target()
	
	def make_robot(self):
		r = Robot()
		r.add_revolute_link(30,np.pi/2,0,-10*np.pi,10*np.pi)
		r.add_revolute_link(30,0,0,-10*np.pi,10*np.pi)
		r.add_revolute_link(30,0,0,-10*np.pi,10*np.pi)
		r.add_revolute_link(30,0,0,-10*np.pi,10*np.pi)

		self.colors = ["b","b","b","b"]

		self.robot = r

	def make_obstacles(self):
		self.obstacles = []
		self.obstacles.append(Circle(np.array([-30,50]),5))
	
	def make_target(self):
		self.target = Circle(np.array([0,120]),5)

def anim_frame(robot,robot_poses,obstacles,target,robot_colors=None):
	plt.clf()
	robot.draw_self(robot_poses,robot_colors)
	[obstacle.draw_self(color="black") for obstacle in obstacles]
	target.draw_self(color="blue")
	plt.axis([-50,100,-50,100])
	plt.draw()
	plt.pause(.001)

def main():
	w = world1()

	robot = w.robot
	obstacles = w.obstacles
	target = w.target
	colors = w.colors

	timestep = .01

	plt.axis([-100,100,-100,100])
	plt.ion()
	plt.show()

	r = RMPRoot('root')

	n_links = len(robot.robot_links)

	for i in range(n_links):
		t_l = RMPTransformNode("T_"+str(i),r,robot.evaluate_position,i)

		t_pos = Keep_Dims("T_"+str(i)+"_pos",t_l,[0,1],3) #node with just position information
		
		if i == n_links-1:
			g = GoalAttractorUni("goal_attractor",t_pos,target.position,gain=10)
			pass
		
		d = Damper("damper",t_pos,w=.01)
		link = robot.robot_links[i]
		if not link.is_prismatic():
			n = 5
			for off in np.arange(0,link.link_length-.001,link.link_length/n):
				rev_col_sph = RevoluteJointCollisionSphere("rev_col_sphere",t_l,off)
				for obstacle in obstacles:
					ob = CollisionAvoidance("ee_obs",rev_col_sph,False,c=obstacle.position,R=obstacle.radius)
		elif link.is_prismatic():
			n = 30
			for frac in np.arange(0,1,1/n):
				pris_col_sph = PrismaticJointCollisionSphere("pris_col_sphere",t_l,frac,r,i)
				for obstacle in obstacles:
					ob = CollisionAvoidance("ee_obs",pris_col_sph,False,c=obstacle.position,R=obstacle.radius)

	for i in range(30000):
		x = robot.get_q()
		x_dot = robot.get_q_dot()
		x_ddot = r.solve(x,x_dot)
		robot.update_positions(x_ddot.flatten(),1/300)
		if i % 33 == 0:
			print(i)
			pos,_,_,_ = robot.evaluate_position()
			anim_frame(robot,pos,obstacles,target,colors)
			error = np.linalg.norm(target.position-pos[:2,-1])
			print("error:",np.linalg.norm(target.position-pos[:2,-1]))


if __name__ == "__main__":
	main()

