"""
This file was copied and modified from https://github.com/gtrll/multi-robot-rmpflow

The copied code lays out the framework for controlling single-point
robots that are actuated directly by x,y coordinates

I added RMPTransformNodes and RMPCollisionSphereNodes, which are added to the computational
graph of RMP nodes in order to control an arbitrary 2D n-link robot.
"""


# RMPflow basic classes
# @author Anqi Li
# @date April 8, 2019

import numpy as np

class RMPNode:
	"""
	A Generic RMP node
	"""
	def __init__(self, name, parent, psi, J, J_dot, verbose=False):
		self.name = name

		self.parent = parent
		self.children = []

		# connect the node to its parent
		if self.parent:
			self.parent.add_child(self)

		# mapping/J/J_dot for the edge from the parent to the node
		self.psi = psi
		self.J = J
		self.J_dot = J_dot

		# state
		self.x = None
		self.x_dot = None

		# RMP
		self.f = None
		self.a = None
		self.M = None

		# print the name of the node when applying operations if true
		self.verbose = verbose


	def add_child(self, child):
		"""
		Add a child to the current node
		"""

		self.children.append(child)


	def pushforward(self):
		"""
		apply pushforward operation recursively
		"""

		if self.verbose:
			print('%s: pushforward' % self.name)

		if self.psi is not None and self.J is not None:
			self.x = self.psi(self.parent.x)
			self.x_dot = np.dot(self.J(self.parent.x), self.parent.x_dot)

			assert self.x.ndim == 2 and self.x_dot.ndim == 2

		[child.pushforward() for child in self.children]



	def pullback(self):
		"""
		apply pullback operation recursively
		"""

		[child.pullback() for child in self.children]

		if self.verbose:
			print('%s: pullback' % self.name)

		f = np.zeros_like(self.x, dtype='float64')
		M = np.zeros((max(self.x.shape), max(self.x.shape)),
			dtype='float64')

		for child in self.children:
			
			J_child = child.J(self.x)
			J_dot_child = child.J_dot(self.x, self.x_dot)

			assert J_child.ndim == 2 and J_dot_child.ndim == 2

			if child.f is not None and child.M is not None:
				f += np.dot(J_child.T , (child.f - np.dot(np.dot(child.M, J_dot_child), self.x_dot)))
				M += np.dot(np.dot(J_child.T, child.M), J_child)

		self.f = f
		self.M = M
	
	def is_transform_node(self):
		return False

class RMPTransformNode(RMPNode):

	def __init__(self, name, parent, evaluate_position, link_idx, verbose=False):
		self.name = name

		self.parent = parent
		self.children = []

		# connect the node to its parent
		if self.parent:
			self.parent.add_child(self)

		# mapping/J/J_dot for the edge from the parent to the node
		self.evaluate_position = evaluate_position
		self.link_idx = link_idx

		# state
		self.x = None
		self.x_dot = None

		# RMP
		self.f = None
		self.a = None
		self.M = None

		# print the name of the node when applying operations if true
		self.verbose = verbose



	def add_child(self, child):
		"""
		Add a child to the current node
		"""

		self.children.append(child)


	def pushforward(self):
		"""
		apply pushforward operation recursively
		"""

		if self.verbose:
			print('%s: pushforward' % self.name)

		self.x,self.x_dot,_,_ = self.evaluate_position(self.link_idx)
		assert self.x.ndim == 2 and self.x_dot.ndim == 2

		[child.pushforward() for child in self.children]

	def get_state(self):
		return self.evaluate_position(self.link_idx)
	
	def is_transform_node(self):
		return True

class PrismaticJointCollisionSphere(RMPNode):
	def __init__(self,name,parent,frac,root,idx,verbose=False):
		"""
		frac: fraction of prismatic link length to travel down
		root: root of RMP graph
		idx: index of primatic joint q in root.x
		"""
		self.root = root
		psi = lambda y : y[:2] - frac*self.root.x[idx]*np.array([np.cos(y[2]),np.sin(y[2])])
		def J(y):
			mat = np.zeros((2,3))
			mat[0,0] = 1
			mat[1,1] = 1
			mat[0,2] = self.root.x[idx]*np.sin(y[2])
			mat[1,2] = -self.root.x[idx]*np.cos(y[2])
			return mat
		
		def J_dot(y,y_dot):
			mat = np.zeros((2,3))
			mat[0,2] = self.root.x[idx]*np.cos(y[2])*y_dot[2]
			mat[1,2] = self.root.x[idx]*np.sin(y[2])*y_dot[2]
			return mat

		super().__init__(name, parent, psi, J, J_dot, verbose=verbose)


class RevoluteJointCollisionSphere(RMPNode):
	def __init__(self,name,parent,offset_size,verbose=False):
		self.offset_size = offset_size
		psi = lambda y : y[:2] - self.offset_size*np.array([np.cos(y[2]),np.sin(y[2])])
		def J(y):
			mat = np.zeros((2,3))
			mat[0,0] = 1
			mat[1,1] = 1
			mat[0,2] = self.offset_size*np.sin(y[2])
			mat[1,2] = -self.offset_size*np.cos(y[2])
			return mat
		
		def J_dot(y,y_dot):
			mat = np.zeros((2,3))
			mat[0,2] = self.offset_size*np.cos(y[2])*y_dot[2]
			mat[1,2] = self.offset_size*np.sin(y[2])*y_dot[2]
			return mat

		super().__init__(name, parent, psi, J, J_dot, verbose=verbose)

class Keep_Dims(RMPNode):
	"""
	Remove specified dimensions from x and x_dot in RMP node
	"""

	def __init__(self, name, parent, keep_dims, num_parent_dims, verbose=False):
		self.keep_dims = keep_dims

		psi = lambda y : y[keep_dims]
		mat = np.zeros((len(keep_dims),num_parent_dims))
		for i,d in enumerate(keep_dims):
			mat[i,d] = 1
		J = lambda y : mat
		J_dot = lambda x,y : np.zeros_like(mat)

		super().__init__(name, parent, psi, J, J_dot, verbose=verbose)

class RMPRoot(RMPNode):
	"""
	A root node
	"""

	def __init__(self, name):
		RMPNode.__init__(self, name, None, None, None, None)

	def set_root_state(self, x, x_dot):
		"""
		set the state of the root node for pushforward
		"""

		assert x.ndim == 1 or x.ndim == 2
		assert x_dot.ndim == 1 or x_dot.ndim == 2

		if x.ndim == 1:
			x = x.reshape(-1, 1)
		if x_dot.ndim == 1:
			x_dot = x_dot.reshape(-1, 1)

		self.x = x
		self.x_dot = x_dot


	def pushforward(self):
		"""
		apply pushforward operation recursively
		"""

		if self.verbose:
			print('%s: pushforward' % self.name)

		[child.pushforward() for child in self.children]


	def resolve(self):
		"""
		compute the canonical-formed RMP
		"""

		if self.verbose:
			print('%s: resolve' % self.name)

		self.a = np.dot(np.linalg.pinv(self.M), self.f)
		
		return self.a

	def pullback(self):
		"""
		apply pullback operation recursively
		"""

		[child.pullback() for child in self.children]

		if self.verbose:
			print('%s: pullback' % self.name)

		f = np.zeros_like(self.x, dtype='float64')
		M = np.zeros((max(self.x.shape), max(self.x.shape)),
			dtype='float64')

		for child in self.children:
			if child.is_transform_node():
				_,_,J_child,J_dot_child = child.get_state()
			else:
				J_child = child.J(self.x)
				J_dot_child = child.J_dot(self.x, self.x_dot)

			assert J_child.ndim == 2 and J_dot_child.ndim == 2

			if child.f is not None and child.M is not None:
				f += np.dot(J_child.T , (child.f - np.dot(
					np.dot(child.M, J_dot_child), self.x_dot)))
				M += np.dot(np.dot(J_child.T, child.M), J_child)

		self.f = f
		self.M = M

	def solve(self, x, x_dot):
		"""
		given the state of the root, solve for the controls
		"""

		self.set_root_state(x, x_dot)
		self.pushforward()
		self.pullback()
		x_ddot = self.resolve()
		return x_ddot


class RMPLeaf(RMPNode):
	"""
	A leaf node
	"""

	def __init__(self, name, parent, parent_param, psi, J, J_dot, RMP_func):
		RMPNode.__init__(self, name, parent, psi, J, J_dot)
		self.RMP_func = RMP_func
		self.parent_param = parent_param


	def eval_leaf(self):
		"""
		compute the natural-formed RMP given the state
		"""
		self.f, self.M = self.RMP_func(self.x, self.x_dot)


	def pullback(self):
		"""
		pullback at leaf node is just evaluating the RMP
		"""

		if self.verbose:
			print('%s: pullback' % self.name)

		self.eval_leaf()

	def add_child(self, child):
		print("CANNOT add a child to a leaf node")
		pass


	def update_params(self):
		"""
		to be implemented for updating the parameters
		"""
		pass


	def update(self):
		self.update_params()
		self.pushforward()
