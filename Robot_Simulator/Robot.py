import numpy as np
from matplotlib import pyplot as plt

class Link():
    def __init__(self):
        pass

    def update_position(self,q_ddot,timestep):
        self.q += self.q_dot*timestep
        self.q_dot += q_ddot*timestep

        self.q = min(max(self.q,self.q_min),self.q_max)
    
    def set_q(self,q):
        self.q = q

    def set_q_dot(self,q_dot):
        self.q_dot = q_dot


class Revolute_Link(Link):
    def __init__(self,link_length, q, q_dot, q_min, q_max):
        self.link_length = link_length
        self.q = q #angle
        self.q_dot = q_dot #angular velocity

        self.q_min = q_min
        self.q_max = q_max
    
    def get_angle(self):
        return self.q
    
    def get_angle_vel(self):
        return self.q_dot

    def get_end_pos(self,base_pos,base_pos_vel):
        base_rot = base_pos[2]
        base_pos = base_pos[:2]
        base_rot_vel = base_pos_vel[2]
        base_pos_vel = base_pos_vel[:2]

        end_ang = self.q+base_rot
        end_ang_vel = self.q_dot + base_rot_vel
        pos = base_pos + np.array([self.link_length*np.cos(end_ang),self.link_length*np.sin(end_ang)])
        pos_dot = base_pos_vel + self.link_length*end_ang_vel*np.array([-np.sin(end_ang),np.cos(end_ang)])

        jac_part_rev = (self.link_length*np.array([-np.sin(self.q+base_rot),np.cos(self.q+base_rot)])).reshape((2,1))
        jac_part_rev_dot = (self.link_length*(base_rot_vel+self.q_dot)*np.array([-np.cos(self.q+base_rot),-np.sin(self.q+base_rot)])).reshape((2,1))

        return pos,pos_dot,jac_part_rev,jac_part_rev_dot

    def is_prismatic(self):
        return False

class Prismatic_Link(Link):
    def __init__(self, q, q_dot, q_min, q_max):
        self.q = q #length
        self.q_dot = q_dot #velocity

        self.q_min = q_min
        self.q_max = q_max

    def get_angle(self):
        return 0
    
    def get_angle_vel(self):
        return 0
    
    def get_end_pos(self,base_pos,base_pos_vel):
        base_rot = base_pos[2]
        base_pos = base_pos[:2]
        base_rot_vel = base_pos_vel[2]
        base_pos_vel = base_pos_vel[:2]

        pos = base_pos + self.q*np.array([np.cos(base_rot),np.sin(base_rot)])
        pos_dot = base_pos_vel + self.q_dot*np.array([np.cos(base_rot),np.sin(base_rot)]) + self.q*base_rot_vel*np.array([np.sin(base_rot),np.cos(base_rot)])

        jac_part_rev = (self.q*np.array([-np.sin(base_rot),np.cos(base_rot)])).reshape((2,1))
        jac_part_rev_dot = (self.q_dot*np.array([-np.sin(base_rot),np.cos(base_rot)]).reshape((2,1)) -
                                self.q*base_rot_vel*np.array([np.cos(base_rot),np.sin(base_rot)]).reshape((2,1)))

        return pos,pos_dot,jac_part_rev,jac_part_rev_dot
    
    def is_prismatic(self):
        return True


class Robot():
    def __init__(self):
        self.robot_links = []

    def add_revolute_link(self,link_length, q, q_dot, q_min, q_max):
        self.robot_links.append(Revolute_Link(link_length, q, q_dot, q_min, q_max))
    
    def add_prismatic_link(self, q, q_dot, q_min, q_max):
        self.robot_links.append(Prismatic_Link(q, q_dot, q_min, q_max))

    def set_robot_pose(self,qs):
        assert(len(qs) == len(self.robot_links))
        for i,q in enumerate(qs):
            self.robot_links[i].set_q(q)
    
    def set_robot_vel(self,q_dots):
        assert(len(q_dots) == len(self.robot_links))
        for i,q_dot in enumerate(q_dots):
            self.robot_links[i].set_q_dot(q_dot)

    def get_q(self):
        return np.array([link.q for link in self.robot_links])
    
    def get_q_dot(self):
        return np.array([link.q_dot for link in self.robot_links])
    
    def evaluate_position(self,idx=None):
        '''
        returns
            position: end of each link
            jacobians: jacobian of each link
        '''
        n_links = len(self.robot_links)

        total_angle = 0
        total_angle_vel = 0

        poses = np.zeros((3,n_links))
        pose_dots = np.zeros((3,n_links))

        jacobians = np.zeros((n_links,3,n_links))
        jacobian_dots = np.zeros((n_links,3,n_links))

        rev_mask = np.zeros((2,n_links))
        for i,link in enumerate(self.robot_links):
            if not link.is_prismatic():
                rev_mask[:,i] = np.ones(2)
                jacobians[i,2,:] = 1
                jacobian_dots[i,2,:] = 0

            pos,pos_dot,jac_part,jac_dot_part = link.get_end_pos(poses[:,i-1],pose_dots[:,i-1])
            total_angle += link.get_angle()
            total_angle_vel += link.get_angle_vel()

            link_jac = jac_part*rev_mask
            link_jac_dot = jac_dot_part*rev_mask

            if link.is_prismatic():
                link_jac[:,i] = np.array([np.cos(total_angle),np.sin(total_angle)])
                link_jac_dot[:,i] = total_angle_vel * np.array([-np.sin(total_angle),np.cos(total_angle)])

            poses[:2,i] = pos
            poses[2,i] = total_angle
            pose_dots[:2,i] = pos_dot
            pose_dots[2,i] = total_angle_vel

            jacobians[i,:2,:] = jacobians[i-1,:2,:] + link_jac
            jacobian_dots[i,:2,:] = jacobian_dots[i-1,:2,:] + link_jac_dot

            if idx is not None and i == idx:
                break

        if idx is not None:
            return poses[:,idx].reshape((-1,1)), pose_dots[:,idx].reshape((-1,1)), jacobians[idx], jacobian_dots[idx]
        else:
            return poses, pose_dots, jacobians, jacobian_dots

    def update_positions(self,q_ddot,timestep):
        for i,link in enumerate(self.robot_links):
            link.update_position(q_ddot[i],timestep)

    def draw_self(self,robot_link_poses,colors=None):

        base_pt = np.zeros(2)
        for i in range(robot_link_poses.shape[1]):
            if colors is None:
                color = "blue"
            else:
                color = colors[i]
            link_pos = robot_link_poses[:,i]
            plt.plot([base_pt[0],link_pos[0]],[base_pt[1],link_pos[1]],color=color)
            c = plt.Circle((link_pos[0],link_pos[1]), 2, color='red')
            ax = plt.gca()
            ax.add_patch(c)
            base_pt = link_pos

if __name__=="__main__":
    r = Robot()
    r.add_prismatic_link(10,1,0,150)
    r.add_revolute_link(30,np.pi/8,.1,0,np.pi)
    r.add_revolute_link(30,-np.pi/8,.1,-np.pi,np.pi)
    r.add_prismatic_link(10,1,0,150)

    poses,pose_dots,jacobians,jacobian_dots = r.evaluate_position(3)
    print(r.get_q_dot())
    
    r.update_positions(np.zeros(4),.1)
    p2,pds,j2,jd2 = r.evaluate_position(3)
    print(r.get_q_dot())

    r.update_positions(np.zeros(4),.1)
    p3,pds3,j3,jd3 = r.evaluate_position(3)
    print(r.get_q_dot())
    breakpoint()



