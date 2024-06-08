from math import inf,pi,sin,cos
from autograd import sin as asin, cos as acos
import numpy as np
#from autograd import numpy as np
import csv
import pyxel
from autograd.variable import Variable


class Robot:

    class Joint:
        def __init__(self,data):
            self.data = data
            limits = data["limits"]
            value  = data["value"]
            if not (limits[0]< value < limits[1]):
                raise Exception("value and limits are inconsistent") #TODO: find a better exception
            self.data["pose"] = [0,0,0]

        def change_value(self,value):
            m,M = self.data["limits"]
            if value < m:
                self.data["value"] = m
                print(f"WARNING: value for joint {self.data['id']} ({self.data['type']}) has been set to the limit")
            elif value > M:
                self.data["value"] = M
                print(f"WARNING: value for joint {self.data['id']} ({self.data['type']}) has been set to the limit")
            else:
                self.data["value"] = value
        def end_pose(self):
            x,y,theta = self.data["pose"]
            if self.data["type"]=="linear":
                length = self.data["length"] + self.data["value"]
            else:
                length = self.data["length"]
                theta = theta + self.data["value"]
            R = np.array([[cos(theta),-sin(theta)],
                        [sin(theta), cos(theta)]])
            X1_world = np.array([x,y])
            X2_joint = np.array([length,0])
            X2_world = X1_world+R@X2_joint
            return list(X2_world)+[theta]
        
        def end_pose_grad(self):
            x,y,theta = self.data["pose"]
            if self.data["type"]=="linear":
                length = self.data["length"] + self.data["value"]
            else:
                length = self.data["length"]
                theta = theta + self.data["value"]
            if (type(theta)==float or type(theta)==np.int64 or type(theta)==np.float64):
                ct = cos(theta)
                st = sin(theta)
            elif (type(theta)==Variable):
                ct = acos(theta)
                st = asin(theta)
            else:
                print("OOOOOOO",type(theta))

            newX = x + length*ct
            newY = y + length*st
            
            return [newX,newY,theta]
        
        def draw(self):
            pass


        value = property(fget = lambda self:self.data["value"],fset=change_value)
        end = property(end_pose)

    def __init__(self,filename):
        self.joints = []
        with open(filename,'r') as file:
            reader = csv.DictReader(file,skipinitialspace=True,delimiter=";")
            for row in reader:
                for (k,v) in row.items():
                    row[k] = eval(v)
                self.joints.append(self.Joint(row))
        self.compute_chain()

        # Compute maximum length of the robot to scale to the window
        lMax = 0
        for joint in self.joints:
            if joint.data["type"] == "linear":
                lMax+=joint.data["length"]+joint.data["limits"][1]
            else:
                lMax+=joint.data["length"]
        self.lMax = lMax
    
    def compute_chain(self):
        lastPose = [0,0,0]
        for joint in self.joints:
            joint.data["pose"]=lastPose
            lastPose = joint.end
    def compute_chain_ad(self):
        lastPose = [0,0,0]
        for joint in self.joints:
            joint.data["pose"]=lastPose
            lastPose = joint.end_pose_grad()

    def return_end(self):
        return self.joints[-1].end

    def change_joint_value(self,i,plus_minus):
        N = 20 # Number of step considered
        m,M = self.joints[i].data["limits"]
        step = (M-m)/N
        val = self.joints[i].value
        val = max(m,min(M,val))
        self.joints[i].value += step*plus_minus
        self.compute_chain()

    def direct_Model(self,q):
        for i , joint in enumerate(self.joints):
            joint.data["value"]= q[i] 
        self.compute_chain_ad()
        return self.joints[-1].end_pose_grad()
    
    end = property(return_end)


