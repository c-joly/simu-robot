import pyxel
import robot as rob
from math import cos,sin,pi
from time import time
import numpy as np
from random import random,shuffle,randint
import autograd as ad
from autograd.variable import Variable
from scipy.integrate import solve_ivp
from scipy.signal import place_poles


###########
# CONSTANTS
HEIGHT = 700
WIDTH = 700
YELLOW = 10
WHITE = 7
BLACK = 0
RED = 8
GREY=13
GREEN = 11
BLUE = 5

# Load a robot
filename = "rob_RRR.csv"
robot = rob.Robot(filename) # Robot for display
robot_grad = rob.Robot(filename) # Garbage for computing (and prevent display to be affected)

ARM_WIDTH = robot.lMax/10 # to represent the robot's parts with a pretty width
SCALE = min(HEIGHT,WIDTH)/2/(robot.lMax*1.1)
CENTER = (WIDTH/2,HEIGHT/2)
SIZE_CIRCLE = ARM_WIDTH*SCALE/4



# Define the joint coordinates in a meshgrid format
l  = []
l_simple = []
N = 50 # Number of point per joint
for joint in robot.joints:
    m,M = joint.data["limits"]
    if joint.data["type"]=="rotule":
        val = (np.linspace(m,M,N))
    else:
        val = np.linspace(m,M,max(round(N/20),4))
    val1 = val[0::2]
    val2 = val[1::2]
    val2 = val2[::-1]
    l_simple.append(val)
    val = np.concatenate((val1,val2))
    l.append(val)
l_ini = l.copy()
q = np.meshgrid(*l)
print(np.shape(q))
q = [item.flatten() for item in q]



print("Precomputing free space vizualisation - Please wait...")
precomp_free = set()
# Monte Carlo simulation
## TODO
## Put in precom_free the free space in pyxel coordinates


print("Done...")
# restore initial joint values


focus = 0 # Which joint is movable
print_free = True # By default we print the free space in green
automatic_free_space = False  # To generate a trajectory that scans all the joints values
automatic_trajectory = False  # To draw a predefined trajectory (rectangle currently) starting at the good initial condition
closed_loop_trajectory = False # Following the previous trajectory with arbitrary initial condition

# Two trajectory to  display
current_free_space = []  # Trajectory for scanning all the free space
current_traj = [] # For open / closed loop trajectories

auto_index = 0
angle = 0
stop = False


    
id = list(range(len(l)))   

dir = [True for _ in l_simple]
cur_id = 0
frame_traj = 0

def update():
    global focus,print_free,automatic_free_space,automatic_trajectory,auto_index,q,id,l,dir,cur_id
    global frame_traj,current_traj,closed_loop_trajectory,traj
    if pyxel.btnp(pyxel.KEY_Q):
        pyxel.quit()
    if pyxel.btnp(pyxel.KEY_F): #Toggle Free Space Visualization
        print_free = not print_free
    if pyxel.btnp(pyxel.KEY_A): #Toggle Free Space Visualization
        automatic_free_space = not automatic_free_space
        automatic_trajectory = False
        closed_loop_trajectory = False
    if pyxel.btnp(pyxel.KEY_T): #Toggle automatic trajectory
        automatic_trajectory = not automatic_trajectory
        automatic_free_space = False
        closed_loop_trajectory = False
        if automatic_trajectory:
            current_traj = []
            frame_traj = 0
            traj = q_ref
    if pyxel.btnp(pyxel.KEY_C): #Toggle closed loop trajectory
        closed_loop_trajectory = not closed_loop_trajectory
        automatic_free_space = False
        automatic_trajectory = False
        current_traj = []
        frame_traj = 0
        if closed_loop_trajectory:
            q0 = [qi.data["value"] for qi in robot.joints]
            closed_loop = solve_ivp(fun= u1,y0=q0,t_span=[0,tf],dense_output=True,rtol=1e-9,atol=1e-9)
            q_closed_loop = np.array(closed_loop["sol"](t))
            traj = q_closed_loop

    if pyxel.btnp(pyxel.KEY_LEFT):
        focus = (focus-1) % len(robot.joints)
    if pyxel.btnp(pyxel.KEY_RIGHT):
        focus = (focus+1) % len(robot.joints)
    if pyxel.btnp(pyxel.KEY_UP):
        robot.change_joint_value(focus,1)
    if pyxel.btnp(pyxel.KEY_DOWN):
        robot.change_joint_value(focus,-1)


    if automatic_free_space:
        if auto_index == len(q[0]):
            auto_index = 0
            id = list(range(len(l)))
            shuffle(id)
            l = [l_ini[i] for i in id]

            q = np.meshgrid(*l)
            q = [item.flatten() for item in q]

        for j,joint in enumerate(robot.joints):
            joint.data["value"] = q[id.index(j)][auto_index]
        robot.compute_chain()
        auto_index += 1
        (x,y,_) = robot.end
        x = (CENTER[0]+x*SCALE)
        y = (CENTER[1]-y*SCALE)
        current_free_space.append((x,y))

    if automatic_trajectory or closed_loop_trajectory:
        if frame_traj<len(traj[0]):
            for j,joint in enumerate(robot.joints):
                joint.data["value"] = traj[j][frame_traj]
            robot.compute_chain()
            frame_traj += 1
            (x,y,_) = robot.end
            x = (CENTER[0]+x*SCALE)
            y = (CENTER[1]-y*SCALE)
            current_traj.append((x,y))


def draw():
    global angle,t1,t0
    t1=time()
    #print(1/(t1-t0))
    t0 = t1  
    pyxel.cls(WHITE)
    drawRobot()
    if automatic_free_space:
        drawTraj(traj=current_free_space,color=BLUE)
    if automatic_trajectory or closed_loop_trajectory:
        drawTraj(traj=current_traj,color=BLUE)
        x_target = x_ref(frame_traj*1/100)
        y_target = y_ref(frame_traj*1/100)
        pyxel.circb(CENTER[0]+x_target*SCALE,CENTER[1]-y_target*SCALE,10,BLUE)
    if print_free:
        draw_precom_free()
    drawAxes()

def drawRect(x,y,w,h,alpha,color,circColor=BLACK):
    (x1,y1) = x-h/2*sin(alpha),y+h/2*cos(alpha)
    (x2,y2) = x1+w*cos(alpha),y1+w*sin(alpha)
    (x4,y4) = x+h/2*sin(alpha),y-h/2*cos(alpha)
    (x3,y3) = x4+w*cos(alpha),y4+w*sin(alpha)
    pyxel.line(x1,y1,x2,y2,BLACK)
    pyxel.line(x2,y2,x3,y3,BLACK)
    pyxel.line(x3,y3,x4,y4,BLACK)
    pyxel.line(x4,y4,x1,y1,BLACK)
    pyxel.fill(x+w*0.7*cos(alpha),y+w*.7*sin(alpha),color)
    pyxel.circ(x,y,SIZE_CIRCLE,circColor)

def drawRobot():
    # Will use the global variable robot
    for i,joint in enumerate(robot.joints):
        (x,y,alpha) = joint.data["pose"]
        if joint.data["type"]=="rotule":
            alpha = alpha+joint.data["value"]
            length = joint.data["length"]
            color = RED
        else:
            length = joint.data["length"]+joint.data["value"]
            color = GREY
        x = CENTER[0]+x*SCALE
        y = CENTER[1]-y*SCALE
        w = length*SCALE
        h = ARM_WIDTH*SCALE
        if focus == i and not automatic_free_space and not automatic_trajectory:
            drawRect(x,y,w,h,-alpha,color,circColor=YELLOW)
        else:
            drawRect(x,y,w,h,-alpha,color)
        
    

def drawAxes(color = BLACK):
    #Plot x
    for i in range(WIDTH):
        if pyxel.pget(i,HEIGHT/2)==WHITE:
            pyxel.pset(i,HEIGHT/2,BLACK)
    for i in range(HEIGHT):
        if pyxel.pget(WIDTH/2,i)==WHITE:
            pyxel.pset(WIDTH/2,i,BLACK)

def drawTraj(traj,color=GREEN):
    #Plot x
    first = traj[0]
    for px in traj:
        x0,y0 = first
        x1,y1 = px
        pyxel.line(x0,y0,x1,y1,color)
        first = px
        #if pyxel.pget(x,y)==WHITE:
        #    pyxel.circ(x,y,SIZE_CIRCLE/2,color)
def draw_precom_free(color = GREEN):
    #Plot x
    for x,y in precomp_free:
        if pyxel.pget(x,y)==WHITE:
            pyxel.pset(x,y,color)
            #pyxel.circ(x,y,2,color)

########################
# NUMERICAL STUFF

def jacobian(x):
    x = Variable(x)
    direct = robot_grad.direct_Model(x)
    for item in direct:
        item.compute_gradients()
    
    grad = [list(k.gradient) for k in direct]
    jaco = np.row_stack(grad)
    return jaco

######################
# REF TRAJECTORY

tf = 16
speed = 0.4/2
def xd(t):
    return 0
def yd(t):
    return 0

x0,y0 = 2,1
def x_ref(t):
    return x0

def y_ref(t):
    return y0

q0 = [0,pi/2,-pi/2]
def q_d(t,q):
    return np.array(q)*0

jacobian(np.array(q0))
open_loop = solve_ivp(fun= q_d,y0=q0,t_span=[0,tf],dense_output=True,rtol=1e-6,atol=1e-6)
t = np.arange(0,tf,1/100)
q_ref = np.array(open_loop["sol"](t))

########################
# Closed loop trajectory
poles = np.array([-1,-1.1,-1.2])
def u1(t,q):
    return q_d(t,q)


    
pyxel.init(WIDTH, HEIGHT,fps=100)
pyxel.run(update, draw)