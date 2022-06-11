import argparse
import inspect
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicg, bicgstab, cg, cgs, gmres, minres, qmr
import os

# ... Preparate Program Execution

#################
# Set Argparse
#
problems = ["poisson", "p", "laplace", "l", "helmholtz", "h"]
solvers  = ["lu", "bicg", "bicgstab", "cg", "cgs", "gmres", "qmr"]

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str,
                    help="path of directory.")
parser.add_argument("flname", type=str,
                    help="file name of input data and output figure.")
parser.add_argument("n_x", type=int,
                    help="division number for x-axis.")
parser.add_argument("n_y", type=int,
                    help="division number for y-axis.")
parser.add_argument("prob", type=str, choices=problems,
                    help="choice the boundary-valeu problem solved by this code.")
parser.add_argument("--neumann", "-N", action="store_true",
                    help="impliment mixed boundary condition (Dirichlet and Neumann conindtion).")
parser.add_argument("--solver", "-SLV", type=str, default="lu", choices=solvers,
                    help="choice the solver for linear system.")
parser.add_argument("--epslon", type=float, default=1e-12,
                    help="convergence test value for iterative method.")
parser.add_argument("--kmax", type=int, default=200,
                    help="restart number for the GMRES method.")
parser.add_argument("--show_figure", "-fig", action="store_true",
                    help="show compulational result on X-window.")
args = parser.parse_args()


#################
# Set Hyper parameter
#
meth_type = 3
xmin = 0.0
xmax = 1.0
ymin = 0.0
ymax = 1.0
n_x  = args.n_x
n_y  = args.n_y
prob = args.prob
data_path   = args.data_path
flname      = args.flname
show_figure = args.show_figure
solver      = args.solver
kmax = args.kmax


#################
# Difine Given Equation
#
def poisson_equation(x, y):
    dk = 2.0 * np.pi
    u  =  np.sin(dk * x) * np.cos(dk * y)
    gx =  np.cos(dk * x) * np.cos(dk * y) * dk
    gy = -np.sin(dk * x) * np.sin(dk * y) * dk
    g  =  np.array([gx, gy]).T
    f  =  np.sin(dk * x) * np.cos(dk * y) * dk**2 * 2.0
    return f, g, u

def laplace_equation(x, y):
    u  = np.sin(x) * np.exp(y)
    gx = np.cos(x) * np.exp(y)
    gy = np.sin(x) * np.exp(y)
    g  =  np.array([gx, gy]).T
    f  = np.zeros_like(x)
    return f, g, u

def helmholtz_equation(x, y):
    dk =  2.0 * np.pi
    u  =  np.cos(dk * x) * np.sin(dk * y)
    gx = -np.sin(dk * x) * np.sin(dk * y) * dk
    gy =  np.cos(dk * x) * np.cos(dk * y) * dk
    g  =  np.array([gx, gy]).T
    f  =  np.zeros_like(x)
    return f, g, u, -2.0*dk**2


#################
# Get Residual Norm of Iteration Method
#
def get_resinorm(xk):
    global residual
    frame = inspect.currentframe().f_back
    residual.append(frame.f_locals['resid'])


##################
# Number of nodes, element, bounday-node, and boundaty-element
#
m_x  = n_x - 1
m_y  = n_y - 1
n_n  = n_x * n_y
n_e  = 2 * m_x * m_y
n_bn = 2 * (m_x + m_y)
n_be = 2 * (m_x + m_y)


##################
# set boundaty conditon
#
boundary_condition = np.ones(n_be, dtype="int")
boundary_condition[1:n_x-1] = int(not args.neumann)
dirichlet_node = (boundary_condition == 1)
neumann_node   = (boundary_condition == 0)


##################
# Generate Node
#
x, y = np.meshgrid(np.linspace(xmin, xmax, n_x), np.linspace(ymin, ymax, n_y))
x = x.flatten()
y = y.flatten()
if(prob in ["poisson", "p"]):
    node_value, q, u = poisson_equation(x, y)
if(prob in ["laplace", "l"]):
    node_value, q, u = laplace_equation(x, y)
if(prob in ["helmholtz", "h"]):
    node_value, q, u, dk = helmholtz_equation(x, y)
u_A = u


##################
# Generate Triangle Mesh
#
indx_lower = np.arange(n_x*m_y, dtype="int") + 1
indx_upper = np.arange(n_x*m_y, dtype="int") + n_x + 1

i1 = indx_lower[np.mod(indx_lower, n_x) != 0] - 1
i2 = indx_upper[np.mod(indx_upper, n_x) != 0] - 1 
i3 = indx_lower[np.mod(indx_lower, n_x) != 1] - 1
i4 = indx_upper[np.mod(indx_upper, n_x) != 1] - 1

element_node = np.zeros((n_e, 3), dtype="int")
element_node[0::2] = np.array((i1, i4, i2)).T
element_node[1::2] = np.array((i1, i3, i4)).T

dx1 = x[element_node[:,0]] - x[element_node[:,2]]
dy1 = y[element_node[:,0]] - y[element_node[:,2]]
dx2 = x[element_node[:,1]] - x[element_node[:,2]]
dy2 = y[element_node[:,1]] - y[element_node[:,2]]
element_area = np.abs(dx1*dy2 - dx2*dy1) / 2.0


##################
# Generate bondaty node
#
i1 = np.arange(m_x)
i2 = (np.arange(m_y) + 1)*n_x - 1
i3 = n_n - np.arange(m_x) - 1
i4 = (np.arange(m_y)[::-1] + 1) * n_x
boundary_node = np.concatenate([i1, i2, i3, i4], dtype="int")

normal = np.zeros((n_bn, 2), dtype="float")
normal[:m_x, 1] = -1.0
normal[m_x:m_x+m_y, 0] = 1.0
normal[m_x+m_y:2*m_x+m_y, 1] = 1.0
normal[2*m_x+m_y:, 0] = -1.0
print(normal[0])
boundary_value = np.zeros(n_bn)
boundary_value[dirichlet_node] = u[boundary_node[dirichlet_node]]
boundary_value[neumann_node]   = np.sum(q[boundary_node[neumann_node]]*normal[boundary_node[neumann_node]],
                                        axis=1)

##################
# Generate bondaty element
#
boundary_element_node = np.array((np.arange(n_bn),
                                  np.mod(np.arange(n_bn)+1,n_bn)), dtype="int").T

x_st = x[boundary_node[boundary_element_node[:,0]]]
x_ed = x[boundary_node[boundary_element_node[:,1]]]
y_st = y[boundary_node[boundary_element_node[:,0]]]
y_ed = y[boundary_node[boundary_element_node[:,1]]]
boundary_element_length = np.sqrt((x_st - x_ed)**2 + (y_st - y_ed)**2)


#################
# Finite Element Matrix
#
A = np.zeros((n_n, n_n))
B = np.zeros((n_n, n_n))
for e, S in enumerate(element_area):
    i = element_node[e,0]
    j = element_node[e,1]
    k = element_node[e,2]

    b0 = y[j] - y[k]
    b1 = y[k] - y[i]
    b2 = y[i] - y[j]
    c0 = x[k] - x[j]
    c1 = x[i] - x[k]
    c2 = x[j] - x[i]
    
    A[i, i] += (b0*b0 + c0*c0) / (4.0 * S)
    A[i, j] += (b0*b1 + c0*c1) / (4.0 * S)
    A[i, k] += (b0*b2 + c0*c2) / (4.0 * S)
    A[j, i] += (b1*b0 + c1*c0) / (4.0 * S)
    A[j, j] += (b1*b1 + c1*c1) / (4.0 * S)
    A[j, k] += (b1*b2 + c1*c2) / (4.0 * S)
    A[k, i] += (b2*b0 + c2*c0) / (4.0 * S)
    A[k, j] += (b2*b1 + c2*c1) / (4.0 * S)
    A[k, k] += (b2*b2 + c2*c2) / (4.0 * S)

    B[i, i] += S /  6.0
    B[i, j] += S / 12.0
    B[i, k] += S / 12.0
    B[j, i] += S / 12.0
    B[j, j] += S /  6.0
    B[j, k] += S / 12.0
    B[k, i] += S / 12.0
    B[k, j] += S / 12.0
    B[k, k] += S /  6.0

C = np.zeros((n_n, n_n))
for e, h in enumerate(boundary_element_length):
    i = boundary_node[boundary_element_node[e,0]]
    j = boundary_node[boundary_element_node[e,1]]
    
    if boundary_condition[boundary_element_node[e,0]] + boundary_condition[boundary_element_node[e,1]] == 0:
        C[i, i] += h / 3.0
        C[i, j] += h / 6.0
        C[j, i] += h / 6.0
        C[j, j] += h / 3.0


#################
# Cofficient Matrix of Helmholtz Problem
#
if(prob in ["helmholtz", "h"]):
    A = A + dk * B

#################
# Right-Hand Side
#
f = np.dot(B, node_value)


#################
# Boundary Condition
#
d = np.zeros(n_n)
d[boundary_node[dirichlet_node]] = boundary_value[dirichlet_node]

b = f - np.dot(A, d)
b[boundary_node[dirichlet_node]] = boundary_value[dirichlet_node]

A[boundary_node[dirichlet_node],:] = 0.0
A[:,boundary_node[dirichlet_node]] = 0.0
A[boundary_node[dirichlet_node],boundary_node[dirichlet_node]] = 1.0

d = np.zeros(n_n)
d[boundary_node[neumann_node]] = boundary_value[neumann_node]
b += np.dot(C, d)


#################
# Solve Linear System
#
if(solver == "lu"):
    A_lu, piv = lu_factor(A)
    u_N = lu_solve((A_lu,piv), b)
else:
    A = csc_matrix(A)
    residual = []
    x0 = np.zeros(n_n)
    epsln = args.epslon


if(solver == "bicg"):
    u_N, info = bicg(A, b, x0=x0, tol=epsln, callback=get_resinorm)
if(solver == "bicgstab"):
    u_N, info = bicgstab(A, b, x0=x0, tol=epsln, callback=get_resinorm)
if(solver == "cg"):
    u_N, info = cg(A, b, x0=x0, tol=epsln, callback=get_resinorm)
if(solver == "cgs"):
    u_N, info = cgs(A, b, x0=x0, tol=epsln, callback=get_resinorm)
if(solver == "gmres"):
    u_N, info = gmres(A, b, x0=x0, tol=epsln, restart=kmax, callback=get_resinorm)
if(solver == "qmr"):
    u_N, info = qmr(A, b, x0=x0, tol=epsln, callback=get_resinorm)


#################
# Output Computational Result
#
rerror = np.max(np.abs(u_N - u_A)) / np.max(np.abs(u_A))
print("Number of Nodes:\t", n_n)
print("Relative Error :\t", rerror)
if(not solver == "lu"):
    print("Number of Iterlations:\t", len(residual))

data   = np.array((x, u_N, u_A)).T
np.savetxt(os.path.join(data_path, flname+"_solution.txt"),
           data,fmt="%.18e", delimiter=" ",  header="x u_N u_A")

if(not solver == "lu"):
    data   = np.array((np.arange(len(residual))+1, residual)).T
    np.savetxt(os.path.join(data_path, flname+"_residual.txt"),
               data,fmt="%.18e", delimiter=" ",  header="iter residual")
fig, ax = plt.subplots()
ax.scatter(np.arange(n_n), u_N, color="blue", label="Numerical")
ax.plot(np.arange(n_n), u_A, color="red", label="Analytical")
ax.tick_params(direction="in") 
ax.tick_params(which="minor", length=0.0) 
ax.set_title("Solution of Boundary-Value Problem")
ax.set_xlabel("Node Number, $n$")
ax.set_ylabel("$u$")
plt.legend()
plt.savefig(os.path.join(data_path, flname+"_solution.png"))
if(show_figure):
    plt.show()
plt.close()

fig, ax = plt.subplots()
ax.plot(np.arange(n_n), np.abs(u_A - u_N) / np.max(np.abs(u_A)), color="red")
ax.tick_params(direction="in") 
ax.tick_params(which="minor", length=0.0) 
ax.set_title("Relative Error")
ax.set_xlabel("Node Number, $n$")
ax.set_ylabel("$||u_N - u_A||_\infty ~/~ ||u_A||_\infty$")
plt.savefig(os.path.join(data_path, flname+"_rerror.png"))
if(show_figure):
    plt.show()
plt.close()

if(not solver == "lu"):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(residual)), residual, color="blue")
    ax.tick_params(direction="in")
    ax.tick_params(which="minor", length=0.0) 
    ax.set_title("Residual History")
    ax.set_xlabel("Iteration Number")
    ax.set_ylabel("Residual Norm, $||Ab - x||_\infty$")
    ax.set_yscale("log")
    plt.savefig(os.path.join(data_path, flname+"_residual.png"))
    if(show_figure):
        plt.show()
    plt.close()
