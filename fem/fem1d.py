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
parser.add_argument("n_n", type=int,
                    help="number of nodes.")
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
# Set Hyper Parameter
#
xmin = 0.0
xmax = 1.0
n_n  = args.n_n
prob = args.prob
data_path   = args.data_path
flname      = args.flname
show_figure = args.show_figure
solver      = args.solver
kmax = args.kmax


#################
# Check Directory
#
if not os.path.exists(data_path):
    os.makedirs(data_path)


#################
# Get Residual Norm of Iteration Method
#
def get_resinorm(xk):
    global residual
    frame = inspect.currentframe().f_back
    residual.append(frame.f_locals['resid'])


#################
# Difine Given Equation
#
def poisson_equation(x):
    dk = 2.0 * np.pi
    u = np.sin(dk * x)
    g = np.cos(dk * x) * dk
    f = np.sin(dk * x) * dk**2
    return f, g, u

def laplace_equation(x):
    u  = 2.0 * x + 1.0
    g  = 2.0 * np.ones_like(x)
    f  = np.zeros_like(x)
    return f, g, u

def helmholtz_equation(x):
    dk = 2.0 * np.pi
    u  =  np.cos(dk * x)
    g  = -np.sin(dk * x) * dk
    f  =  np.zeros_like(x)
    return f, g, u, -dk**2


# ... Generate Finite Element

##################
# Number of nodes, element, bounday-node, and boundaty-element
#
n_e  = n_n - 1
n_bn = 2

##################
# Imptiment Boundary Condition
#
boundary_condition = np.array([1, int(not args.neumann)], dtype="int")
dirichlet_node = (boundary_condition == 1)
neumann_node   = (boundary_condition == 0)


##################
# Generate Node
#
x = np.linspace(xmin, xmax, n_n)
if(prob in ["poisson", "p"]):
    node_value, q, u = poisson_equation(x)
if(prob in ["laplace", "l"]):
    node_value, q, u = laplace_equation(x)
if(prob in ["helmholtz", "h"]):
    node_value, q, u, dk = helmholtz_equation(x)
u_A = u


##################
# Generate Element
#
element_node = np.array((np.arange(n_e), np.arange(n_e)+1), dtype="int").T
element_length = x[element_node[:,1]] - x[element_node[:,0]]
element_value  = np.ones(n_e)


##################
# Generate Boundary Node
#
boundary_node  = np.array([0, n_n-1], dtype="int")

boundary_value = np.zeros(n_bn)
boundary_value[dirichlet_node] = u[boundary_node[dirichlet_node]]
boundary_value[neumann_node]   = q[boundary_node[neumann_node]]


#################
# Finite Element Matrix
#
A = np.zeros((n_n, n_n))
B = np.zeros((n_n, n_n))
for e, h in enumerate(element_length):
    i = element_node[e,0]
    j = element_node[e,1]
    
    A[i, i] += 1.0 / h
    A[j, j] += 1.0 / h
    A[i, j] -= 1.0 / h
    A[j, i] -= 1.0 / h

    B[i, i] += h / 3.0
    B[j, j] += h / 3.0
    B[i, j] += h / 6.0
    B[j, i] += h / 6.0


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
b[boundary_node[neumann_node]]  += boundary_value[neumann_node]

A[boundary_node[dirichlet_node],:] = 0.0
A[:,boundary_node[dirichlet_node]] = 0.0
A[boundary_node[dirichlet_node],boundary_node[dirichlet_node]] = 1.0


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


#################
# Visualization Computational Result
#
fig, ax = plt.subplots()
ax.scatter(x, u_N, color="blue", label="Numerical")
ax.plot(x, u_A, color="red", label="Analytical")
ax.tick_params(direction="in") 
ax.tick_params(which="minor", length=0.0) 
ax.set_title("Solution of Boundary-Value Problem")
ax.set_xlabel("$x$")
ax.set_ylabel("$u$")
plt.legend()
plt.savefig(os.path.join(data_path, flname+"_solution.png"))
if(show_figure):
    plt.show()
plt.close()

fig, ax = plt.subplots()
ax.plot(x, np.abs(u_N - u_A) / np.max(np.abs(u_A)), color="red")
ax.tick_params(direction="in") 
ax.tick_params(which="minor", length=0.0) 
ax.set_title("Relative Error")
ax.set_xlabel("$x$")
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
