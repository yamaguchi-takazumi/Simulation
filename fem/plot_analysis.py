import argparse
import matplotlib.pyplot as plt
import numpy as np
import os


#################
# Setting Hyper Parameter
#
parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, 
                    help="path of directory.")
parser.add_argument("flname", type=str, 
                    help="file name of input data and output figure.")
parser.add_argument("--show_figure", action="store_true",
                    help="show compulational result on X-window.")

args = parser.parse_args()
data_path   = args.data_path
flname      = args.flname
show_figure = args.show_figure
flname_txt  = os.path.join(data_path, flname+".txt")

assert os.path.isdir(data_path), data_path + " does not exist!"
assert os.path.isfile(flname_txt), flname_txt + " does not exist!"


#################
# Load Computational Result
#
data = np.loadtxt(flname_txt).T
n = data[0]
rerror = data[1:]


#################
# Fiiting Computational Result
#
"""a_D, b_D = np.polyfit(np.log10(n), np.log10(rerror[0]), 1)
a_N, b_N = np.polyfit(np.log10(n), np.log10(rerror[1]), 1)
rerror_fit = 10**np.array([b_D, b_N]) * n**np.array([a_D, a_N])
print(a_D, b_D)
print(a_N, b_N)
"""
#################
# Visualize Computational Result
#
fig, ax = plt.subplots()
ax.plot(n, rerror[0], color="blue", marker="v", markersize=12, label="Dirichlet")
ax.plot(n, rerror[1], color="red" , marker="^", markersize=12, label="Neumann")
ax.tick_params(direction="in") 
ax.tick_params(which="minor", length=0.0) 
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of Nodes, $N$")
ax.set_ylabel("Relative Error, $\epsilon_R$")
plt.legend()
plt.savefig(os.path.join(data_path, flname+".png"))
if(show_figure):
    plt.show()
