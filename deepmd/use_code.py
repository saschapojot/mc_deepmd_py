from decimal import Decimal, getcontext
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
from deepmd.infer import DeepPot


def format_using_decimal(value, precision=4):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)
N=4
T=100
P=0.1

TStr=format_using_decimal(T)
PStr=format_using_decimal(P)

dataDir=f"./mcDataAll/Nx{N}_Ny{N}_Nz{N}/T{TStr}/P{PStr}/"

in_box_dir=dataDir+f"/U_dist_dataFiles/box/"
in_coord_dir=dataDir+f"/U_dist_dataFiles/coord/"
def load_pickle_data(flushEnd):

    in_pkl_box_file=in_box_dir+f"/flushEnd{flushEnd}.box.pkl"
    with open(in_pkl_box_file,"rb") as fptr:
        in_box_arr=np.array(pickle.load(fptr))

    in_pkl_coord_file=in_coord_dir+f"/flushEnd{flushEnd}.coord.pkl"

    with open(in_pkl_coord_file,"rb") as fptr:
        in_coord_arr=np.array(pickle.load(fptr))

    return in_box_arr, in_coord_arr


flushEnd=2
sweep_to_write=7
sweep_multiple=5
in_box_arr, in_coord_arr=load_pickle_data(flushEnd)


which_frame=0
box_flat_size=3*3
coord_flat_size=N**3*3*3

box_1_frame_flat=in_box_arr[which_frame*box_flat_size:(which_frame+1)*box_flat_size]

coord_1_frame_flat=in_coord_arr[which_frame*coord_flat_size:(which_frame+1)*coord_flat_size]

cell=box_1_frame_flat.reshape((-1,3))

#the numbers in coordinates are:
#O_x, O_y, O_z, H1_x, H1_y, H1_z, H2_x, H2_y, H2_z
coordinates =coord_1_frame_flat.reshape((-1,3))

########################################
#visualize
# num_molecules = len(coordinates) // 3
# water_molecules = coordinates.reshape(num_molecules, 3, 3)
#
# # Create figure
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot atoms and bonds
# for molecule in water_molecules:
#     # Extract atoms
#     oxygen = molecule[0]
#     hydrogen1 = molecule[1]
#     hydrogen2 = molecule[2]
#
#     # Plot oxygen atoms (red)
#     ax.scatter(oxygen[0], oxygen[1], oxygen[2], color='red', s=100)
#
#     # Plot hydrogen atoms (white with black edge)
#     ax.scatter(hydrogen1[0], hydrogen1[1], hydrogen1[2], color='white', s=50, edgecolor='black')
#     ax.scatter(hydrogen2[0], hydrogen2[1], hydrogen2[2], color='white', s=50, edgecolor='black')
#
#     # Draw O-H bonds
#     ax.plot([oxygen[0], hydrogen1[0]], [oxygen[1], hydrogen1[1]], [oxygen[2], hydrogen1[2]], color='black')
#     ax.plot([oxygen[0], hydrogen2[0]], [oxygen[1], hydrogen2[1]], [oxygen[2], hydrogen2[2]], color='black')
#
# # Draw simulation box
# # Extract box dimensions
# box_min = np.array([0, 0, 0])
# box_max = np.array([cell[0, 0], cell[1, 1], cell[2, 2]])
#
# # Create box corners
# corners = np.array([
#     [box_min[0], box_min[1], box_min[2]],
#     [box_max[0], box_min[1], box_min[2]],
#     [box_max[0], box_max[1], box_min[2]],
#     [box_min[0], box_max[1], box_min[2]],
#     [box_min[0], box_min[1], box_max[2]],
#     [box_max[0], box_min[1], box_max[2]],
#     [box_max[0], box_max[1], box_max[2]],
#     [box_min[0], box_max[1], box_max[2]]
# ])
#
# # Define edges
# edges = [
#     (0, 1), (1, 2), (2, 3), (3, 0),
#     (4, 5), (5, 6), (6, 7), (7, 4),
#     (0, 4), (1, 5), (2, 6), (3, 7)
# ]
#
# # Plot box
# for edge in edges:
#     ax.plot([corners[edge[0]][0], corners[edge[1]][0]],
#             [corners[edge[0]][1], corners[edge[1]][1]],
#             [corners[edge[0]][2], corners[edge[1]][2]], color='blue', linestyle=':')
#
# # Set axis labels
# ax.set_xlabel('X (Å)')
# ax.set_ylabel('Y (Å)')
# ax.set_zlabel('Z (Å)')
# ax.set_title(f'Water Molecules in Simulation Box (N={N})')
#
# # Make the plot a bit less crowded for better visualization
# ax.set_xlim(box_min[0], box_max[0])
# ax.set_ylim(box_min[1], box_max[1])
# ax.set_zlim(box_min[2], box_max[2])
#
# # Equal aspect ratio
# ax.set_box_aspect([1, 1, 1])
#
# plt.tight_layout()
# # plt.show()
# plt.savefig("m.png")
########################################

# For water, we need to create atom_types array
# Each water molecule has 1 oxygen (type 0) and 2 hydrogens (type 1)
num_water_molecules = N**3
atom_types = np.zeros(num_water_molecules * 3, dtype=np.int32)

# Set atom types: O=0, H=1 for each water molecule
for i in range(num_water_molecules):
    atom_types[i*3] = 0      # Oxygen
    atom_types[i*3+1] = 1    # Hydrogen 1
    atom_types[i*3+2] = 1    # Hydrogen 2


model_file=f"./se_e2_a/compressed_model_water.pth"
# Load the DeepMD model
dp = DeepPot(model_file)  # Replace with your model path

energy, forces, virial, atomic_energies = dp.eval(coordinates, cell, atom_types, atomic=True)
print(f"Atomic energies shape: {atomic_energies.shape}")
