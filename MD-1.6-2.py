# Name: Pieter Leonard
# Student-id: 5121663
# Date: 17/06/2024

import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.constants import Avogadro, Boltzmann
import time

#%% Write Trajectory files in LAMMPS format

TrajectoryFile = ('VMD files\TrajectoryFile-Q4.lammpstrj')

def write_frame(coords, L, vels, forces, trajectory_name, step):
    '''
    function to write trajectory file in LAMMPS format

    In VMD you can visualize the motion of particles using this trajectory file.

    :param coords: coordinates
    :param vels: velocities
    :param forces: forces
    :param trajectory_name: trajectory filename

    :return:
    '''

    nPart = len(coords[:, 0])
    nDim = len(coords[0, :])
    with open(trajectory_name, 'a') as file:
        file.write('ITEM: TIMESTEP\n')
        file.write('%i\n' % step)
        file.write('ITEM: NUMBER OF ATOMS\n')
        file.write('%i\n' % nPart)
        file.write('ITEM: BOX BOUNDS pp pp pp\n')
        for dim in range(nDim):
            file.write('%.6f %.6f\n' % (-0.5 * Lbox, 0.5 * Lbox))
        for dim in range(3 - nDim):
            file.write('%.6f %.6f\n' % (0, 0))
        file.write('ITEM: ATOMS id type xu yu zu vx vy vz fx fy fz\n')

        temp = np.zeros((nPart, 9))
        for dim in range(nDim):
            temp[:, dim] = coords[:, dim]
            temp[:, dim + 3] = vels[:, dim]
            temp[:, dim + 6] = forces[:, dim]

        for part in range(nPart):
            file.write('%i %i %.4f %.4f %.4f %.6f %.6f %.6f %.4f %.4f %.4f\n' % (part + 1, 1, *temp[part, :]))

#%% Read a LAMMPS file

def read_lammps_data(data_file, verbose=False):
    """Reads a LAMMPS data file
        Atoms
        Velocities
    Returns:
        lmp_data (dict):
            'xyz': xyz (numpy.ndarray)
            'vel': vxyz (numpy.ndarray)
        box (numpy.ndarray): box dimensions
    """
    print("Reading '" + data_file + "'")
    with open(data_file, 'r') as f:
        data_lines = f.readlines()

    # improve robustness of xlo regex
    directives = re.compile(r"""
        ((?P<n_atoms>\s*\d+\s+atoms)
        |
        (?P<box>.+xlo)
        |
        (?P<Atoms>\s*Atoms)
        |
        (?P<Velocities>\s*Velocities))
        """, re.VERBOSE)

    i = 0
    while i < len(data_lines):
        match = directives.match(data_lines[i])
        if match:
            if verbose:
                print(match.groups())

            elif match.group('n_atoms'):
                fields = data_lines.pop(i).split()
                n_atoms = int(fields[0])
                xyz = np.empty(shape=(n_atoms, 3))
                vxyz = np.empty(shape=(n_atoms, 3))

            elif match.group('box'):
                dims = np.zeros(shape=(3, 2))
                for j in range(3):
                    fields = [float(x) for x in data_lines.pop(i).split()[:2]]
                    dims[j, 0] = fields[0]
                    dims[j, 1] = fields[1]
                L = dims[:, 1] - dims[:, 0]

            elif match.group('Atoms'):
                if verbose:
                    print('Parsing Atoms...')
                data_lines.pop(i)
                data_lines.pop(i)

                while i < len(data_lines) and data_lines[i].strip():
                    fields = data_lines.pop(i).split()
                    a_id = int(fields[0])
                    xyz[a_id - 1] = np.array([float(fields[2]),
                                              float(fields[3]),
                                              float(fields[4])])

            elif match.group('Velocities'):
                if verbose:
                    print('Parsing Velocities...')
                data_lines.pop(i)
                data_lines.pop(i)

                while i < len(data_lines) and data_lines[i].strip():
                    fields = data_lines.pop(i).split()
                    va_id = int(fields[0])
                    vxyz[va_id - 1] = np.array([float(fields[1]),
                                                float(fields[2]),
                                                float(fields[3])])

            else:
                i += 1
        else:
            i += 1

    return xyz, vxyz, L

#%% rarial pair distribution function


def rdf(xyz, LxLyLz, n_bins=100, r_range=(0.01, 10.0)):
    '''
    rarial pair distribution function

    :param xyz: coordinates in xyz format per frame
    :param LxLyLz: box length in vector format
    :param n_bins: number of bins
    :param r_range: range on which to compute rdf
    :return:
    '''

    g_r, edges = np.histogram([0], bins=n_bins, range=r_range)
    g_r[0] = 0
    g_r = g_r.astype(np.float64)
    rho = 0

    for i, xyz_i in enumerate(xyz):
        xyz_j = np.vstack([xyz[:i], xyz[i + 1:]])
        d = np.abs(xyz_i - xyz_j)
        d = np.where(d > 0.5 * LxLyLz, LxLyLz - d, d)
        d = np.sqrt(np.sum(d ** 2, axis=-1))
        temp_g_r, _ = np.histogram(d, bins=n_bins, range=r_range)
        g_r += temp_g_r

    rho += (i + 1) / np.prod(LxLyLz)
    r = 0.5 * (edges[1:] + edges[:-1])
    V = 4./3. * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
    norm = rho * i
    g_r /= norm * V

    return r, g_r

#%%
def read_state_variables(file_name):
    # making an empty array to store the coordinates
    variables = []

    with open(file_name, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) == 6:
                step, temp, press, KE, PE = map(float, parts[0:5])
                variables.append((step, temp, press, KE, PE))

    return np.array(variables)

#%% Variables
# import all the variables

mass = 16.04                #g/mol          | molar mass of methane in grams/mol
Lbox = 30.0                 #Å              | side length of cubic box in Angstrom meters
k_b = Boltzmann *1e-3       #kJ/K            | boltzmann constant
N_av = Avogadro             #1/mol          | avogadro's number
R = N_av*k_b                #kJ/(K*mol)      | gasconstant
Rcut = 10.0                 #Å              | cut-off distance in Angstrom meters
sigma = 3.73                #Å              | Lennard-Jones parameter sigma in Angstrom meters
epsilon = 148*R             #kJ/mol          | Lennard-Jones parameter epsilon
rho_dens = 358.4            #kg/m^3         | density in kilograms per cubic meters 
Nfreq = 100                 #               | sample frequency
steps = 1000                #fs             | length of simulation run
dt = 1                      #fs             | timestep
nDim = 3
Tstart = 150
zeta_start = 0.0

# some unit conversions
# 1 cal = 4.184 J
# 1 J = 1 Kg m^2 / s^2
# 1 Kg = 1e3 g
# 1m = 1e10 Angstrom
# 1s = 1e15 fs
# 1 Kg m^2 / s^2 = 1e3*(1e10/1e15)^2 = 1e-7  g Angstrom^2 / fs^2
# Kcal/mol = 4.184 * 1e-4 g/mol*(Angstrom/fs)^2
# g/mol*(Angstrom/fs)^2 = 1/(4.184 * 1e-4) Kcal/mol

#%% Initial configuration of the grid

def initGrid(rho_dens, Lbox):
    
    rho = (rho_dens * N_av) / (1e27) 
    V = Lbox**nDim #angstrom^3
    nPart = int(np.rint((rho * V / mass)))
    
    
    coords = np.zeros((nPart, 3))
    index = np.zeros(nDim)
    
    nPart_side = int(np.ceil(np.cbrt(nPart)))
    spacing = Lbox / nPart_side  
    
    # assign particle positions
    for part in range(nPart):
        coords[part, :] = index * spacing

        # advance particle position
        index[0] += 1

        # if x had reached it's limit -> to y-axis
        if index[0] == nPart_side:
            index[0] = 0
            index[1] += 1
            # if y had reached it's limit -> to z-axis
            if index[1] == nPart_side:
                index[1] = 0
                index[2] += 1
    
    return coords - Lbox / 2, Lbox #Angstrom
 
coords, L= initGrid(rho_dens, Lbox)
#coords_halfrho , Lhalf = initGrid(rho_dens/2, Lbox)
#coords_doublerho , Ldouble = initGrid(rho_dens*2, Lbox)

#print("number of Particles with normal density:", coords.shape[0])
#print("number of Particles with half density:", coords_halfrho.shape[0])
#print("number of Particles with double density:", coords_doublerho.shape[0])

#%% Export coords to VMD-file

def write_pdb(coordinates, filename="output.pdb"):
    with open(filename, "w") as file:
        atom_index = 1
        for i, coord in enumerate(coordinates):
            file.write(f"ATOM  {atom_index:5d}  C   MET A   1    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00           C  \n")
            atom_index += 1
        file.write("END\n")
            
#write_pdb(coords, "normal_density.pdb")
#write_pdb(coords_halfrho, "half_density.pdb")
#write_pdb(coords_doublerho, "double_density.pdb")
            
#%% Initial configuration of the grid's velocities

def initVel(coords, T):
    nPart, nDim = coords.shape[0], coords.shape[1]
    
    initVel = np.random.randn(nPart, nDim)       # Ansgtrom/fs  
    mean_velocity = np.mean(initVel, axis=0)     # Ansgtrom/fs
    initVel -= mean_velocity                     # Ansgtrom/fs
    
    initTemp = (np.sum(initVel**2) * mass * 1e4) / (R * nDim * nPart) # K
    
    scale_factor = np.sqrt(T / initTemp)
    Velocities = initVel * scale_factor # Angstrom/fs
    
    # Check the scaled temperature for debugging
    #scaled_temp = (np.sum(Velocities**2) * mass * 1e4) / (R * nDim * nPart)  # K
    #print(f"Temperature: {T:.2f} K")
    #print(f"Scaled Temperature: {scaled_temp:.2f} K")
    
    return Velocities # Ansgtrom/fs

#Velocities = initVel(coords, Tstart)

#%% Compute the LJ-Forces

# Define a function for the total energy:
def LJ_forces(coords, Lbox, Rcut):
    nPart, nDim = coords.shape[0], coords.shape[1]
    Forces = np.zeros((nPart, nDim))
    
    Rcut2 = Rcut * Rcut
    r = np.broadcast_to(coords, (nPart, nPart, nDim))
    d = (r.transpose(1, 0, 2) - r + Lbox/2) % Lbox - Lbox/2
    r2 = np.sum(d * d, axis=-1)
    
    cutoff = (r2 > Rcut2)
    r2[cutoff] = np.inf
    r2[r2 == 0] = np.inf
    
    sr2  = sigma*sigma / r2
    sr6 = sr2 * sr2 *sr2
    sr12 = sr6 * sr6
    
    force_mag = 24 * epsilon * (2*sr12 - sr6) / r2
    force_mag2 = force_mag.reshape(nPart, nPart, 1)
    Forces = np.sum(d * force_mag2, axis=1)

    return Forces # kJ / (mol*Angstrom)

#Forces = LJ_forces(coords, Lbox, Rcut)

#%% Function to compute kinetic energy

def kineticEnergy(Velocities):
    
    v2 = np.sum(Velocities*Velocities, axis=1) # (Å/fs)^2
    
    KE = 0.5 * mass * np.sum(v2, axis=0) * 1e4
    
    #print(f"Potential Energy: {KE:.2f} kJ/mol")
    
    return KE

#KE = kineticEnergy(Velocities)

#%% Function to compute potential energy

def potentialEnergy(coords, Lbox, Rcut):
    PE = 0
    nPart, nDim = coords.shape[0], coords.shape[1]
    
    Rcut2 = Rcut * Rcut
    r = np.broadcast_to(coords, (nPart, nPart, nDim))
    d = (r.transpose(1, 0, 2) - r + Lbox/2) % Lbox - Lbox/2
    r2 = np.sum(d * d, axis=-1)
    
    cutoff = (r2 > Rcut2)
    r2[cutoff] = np.inf
    r2[r2 == 0] = np.inf
    
    upper_triangle = np.triu_indices(nPart, k=1)
    r2_triangle = r2[upper_triangle]

    sr2 = sigma*sigma / r2_triangle
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    
    U_LJ = np.sum(sr12 - sr6)
    
    # Compute the potential energies
    PE = 4 * epsilon * U_LJ
    
    #print(f"Potential Energy: {PE:.2f} kJ/mol")

    return PE  # kJ/mol

#PE = potentialEnergy(coords, Lbox, Rcut)

#%% Function to compute temperature

def Temperature(Velocities):
    nPart, nDim = coords.shape[0], coords.shape[1]
    
    KE = kineticEnergy(Velocities)
    
    T = ((KE / nPart) * 2) / (nDim * R)
    
    #print(f"Temperature: {T:.2f} K")
    
    return T #K

#T = temperature(Velocities)

#%% Function to compute pressure

def Pressure(coords, Velocities, Lbox, Rcut):
    nPart, nDim = coords.shape[0], coords.shape[1]
    V = Lbox**3
    T = Temperature(Velocities)
    pressure = 0
    
    Rcut2 = Rcut * Rcut
    r = np.broadcast_to(coords, (nPart, nPart, nDim))
    d = (r.transpose(1, 0, 2) - r + Lbox/2) % Lbox - Lbox/2
    r2 = np.sum(d * d, axis=-1)
    
    cutoff = (r2 > Rcut2)
    r2[cutoff] = np.inf
    r2[r2 == 0] = np.inf
    
    upper_triangle = np.triu_indices(nPart, k=1)
    r2_triangle = r2[upper_triangle]

    sr2 = sigma*sigma / r2_triangle
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
        
    P_LJ = np.sum(2*sr12 - sr6)
    P_virial = (24 * epsilon * P_LJ) / (nDim * V)

    P_ID = (nPart * R * T) / (V)
        
    pressure = P_ID + P_virial
    return pressure # Bar

#pres = pressure(coords, Velocities, Lbox, Rcut)

#%% Velocity Verlet integrator

def VelocityVerlet(positions, velocities, dt, mass, Lbox, Rcut, epsilon, sigma):
    
    forces = LJ_forces(positions, Lbox, Rcut)
    
    accelerations = (forces * 1e-4) / mass
    # Update positions
    new_positions = (positions + velocities * dt + 0.5 * accelerations * dt ** 2 + Lbox / 2) % Lbox - Lbox / 2
    
    # Update velocities (half step)
    half_step_velocities = velocities + 0.5 * accelerations * dt
    
    # Compute new forces
    new_forces = LJ_forces(new_positions, Lbox, Rcut)
    
    new_accelerations = (new_forces * 1e-4) / mass
    
    # Update velocities (second half step)
    new_velocities = half_step_velocities + 0.5 * new_accelerations * dt
    
    return new_positions, new_velocities
    
#%% Velocity Verlet integrator with Thermostat

def VelocityVerletThermostat(positions, velocities, zeta, T_end, dt, mass, Lbox, Rcut, epsilon, sigma):
    nPart, nDim = coords.shape[0], coords.shape[1]
    
    tau = 100 # Thermostat relaxation time in fs #TODO

    Q_guess = nPart * R * T_end * tau**2     #TODO: change this value to make the thermostat better, removed nDim*
    Q = Q_guess       # damping parameter
    
    forces = LJ_forces(positions, Lbox, Rcut)
    KE = kineticEnergy(velocities)
    accelerations = (forces * 1e-4) / mass

    # Update positions
    new_positions = (positions + velocities * dt + 0.5 * (dt**2) * (accelerations - zeta * velocities) + Lbox / 2) % (Lbox - Lbox / 2)
    
    # Update zeta and velocities (half step)
    half_zeta = zeta + dt/(2*Q) * (KE - (nDim * nPart + 1) / 2 * R * T_end)
    
    half_velocities = velocities + 0.5 * dt * (accelerations - half_zeta * velocities)
    
    # Compute new forces
    new_forces = LJ_forces(new_positions, Lbox, Rcut)
    new_accelerations = (new_forces * 1e-4) / mass
    
    new_KE = kineticEnergy(half_velocities)
    
    new_zeta = half_zeta + dt/(2*Q) * (new_KE - nDim * (nPart+1)/2 * R * T_end)
    
    # Update velocities (second half step)
    new_velocities = (half_velocities + 0.5 * dt * new_accelerations) / (1 + 0.5 * dt * new_zeta)
    
    return new_positions, new_velocities, new_zeta

#%% Execute velocity-verlet function

#tic = time.time()
#new_coords, new_vels, new_forces = velocityVerlet(coords, Lbox, dt, steps, T_lammps, TrajectoryFile)
#toc = time.time()
#print("\n Whole calculation:"+str(toc-tic)+ "s")
#print("Whole calculation:"+str((toc-tic)/60)+ "min")

#%% MD solver with own results without thermostat

def MD_solver(steps, dt, mass, Lbox, Rcut, epsilon, sigma, R, trajectory_name, Tstart, Nfreq, rho):
    # Make empty arrays for state variables
    KE_list = []
    PE_list = []
    T_list = []
    P_list = []
    time_list = []
    
    # Generate the initial positions and velocities and write these in the LAMMPS file
    positions, nPart = initGrid(rho_dens, Lbox)
    velocities = initVel(coords, Tstart)
    
    # step = 0
    LxLyLz = np.array([Lbox, Lbox, Lbox])
    forces = LJ_forces(positions, Lbox, Rcut)
    write_frame(positions, LxLyLz, velocities, forces, trajectory_name, 0)

    # Calculate the state values for the initial position
    initKE = kineticEnergy(velocities) # kJ/mol
    initPE = potentialEnergy(positions, Lbox, Rcut)  # kJ/mol
    initT = Temperature(velocities)  # K
    initP = Pressure(positions, velocities, Lbox, Rcut) * 10 ** 28 / N_av  # bar

    KE_list.append(initKE)
    PE_list.append(initPE)
    T_list.append(initT)
    P_list.append(initP)
    time_list.append(0)
    
    tic1 = time.time()
    # for loop to do the desired amount of steps
    for step in range(1, steps + 1):
        # make a step using the velocityVerlet function
        
        positions, velocities = VelocityVerlet(positions, velocities, dt, mass, Lbox, Rcut, epsilon, sigma)
        
        
        # Compute state variables at sample frequency
        if step % Nfreq == 0:
            toc1 = time.time()
            print("\n One step time:"+ str(toc1-tic1) + "s")
            #print("One step time:"+str((toc1-tic1)/60)+ "min")
            forces = LJ_forces(positions, Lbox, Rcut)
            write_frame(positions, LxLyLz, velocities, forces, trajectory_name, step)

            KE = kineticEnergy(velocities)
            PE = potentialEnergy(positions, Lbox, Rcut)
            T = Temperature(velocities)
            P = Pressure(positions, velocities, Lbox, Rcut) * 10 ** 28 / N_av  #bar

            KE_list.append(KE)
            PE_list.append(PE)
            T_list.append(T)
            P_list.append(P)
            time_list.append(step * dt)
            print(f" MD-solver: Time step {step} of {steps} completed, [{step*100/(steps):.2f} %] ")
            
    return KE_list, PE_list, T_list, P_list, time_list, forces, velocities, positions
#%% Execute MD-solver for own results

tic = time.time()
KE_list, PE_list, T_list, P_list, time_list, forces, velocities, positions = MD_solver(steps, dt, mass, Lbox, Rcut, epsilon, sigma, R, TrajectoryFile, Tstart, Nfreq, rho_dens)
toc = time.time()
print("\n Whole calculation:"+str(toc-tic)+ "s")
print("Whole calculation:"+str((toc-tic)/60)+ "min")


#%% MD solver with Lammps results without thermostat

def MD_solver_LAMMPS(steps, dt, mass, Lbox, Rcut, epsilon, sigma, R, trajectory_name, Tstart, Nfreq, rho):
    # Make empty arrays for state variables
    KE_LAMMPS_list = []
    PE_LAMMPS_list = []
    T_LAMMPS_list = []
    P_LAMMPS_list = []
    time_LAMMPS_list = []
    
    # step = 0
    LxLyLz = np.array([Lbox, Lbox, Lbox])
    positions, velocities, L = read_lammps_data('VERIFY1\lammps.data')
    forces = LJ_forces(positions, Lbox, Rcut)
    write_frame(positions, LxLyLz, velocities, forces, trajectory_name, 0)

    # Calculate the state values for the initial position
    initKE_LAMMPS = kineticEnergy(velocities) # kJ/mol
    initPE_LAMMPS = potentialEnergy(positions, Lbox, Rcut)  # kJ/mol
    initT_LAMMPS = Temperature(velocities)  # K
    initP_LAMMPS = Pressure(positions, velocities, Lbox, Rcut) * 10 ** 28 / N_av  # bar

    KE_LAMMPS_list.append(initKE_LAMMPS)
    PE_LAMMPS_list.append(initPE_LAMMPS)
    T_LAMMPS_list.append(initT_LAMMPS)
    P_LAMMPS_list.append(initP_LAMMPS)
    time_LAMMPS_list.append(0)
    
    tic1 = time.time()
    # for loop to do the desired amount of steps
    for step in range(1, steps + 1):
        # make a step using the velocityVerlet function
        
        positions, velocities = VelocityVerlet(positions, velocities, dt, mass, Lbox, Rcut, epsilon, sigma)
        
        # Compute state variables at sample frequency
        if step % Nfreq == 0:
            toc1 = time.time()
            print("\n One step time:"+str(toc1-tic1)+ "s")
            print("One step time:"+str((toc1-tic1)/60)+ "min")
            forces = LJ_forces(positions, Lbox, Rcut)
            write_frame(positions, LxLyLz, velocities, forces, trajectory_name, step)

            KE_LAMMPS = kineticEnergy(velocities)
            PE_LAMMPS = potentialEnergy(positions, Lbox, Rcut)
            T_LAMMPS = Temperature(velocities)
            P_LAMMPS = Pressure(positions, velocities, Lbox, Rcut) * 10 ** 28 / N_av  #bar

            KE_LAMMPS_list.append(KE_LAMMPS)
            PE_LAMMPS_list.append(PE_LAMMPS)
            T_LAMMPS_list.append(T_LAMMPS)
            P_LAMMPS_list.append(P_LAMMPS)
            time_LAMMPS_list.append(step * dt)
            print(f" MD-solver: Time step {step} of {steps} completed, [{step*100/(steps):.2f} %] ")
            
    return KE_LAMMPS_list, PE_LAMMPS_list, T_LAMMPS_list, P_LAMMPS_list, time_LAMMPS_list, forces, velocities, positions

#%% Execute MD-solver for LAMMPS results without Thermostat

tic = time.time()
KE_LAMMPS_list, PE_LAMMPS_list, T_LAMMPS_list, P_LAMMPS_list, time_LAMMPS_list, forces, velocities, positions = MD_solver_LAMMPS(steps, dt, mass, Lbox, Rcut, epsilon, sigma, R, TrajectoryFile, Tstart, Nfreq, rho_dens)
toc = time.time()
print("\n Whole calculation:"+str(toc-tic)+ "s")
print("Whole calculation:"+str((toc-tic)/60)+ "min")

#%% Total energy calculation without Thermostat

KE_array = np.array(KE_list)
KE_LAMMPS_array = np.array(KE_LAMMPS_list)

PE_array = np.array(PE_list)
PE_LAMMPS_array = np.array(PE_LAMMPS_list)

TE_array = KE_array + PE_array
TE_LAMMPS_array = KE_LAMMPS_array + PE_LAMMPS_array

TE_list = TE_array.tolist()
TE_LAMMPS_list = TE_LAMMPS_array.tolist()

#%% plot all the individual comparisons per state variable without Thermostat

def plot_individual_comparisons(time, own_results, lammps_results, variable_name, unit):
    """
    Plot individual graphs comparing own results to LAMMPS results.

    Parameters:
    - time: List or array of time steps.
    - own_results: List or array of own results for the state variable.
    - lammps_results: List or array of LAMMPS results for the state variable.
    - variable_name: Name of the state variable to be used in the plot title and labels.
    """
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, own_results, marker='o', linestyle='-', markersize=3, label=f'Own {variable_name}')
    plt.plot(time, lammps_results, marker='s', linestyle='-', markersize=3, label=f'LAMMPS {variable_name}')
    plt.xlabel('Time (fs)')
    plt.ylabel(f'{variable_name} ({unit})')
    plt.title(f'{variable_name} vs Time')
    plt.grid(True)
    plt.legend()
    plt.show()

# Plot individual comparisons
plot_individual_comparisons(time_list, KE_list, KE_LAMMPS_list, 'Kinetic Energy', 'kJ/mol')
plot_individual_comparisons(time_list, PE_list, PE_LAMMPS_list, 'Potential Energy', 'kJ/mol')
plot_individual_comparisons(time_list, T_list, T_LAMMPS_list, 'Temperature', 'K')
plot_individual_comparisons(time_list, P_list, P_LAMMPS_list, 'Pressure', 'Bar')
plot_individual_comparisons(time_list, TE_list, TE_LAMMPS_list, 'Total Energy', 'kJ/mol')

#%% MD solver with own results with thermostat
#TODO: scroll here

def MD_solverThermostat(steps, dt, mass, Lbox, Rcut, epsilon, sigma, R, trajectory_name, T_end, Nfreq, rho, zeta):
    # Make empty arrays for state variables
    KE_list = []
    PE_list = []
    T_list = []
    P_list = []
    time_list = []
    
    # Generate the initial positions and velocities and write these in the LAMMPS file
    positions, nPart = initGrid(rho_dens, Lbox)
    velocities = initVel(coords, Tstart)
    
    # step = 0
    LxLyLz = np.array([Lbox, Lbox, Lbox])
    forces = LJ_forces(positions, Lbox, Rcut)
    write_frame(positions, LxLyLz, velocities, forces, trajectory_name, 0)

    # Calculate the state values for the initial position
    initKE = kineticEnergy(velocities) # kJ/mol
    initPE = potentialEnergy(positions, Lbox, Rcut)  # kJ/mol
    initT = Temperature(velocities)  # K
    initP = Pressure(positions, velocities, Lbox, Rcut) * 10 ** 28 / N_av  # bar

    KE_list.append(initKE)
    PE_list.append(initPE)
    T_list.append(initT)
    P_list.append(initP)
    time_list.append(0)
    
    tic1 = time.time()
    # for loop to do the desired amount of steps
    for step in range(1, steps + 1):
        # make a step using the velocityVerlet function
        
        positions, velocities, zeta = VelocityVerletThermostat(positions, velocities, zeta, T_end, dt, mass, Lbox, Rcut, epsilon, sigma) 
        
        
        # Compute state variables at sample frequency
        if step % Nfreq == 0:
            toc1 = time.time()
            print("\n One step time:"+ str(toc1-tic1) + "s")
            #print("One step time:"+str((toc1-tic1)/60)+ "min")
            forces = LJ_forces(positions, Lbox, Rcut)
            write_frame(positions, LxLyLz, velocities, forces, trajectory_name, step)

            KE = kineticEnergy(velocities)
            PE = potentialEnergy(positions, Lbox, Rcut)
            T = Temperature(velocities)
            P = Pressure(positions, velocities, Lbox, Rcut) * 10 ** 28 / N_av  #bar

            KE_list.append(KE)
            PE_list.append(PE)
            T_list.append(T)
            P_list.append(P)
            time_list.append(step * dt)
            print(f" MD-solver: Time step {step} of {steps} completed, [{step*100/(steps):.2f} %] ")
            
    return KE_list, PE_list, T_list, P_list, time_list, forces, velocities, positions
#%% Execute MD-solver for own results with Thermostat

tic = time.time()
KE_list, PE_list, T_list, P_list, time_list, forces, velocities, positions = MD_solverThermostat(steps, dt, mass, Lbox, Rcut, epsilon, sigma, R, TrajectoryFile, Tstart, Nfreq, rho_dens, zeta_start)
toc = time.time()
print("\n Whole calculation:"+str(toc-tic)+ "s")
print("Whole calculation:"+str((toc-tic)/60)+ "min")


#%% MD solver with Lammps results with Thermostat

def MD_solverThermostat_LAMMPS(steps, dt, mass, Lbox, Rcut, epsilon, sigma, R, trajectory_name, T_end, Nfreq, rho, zeta):
    # Make empty arrays for state variables
    KE_LAMMPS_list = []
    PE_LAMMPS_list = []
    T_LAMMPS_list = []
    P_LAMMPS_list = []
    time_LAMMPS_list = []
    
    # step = 0
    LxLyLz = np.array([Lbox, Lbox, Lbox])
    positions, velocities, L = read_lammps_data('VERIFY1\lammps.data')
    forces = LJ_forces(positions, Lbox, Rcut)
    write_frame(positions, LxLyLz, velocities, forces, trajectory_name, 0)

    # Calculate the state values for the initial position
    initKE_LAMMPS = kineticEnergy(velocities) # kJ/mol
    initPE_LAMMPS = potentialEnergy(positions, Lbox, Rcut)  # kJ/mol
    initT_LAMMPS = Temperature(velocities)  # K
    initP_LAMMPS = Pressure(positions, velocities, Lbox, Rcut) * 10 ** 28 / N_av  # bar

    KE_LAMMPS_list.append(initKE_LAMMPS)
    PE_LAMMPS_list.append(initPE_LAMMPS)
    T_LAMMPS_list.append(initT_LAMMPS)
    P_LAMMPS_list.append(initP_LAMMPS)
    time_LAMMPS_list.append(0)
    
    tic1 = time.time()
    # for loop to do the desired amount of steps
    for step in range(1, steps + 1):
        # make a step using the velocityVerlet function
        
        positions, velocities, zeta = VelocityVerletThermostat(positions, velocities, zeta, T_end, dt, mass, Lbox, Rcut, epsilon, sigma)
        # Compute state variables at sample frequency
        if step % Nfreq == 0:
            toc1 = time.time()
            print("\n One step time:"+str(toc1-tic1)+ "s")
            print("One step time:"+str((toc1-tic1)/60)+ "min")
            forces = LJ_forces(positions, Lbox, Rcut)
            write_frame(positions, LxLyLz, velocities, forces, trajectory_name, step)

            KE_LAMMPS = kineticEnergy(velocities)
            PE_LAMMPS = potentialEnergy(positions, Lbox, Rcut)
            T_LAMMPS = Temperature(velocities)
            P_LAMMPS = Pressure(positions, velocities, Lbox, Rcut) * 10 ** 28 / N_av  #bar

            KE_LAMMPS_list.append(KE_LAMMPS)
            PE_LAMMPS_list.append(PE_LAMMPS)
            T_LAMMPS_list.append(T_LAMMPS)
            P_LAMMPS_list.append(P_LAMMPS)
            time_LAMMPS_list.append(step * dt)
            print(f" MD-solver: Time step {step} of {steps} completed, [{step*100/(steps):.2f} %] ")
            
    return KE_LAMMPS_list, PE_LAMMPS_list, T_LAMMPS_list, P_LAMMPS_list, time_LAMMPS_list, forces, velocities, positions

#%% Execute MD-solver for LAMMPS results with Thermostat

tic = time.time()
KE_LAMMPS_list, PE_LAMMPS_list, T_LAMMPS_list, P_LAMMPS_list, time_LAMMPS_list, forces, velocities, positions = MD_solverThermostat_LAMMPS(steps, dt, mass, Lbox, Rcut, epsilon, sigma, R, TrajectoryFile, Tstart, Nfreq, rho_dens, zeta_start)
toc = time.time()
print("\n Whole calculation:"+str(toc-tic)+ "s")
print("Whole calculation:"+str((toc-tic)/60)+ "min")

#%% Total energy calculation with Thermostat

KE_array = np.array(KE_list)
KE_LAMMPS_array = np.array(KE_LAMMPS_list)

PE_array = np.array(PE_list)
PE_LAMMPS_array = np.array(PE_LAMMPS_list)

TE_array = KE_array + PE_array
TE_LAMMPS_array = KE_LAMMPS_array + PE_LAMMPS_array

TE_list = TE_array.tolist()
TE_LAMMPS_list = TE_LAMMPS_array.tolist()

#%% plot all the individual comparisons per state variable with Thermostat

def plot_individual_comparisons(time, own_results, lammps_results, variable_name, unit):
    """
    Plot individual graphs comparing own results to LAMMPS results.

    Parameters:
    - time: List or array of time steps.
    - own_results: List or array of own results for the state variable.
    - lammps_results: List or array of LAMMPS results for the state variable.
    - variable_name: Name of the state variable to be used in the plot title and labels.
    """
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, own_results, marker='o', linestyle='-', markersize=3, label=f'Own {variable_name}')
    plt.plot(time, lammps_results, marker='s', linestyle='-', markersize=3, label=f'LAMMPS {variable_name}')
    plt.xlabel('Time (fs)')
    plt.ylabel(f'{variable_name} ({unit})')
    plt.title(f'{variable_name} vs Time')
    plt.grid(True)
    plt.legend()
    plt.show()

# Plot individual comparisons
plot_individual_comparisons(time_list, KE_list, KE_LAMMPS_list, 'Kinetic Energy', 'kJ/mol')
plot_individual_comparisons(time_list, PE_list, PE_LAMMPS_list, 'Potential Energy', 'kJ/mol')
plot_individual_comparisons(time_list, T_list, T_LAMMPS_list, 'Temperature', 'K')
plot_individual_comparisons(time_list, P_list, P_LAMMPS_list, 'Pressure', 'Bar')
plot_individual_comparisons(time_list, TE_list, TE_LAMMPS_list, 'Total Energy', 'kJ/mol')

#%% plot all the individual own data per state variable with Thermostat

def plot_individual_comparisons(time, own_results, variable_name, unit):
    """
    Plot individual graphs comparing own results to LAMMPS results.

    Parameters:
    - time: List or array of time steps.
    - own_results: List or array of own results for the state variable.
    - lammps_results: List or array of LAMMPS results for the state variable.
    - variable_name: Name of the state variable to be used in the plot title and labels.
    """
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, own_results, marker='o', linestyle='-', markersize=3, label=f'Own {variable_name}')
    plt.xlabel('Time (fs)')
    plt.ylabel(f'{variable_name} ({unit})')
    plt.title(f'{variable_name} vs Time')
    plt.grid(True)
    plt.legend()
    plt.show()

# Plot individual comparisons
plot_individual_comparisons(time_list, KE_list, 'Kinetic Energy', 'kJ/mol')
plot_individual_comparisons(time_list, PE_list, 'Potential Energy', 'kJ/mol')
plot_individual_comparisons(time_list, T_list, 'Temperature', 'K')
plot_individual_comparisons(time_list, P_list, 'Pressure', 'Bar')
plot_individual_comparisons(time_list, TE_list, 'Total Energy', 'kJ/mol')