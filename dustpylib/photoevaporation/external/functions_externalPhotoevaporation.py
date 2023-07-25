import numpy as np
from dustpy import constants as c
from scipy.interpolate import interp1d, LinearNDInterpolator

import sys


#####################################
#
# FRIED GRID ROUTINES
#
#####################################


def get_M400(Sigma_out, r_out):
    '''
    Transformed variable for the FRIED Grid.
    Receives r_out [au], and sigma_out  [g/cm^2].
    Returns a representative mass  [jupiter masses]
    '''
    return 2 * np.pi * Sigma_out * (r_out * c.au)**2 * (r_out/400)**(-1) / c.M_jup


def Set_FRIED_Interpolator(r_Table, Sigma_Table, MassLoss_Table):
    '''
    Returns the interpolator function, constructed from the FRIED grid data
    The interpolator takes the (M400 [Jupiter mass], r_out[au]) variables
    The interpolator returns the external photoevaporation mass loss rate [log10 (M_sun/year)]

    The interpolation is performed on the loglog space.
    '''

    # Following Sellek et al.(2020) implementation, the M400 converted variable is used to set the interpolator
    M400_Table = get_M400(Sigma_Table, r_Table)


    # Interpolation in the [log(M400), log(r_out) -> log(MassLoss)] parameter space
    Interpolator = LinearNDInterpolator(list(zip(np.log10(M400_Table), np.log10(r_Table))), MassLoss_Table, fill_value = -10)
    # Inputs outside the boundaries are set to the minimum of 1.e-10 Msun/yr


    # Return a Lambda function that converts the linear M400,r inputs to the logspace to perform the interpolation.
    return lambda M400, r: Interpolator(np.log10(M400), np.log10(r))




#####################################
# FRIED GRID ROUTINES - CALLED ONLY ON SETUP
#####################################


def get_MassLoss_SellekGrid(r_grid, Sigma_grid, r_Table, Sigma_Table, MassLoss_Table):
    '''
    Obtain the MassLoss grid in radius and Sigma, following the interpolation of Sellek et al.(2020) Eq.5
    Note: This only work for a single stellar mass and UV Flux, already available in the FRIED grid
    ----------------------------------------
    r_grid, Sigma_grid:                      Radial[AU] and Surface density [g/cm^2] grid to obtain the Mass Loss interpolation
    r_Table, Sigma_Table, MassLoss_Table:    FRIED Grid data columns (masked to match a single stellar mass and UV Flux)

    returns
    MassLoss_SellekGrid [log Msun/year]:     Mass loss rates as shown Figure 1. in Sellek(2020)
    Sigma_min, Sigma_max [g/cm^2]:           Surface density limits of the FRIED grid for the given parameter space
    ----------------------------------------
    '''

    # Obtain the FRIED interpolator function that returns the Mass loss, given a (M400, r) input
    FRIED_Interpolator = Set_FRIED_Interpolator(r_Table, Sigma_Table, MassLoss_Table)


    # Find out the shape of the table in the r_Table parameter range
    shape_FRIED = (int(r_Table.size/np.unique(r_Table).size), np.unique(r_Table).size)


    # Obtain the values of r_Table, and the corresponding minimum and maximum value of Sigma in the Fried grid for each r_Table
    r_Table = r_Table.reshape(shape_FRIED)[0]                           # Dimension: unique(Table.r_Table)
    Sigma_max = np.max(Sigma_Table.reshape(shape_FRIED), axis= 0)     # Dimension: unique(Table.r_Table)
    Sigma_min = np.min(Sigma_Table.reshape(shape_FRIED), axis= 0)     # Dimension: unique(Table.r_Table)

    # Give a buffer factor, since the FRIED interpolator should not extrapolate outside the original
    buffer_max = 0.9 # buffer for the Sigma upper grid limit
    buffer_min = 1.1 # buffer for the Sigma lower grid limit


    # The interpolation of the grid limits is performed on the logarithmic space
    # See the FRIED grid (r_Table vs. Sigma) data distribution for reference
    f_Sigma_FRIED_max = lambda r_interp: 10**interp1d(np.log10(r_Table), np.log10(buffer_max * Sigma_max), kind='linear', fill_value = 'extrapolate')(np.log10(r_interp))
    f_Sigma_FRIED_min = lambda r_interp: 10**interp1d(np.log10(r_Table), np.log10(buffer_min * Sigma_min), kind='linear', fill_value = 'extrapolate')(np.log10(r_interp))



    # Calculate the density limits and the corresponding mass loss rates for the custom radial grid
    Sigma_max = f_Sigma_FRIED_max(r_grid)
    Sigma_min = f_Sigma_FRIED_min(r_grid)

    # Calculate the limits of M400 for the respective Sigma_max and Sigma_min
    M400_max = get_M400(Sigma_max, r_grid)
    M400_min = get_M400(Sigma_min, r_grid)

    MassLoss_max = FRIED_Interpolator(M400_max, r_grid)  # Upper limit of the mass loss rate from the fried grid
    MassLoss_min = FRIED_Interpolator(M400_min, r_grid)  # Lower limit of the mass loss rate from the fried grid


    # Mask the regions where the custom Sigma grid is outside the FRIED boundaries
    mask_max= Sigma_grid >= Sigma_max
    mask_min= Sigma_grid <= Sigma_min

    # Calculate the mass loss rate for each grid cell according to the FRIED grid
    # Note that the mass loss rate is in logarithmic-10 space
    M400_grid = get_M400(Sigma_grid, r_grid)

    MassLoss_SellekGrid = FRIED_Interpolator(M400_grid, r_grid) # Mass loss rate from the FRIED grid
    MassLoss_SellekGrid[mask_max] = MassLoss_max[mask_max]
    MassLoss_SellekGrid[mask_min] = MassLoss_min[mask_min] + np.log10(Sigma_grid / Sigma_min)[mask_min]
    MassLoss_SellekGrid[MassLoss_SellekGrid < -10] = -10

    return MassLoss_SellekGrid, Sigma_min, Sigma_max

def get_mask_StarUV(Mstar_value, UV_value, Mstar_Table, UV_Table):
    '''
    Construct a boolean mask that indicates rows of the FRIED Grid where Mstar_value and UV_value are present
    Mstar_value, UV_value must be available values of the FRIED Grid
    '''
    mask_Mstar = Mstar_Table == Mstar_value
    mask_UV = UV_Table == UV_value
    mask = mask_UV * mask_Mstar

    return mask



def get_weights_StarUV(Mstar_value, UV_value, Mstar_lr, UV_lr):

    '''
    Returns the interpolation weights for a given Mstar-UV value pair, within a rectangle Mstar-UV rectangle.
    The bi-linear interpolation is performed in the logpace of the Mstar-UV space
    '''

    logspace_weight = lambda value, left, right: (np.log10(value) - np.log10(left)) / (np.log10(right) - np.log10(left))

    f_Mstar = logspace_weight(Mstar_value, Mstar_lr[0], Mstar_lr[1])
    f_UV = logspace_weight(UV_value, UV_lr[0], UV_lr[1])

    f_weights = np.array([1. - f_Mstar, f_Mstar])[:, None] * np.array([1. - f_UV, f_UV])[None, :]
    return f_weights


def get_MassLoss_ResampleGrid(fried_filename = "./friedgrid.dat",
                              Mstar_target = 1., UV_target = 1000.,
                              grid_radii = None, grid_Sigma = None):
    '''
    Resample the FRIED grid into a new radial-Sigma grid for a target stellar mass and UV Flux
    --------------------------------------------
    fried_filename:                      FRIED grid from Haworth+(2018), download from: http://www.friedgrid.com/Downloads/
    Mstar_target [M_sun]:                Target stellar mass to reconstruct the FRIED grid (Must be between 0.05 - 1.9)
    UV_target [G0]:                      Target external UV flux to reconstruct the FRIED grid (Must be between 10 - 10^4)

    grid_radii[array (nr), AU]:                     Target radial grid array to reconstruct the FRIED grid
    grid_Sigma[array (nSig), g/cm^2]:               Target Sigma grid array to reconstruct the FRIED grid

    returns
    grid_MassLoss [array (nr, nSig), log(Msun/yr)]: Resampled Mass loss grid.
    grid_MassLoss_Interpolator:                     A function that returns the interpolated value based on the grid_MassLoss
                                                    The interpolator inputs are M400 [Jupiter mass] and r [AU]
    --------------------------------------------

    '''

    if grid_radii is None:
        grid_radii = np.linspace(1, 400, num = 50)
    if grid_Sigma is None:
        grid_Sigma = np.logspace(-5, 3, num = 100)


    FRIED_Grid = np.loadtxt(fried_filename, unpack=True, skiprows=1)

    Table_Mstar = FRIED_Grid[0]
    Table_UV = FRIED_Grid[1]
    Table_Sigma = FRIED_Grid[3]
    Table_rout = FRIED_Grid[4]
    Table_MassLoss = FRIED_Grid[5]


    #################################################################################
    # CREATE THE RADII-SIGMA MESHGRID AND THE SHAPE OF THE OUTPUT
    #################################################################################

    grid_radii, grid_Sigma = np.meshgrid(grid_radii, grid_Sigma, indexing = "ij")

    # This is the mass loss grid that we want to use for interpolation during simulation time
    grid_MassLoss = np.zeros_like(grid_radii)


    #################################################################################
    # FIND THE CLOSEST VALUES FOR THE STELLAR MASS AND UV FLUX IN THE FRIED GRID
    #################################################################################

    unique_Mstar = np.unique(Table_Mstar)
    unique_UV = np.unique(Table_UV)

    # Check that the UV and Stellar mass are within the FRIED grid
    if Mstar_target < unique_Mstar.min() or Mstar_target > unique_Mstar.max():
        print('Stellar mass out of the FRIED grid boundaries. [0.05 - 1.9] Msun')

    if UV_target < unique_UV.min() or UV_target > unique_UV.max():
        print('UV flux out of the FRIED grid boundaries. [10 - 10^4] G0')


    i_Mstar = np.searchsorted(unique_Mstar, Mstar_target)
    i_UV = np.searchsorted(unique_UV, UV_target)


    # Left/Right values around the available UV and Star Flux
    Mstar_lr = unique_Mstar[[i_Mstar - 1 , i_Mstar]]
    UV_lr = unique_UV[[i_UV - 1, i_UV]]

    #################################################################################
    # CONSTRUCT A MASS LOSS GRID FOR EACH OF THE CLOSEST STELLAR MASSES AND FLUXES USING SELLEK+2020 ALGORITHM
    #################################################################################

    grid_MassLoss_StarUV = []
    for Mstar_value in Mstar_lr:
        grid_MassLoss_StarUV.append([])
        for UV_value in UV_lr:
            # Mask the FRIED grid for available Mstar and UV values
            mask = get_mask_StarUV(Mstar_value, UV_value, Table_Mstar, Table_UV)

            # Save the MassLoss grid into a collection
            # The function also returns the surface density limits, but we do not need them here
            grid_MassLoss_dummy = get_MassLoss_SellekGrid(grid_radii, grid_Sigma, Table_rout[mask], Table_Sigma[mask], Table_MassLoss[mask])[0]
            grid_MassLoss_StarUV[-1].append(grid_MassLoss_dummy)
    grid_MassLoss_StarUV = np.array(grid_MassLoss_StarUV)


    #################################################################################
    # GET THE FINAL MASS LOSS GRID FOR THE TARGET UV FLUX AND STELLAR MASS
    #################################################################################

    interpolation_weights = get_weights_StarUV(Mstar_target, UV_target, Mstar_lr, UV_lr)
    grid_MassLoss = (interpolation_weights[:, :, None, None] * grid_MassLoss_StarUV).sum(axis=(0,1))

    # Return both the resampled grid for mass loss rates, and an interpolator function for it.
    # This is to avoid building the interpolator multiple times during the simulation run
    grid_MassLoss_Interpolator = Set_FRIED_Interpolator(grid_radii.flatten(), grid_Sigma.flatten(), grid_MassLoss.flatten())

    return grid_MassLoss, grid_MassLoss_Interpolator


##########################################################################
# UPDATER OF THE MASS LOSS FROM THE FRIED GRID AND TRUNCATION RADIUS
##########################################################################

# Called every timestep
def MassLoss_FRIED(sim):
    '''
    Calculates the instantaneous mass loss rate from the FRIED Grid (Haworth+, 2018) for each grid cell,

    '''

    # Interpolate the FRIED grid using the simulation radii and Sigma
    r_AU = sim.grid.r/c.au
    Sigma_g = sim.gas.Sigma
    M400 = get_M400(Sigma_g, r_AU)

    # Calls the interpolator hidden inside the FRIED class
    # This way it is not necessary to construct the interpolator every timestep, which is really time consuming
    MassLoss = sim.FRIED._Interpolator(M400, r_AU)

    # Convert to cgs
    MassLoss = np.power(10, MassLoss) * c.M_sun/c.year


    return MassLoss

def TruncationRadius(sim):
    '''
    Find the photoevaporative radii.
    See Sellek et al. (2020) Figure 2 for reference.
    '''


    # Near the FRIED limit, the truncation radius is extremely sensitive to small variations in the MassLoss profile.
    # To avoid these small variation giving large differences we round them
    # If the profile is completely constant, the truncation radius becomes the last grid cell

    MassLoss = sim.FRIED.MassLoss / (c.M_sun/c.year)
    # round to 10^-12 solar masses per year
    MassLoss = np.round(MassLoss, 12)
    ir_ext = np.size(MassLoss) - np.argmax(MassLoss[::-1]) - 1


    return sim.grid.r[ir_ext]


#####################################
# GAS LOSS RATE
#####################################

def SigmaDot_ExtPhoto(sim):
    '''
    Compute the Mass Loss Rate profile using Sellek+(2020) approach, using the mass loss rates from the FRIED grid of Haworth+(2018)
    '''

    # Mask the regions that should be subject to external photoevaporation
    mask = sim.grid.r >= sim.FRIED.rTrunc

    # Obtain Mass at each radial ring and total mass outside the photoevaporative radius
    mass_profile = sim.grid.A * sim.gas.Sigma
    mass_ext = np.sum(mass_profile[mask])

    # Total mass loss rate.
    mass_loss_ext = np.sum((sim.FRIED.MassLoss * mass_profile)[mask] / mass_ext)


    # Obtain the surface density profile using the mass of each ring as a weight factor
    # Remember to add the (-) sign to the surface density mass loss rate
    SigmaDot = np.zeros_like(sim.grid.r)
    SigmaDot[mask] = -sim.gas.Sigma[mask] *  mass_loss_ext / mass_ext


    # If the surface density is within a factor of 10 near the floor, stop futhrer mass loss
    FloorThreshold = 10
    SigmaDot[sim.gas.Sigma < FloorThreshold * sim.gas.SigmaFloor] = 0

    # return the surface density loss rate [g/cm²/s]
    return SigmaDot


#####################################
# DUST ENTRAINMENT AND LOSS RATE
#####################################

def PhotoEntrainment_Size(sim):
    '''
    Returns a radial array of the dust entrainment size.
    See Eq. 11 from Sellek+(2020)
    '''
    v_th = np.sqrt(8/np.pi) * sim.gas.cs                    # Thermal speed
    F = sim.gas.Hp / np.sqrt(sim.gas.Hp**2 + sim.grid.r**2) # Geometric Solid Angle
    rhos = sim.dust.rhos[0,0]                               # Dust material density

    # Calculate the total mass loss rate (remember to add the (-) sign)
    M_loss = -np.sum(sim.grid.A * sim.gas.S.ext)

    a_ent = v_th / (c.G * sim.star.M) * M_loss /(4 * np.pi * F * rhos)
    return a_ent



def SigmaDot_ExtPhoto_Dust(sim):

    mask_ent = np.where(sim.dust.a < sim.dust.a_ent[:, None], 1., 0.)   # Nr-Nm mask. Returns 1 when the grains are small enough to be entrained.

    d2g_ratio = sim.dust.Sigma / sim.gas.Sigma[:, None]                 # Dust-to-gas ratio profile for each dust species
    SigmaDot_Dust = mask_ent * d2g_ratio * sim.gas.S.ext[:, None]       # Dust loss rate [g/cm²/s]

    return SigmaDot_Dust
