import numpy as np
from dustpy import constants as c


from dustpylib.photoevaporation.external.functions_externalPhotoevaporation import get_MassLoss_ResampleGrid
from dustpylib.photoevaporation.external.functions_externalPhotoevaporation import MassLoss_FRIED, TruncationRadius
from dustpylib.photoevaporation.external.functions_externalPhotoevaporation import PhotoEntrainment_Size
from dustpylib.photoevaporation.external.functions_externalPhotoevaporation import SigmaDot_ExtPhoto, SigmaDot_ExtPhoto_Dust



################################################################################################
# Helper routine to add external photoevaporation to your Simulation object in one line.
################################################################################################

def setup_externalPhotoevaporation_FRIED(sim, fried_filename = "./friedgrid.dat", UV_Flux = 1000.,
                                            SigmaFloor = 1.e-40):
    '''
    Add external photoevaporation using the FRIED grid (Haworth et al., 2018) and the Sellek et al.(2020) implementation.
    This setup routine also performs the interpolation in the stellar mass and UV flux parameters.

    Call the setup function after the initialization and then run, as follows:

    sim.initialize()
    setup_extphoto_FRIED(sim)
    sim.run()
    ----------------------------------------------

    fried_filename:             FRIED grid from Haworth+(2018), download from: http://www.friedgrid.com/Downloads/
    UV_target [G0]:             External UV Flux

    SigmaFloor:                 Re-adjust the floor value of the gas surface density to improve the simulation performance

    ----------------------------------------------
    '''


    print("Setting up the backreaction module.")
    print("Please cite the work of Haworth et al.(2018), Sellek et al.(2020), and Garate et al.(in prep.)")
    ##################################
    # SET THE FRIED GRID
    ##################################

    # Obtain a resampled version of the FRIED grid for the simulation stellar mass and UV_flux.
    # Set the external photoevaporation Fields


    # Define a parameter space for the resampled radial and Sigma grids
    grid_radii = np.concatenate((np.array([1, 5]), np.linspace(10,400, num = 40)))
    grid_Sigma = np.concatenate((np.array([1e-8, 1e-6]), np.logspace(-5, 4, num = 100), np.array([5e4, 1e5])))

    # Obtain the mass loss grid.
    # Also obtain the interpolator(M400, r) function to include in the FRIED class as a hidden function
    grid_MassLoss, grid_MassLoss_Interpolator = get_MassLoss_ResampleGrid(fried_filename= fried_filename,
                                                                            Mstar_target= sim.star.M[0]/c.M_sun, UV_target= UV_Flux,
                                                                            grid_radii= grid_radii, grid_Sigma= grid_Sigma)


    sim.addgroup('FRIED', description = "FRIED grid used to calculate mass loss rates due to external photoevaporation")
    sim.FRIED.addgroup('Table', description = "(Resampled) Table of the mass loss rates for a given radial-Sigma grid.")
    sim.FRIED.Table.addfield("radii", grid_radii, description ="Outer disk radius input to calculate FRIED mass loss rates [AU], (array, nr)")
    sim.FRIED.Table.addfield("Sigma", grid_Sigma, description = "Surface density grid to calculate FRIED mass loss rates [g/cm^2] (array, nSigma)")
    sim.FRIED.Table.addfield("MassLoss", grid_MassLoss, description = "FRIED Mass loss rates [log10 (M_sun/year)] (grid, nr*nSigma)")



    # We use this private_Interpolator function to avoid constructing the FRIED interpolator multiple times
    sim.FRIED._Interpolator = grid_MassLoss_Interpolator



    # Add the truncation radius
    sim.FRIED.addfield('rTrunc', sim.grid.r[-1], description = 'Truncation radius [cm]')

    # Add the Mass Loss Rate field from the FRIED Grid
    sim.FRIED.addfield('MassLoss', np.zeros_like(sim.grid.r), description = 'Mass loss rate obtained by interpolating the FRIED Table at each grid cell [g/s]')


    sim.FRIED.rTrunc.updater = TruncationRadius
    sim.FRIED.MassLoss.updater =  MassLoss_FRIED
    sim.updater = ['star', 'grid', 'FRIED', 'gas', 'dust']
    sim.FRIED.updater = ['MassLoss', 'rTrunc']



    ###############################
    # DUST ENTRAINMENT
    ###############################
    # Add the entrainment size and the entrainment fraction for the dust loss rate.
    sim.dust.addfield('a_ent', sim.dust.a.T[-1], description = "Dust entrainment size [cm]")
    sim.dust.a_ent.updater = PhotoEntrainment_Size

    # The entrainment fraction needs to be updated before the dust source terms.
    sim.dust.updater = ['delta', 'rhos', 'fill', 'a', 'St', 'H', 'rho', 'backreaction', 'v', 'D', 'eps', 'kernel', 'p', 'a_ent','S']

    ###################################
    # ASSING GAS AND DUST LOSS RATES
    ###################################
    # Assign the External Photoevaporation Updater to the gas and dust
    sim.gas.S.ext.updater = SigmaDot_ExtPhoto
    sim.dust.S.ext.updater = SigmaDot_ExtPhoto_Dust



    ##################################
    # ADJUST THE GAS FLOOR VALUE
    ##################################
    # Setting higher floor value than the default avoids excessive mass loss rate calculations at the outer edge.
    # This speeds the code significantly, while still reproducing the results from Sellek et al.(2020)

    sim.gas.SigmaFloor = SigmaFloor


    sim.update()
