# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy.astro.time_conversion import DateTime

# Load functions
import utilities as util

"""
This file estimates multiple arcs of observations from SGP4-propagated TLEs. It can also estimate constant or arc-wise drag coefficient.
"""

# Load spice kernels
spice.load_standard_kernels()

# Import tle data
tles, tles_cartesian_states, tles_epochs = util.import_tles()

# Model parameters
simulation_start_epoch = DateTime(2022, 4, 1).epoch()
simulation_end_epoch = DateTime(2022, 5, 1).epoch()
sgp4_time_step = 250
integrator_time_step = sgp4_time_step
observation_time_step = sgp4_time_step
no_iterations = 3
earth_harmonics = (10, 10)
reference_area_drag = 0.013
reference_area_radiation = 0.013
drag_coefficient = 1.9  # use depends on the model setting
radiation_pressure_coefficient = 1.3
arc_length = 3600 * 24
drag_arc_length = 3600 * 24 * 10  # use depends on the model setting

# Model settings
estimate_constant_drag_coefficient = False
estimate_arcwise_drag_coefficient = True

# Get one batch of tles
tles_batch, tles_batch_cartesian_states, tles_batch_epochs, tle_before_batch = util.load_one_tle_batch(
    simulation_start_epoch,
    simulation_end_epoch,
    tles,
    tles_cartesian_states,
    tles_epochs)

# Use SGP4 to propagate one batch of tles
sgp4_cartesian_states, sgp4_epochs = util.sgp4_propagate_tle_batch_forward(
    simulation_start_epoch,
    simulation_end_epoch,
    sgp4_time_step,
    tles_batch,
    tles_batch_epochs,
    tle_before_batch)

# Splitting epochs into arcs
arcs_start_epochs, arcs_termination_epochs, arcs_initial_states, arcs_lengths = util.setup_arcs(
    arc_length, 
    sgp4_cartesian_states, 
    sgp4_epochs)
drag_arcs_start_epochs, drag_arcs_termination_epochs, drag_arcs_initial_states, drag_arcs_lengths = util.setup_arcs(
    drag_arc_length, 
    sgp4_cartesian_states, 
    sgp4_epochs)

# Environment setup
bodies,earth_atmosphere = util.setup_environment_multiarc(
    sgp4_cartesian_states, 
    sgp4_epochs,
    drag_coefficient,
    reference_area_drag,
    radiation_pressure_coefficient,
    reference_area_radiation)

# Propagation setup
multiarc_propagator_settings = util.setup_propagation_multiarc(
    bodies, 
    arcs_start_epochs, 
    arcs_initial_states, 
    arcs_termination_epochs, 
    integrator_time_step,
    earth_harmonics)

# Observation setup
observation_settings, observation_collection, observation_positions, observation_epochs = util.setup_observation(
    simulation_start_epoch,
    simulation_end_epoch, 
    observation_time_step, 
    bodies)

# Estimation setup
estimator, estimatable_parameters = util.setup_estimation_multiarc(
    multiarc_propagator_settings, 
    observation_settings, 
    bodies,
    arcs_start_epochs,
    estimate_constant_drag_coefficient,
    estimate_arcwise_drag_coefficient,
    drag_arcs_start_epochs)

# Performing covariance analysis
covariance_output = util.preform_covariance_analysis(
    estimator,
    observation_collection)

# Show the results
util.plot_correlation_matrix(covariance_output)
