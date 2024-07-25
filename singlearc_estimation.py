# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy.astro.time_conversion import DateTime

# Load functions
import utilities as util

"""
This file estimates a single arc of observations from SGP4-propagated TLEs. It can also estimate a constant drag coefficient.
"""

# Load spice kernels
spice.load_standard_kernels()

# Import tle data
tles, tles_cartesian_states, tles_epochs = util.import_tles()

# Model parameters
simulation_start_epoch = DateTime(2022, 6, 8).epoch()
simulation_end_epoch = DateTime(2022, 6, 9).epoch()
sgp4_time_step = 250
integrator_time_step = sgp4_time_step
observation_time_step = sgp4_time_step
no_iterations = 3
earth_harmonics = (10, 10)
reference_area_drag = 0.013
reference_area_radiation = 0.013
drag_coefficient = 2.3  # use depends on the model setting
radiation_pressure_coefficient = 1.3

# Model settings
estimate_constant_drag_coefficient = False

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

# Environment setup
bodies = util.setup_environment_singlearc(
    sgp4_cartesian_states,
    sgp4_epochs,
    drag_coefficient,
    reference_area_drag,
    radiation_pressure_coefficient,
    reference_area_radiation)

# Propagation setup
initial_state = sgp4_cartesian_states[0]
propagator_settings = util.setup_propagation(
    bodies,
    initial_state,
    simulation_start_epoch,
    simulation_end_epoch,
    integrator_time_step,
    earth_harmonics)

# Observation setup
observation_settings, observation_collection, observation_positions, observation_epochs = util.setup_observation(
    simulation_start_epoch, 
    simulation_end_epoch, 
    observation_time_step,
    bodies)

# Estimation setup
estimator, estimatable_parameters = util.setup_estimation_singlearc(
    propagator_settings, 
    observation_settings, 
    bodies,
    estimate_constant_drag_coefficient)

# Performing estimation
estimation_output, initial_parameters = util.perform_estimation(
    estimatable_parameters, 
    estimator, 
    observation_collection,
    no_iterations)

# Processing the results
estimated_parameters, cartesian_final_residuals, tnw_final_residuals, rms_final_residuals, estimated_cartesian_states, estimated_epochs, dependent_variables = util.process_results_singlearc_estimation(
    estimation_output,
    estimatable_parameters,
    sgp4_cartesian_states)

# Show the results
print("\nInitial paramaters: ", initial_parameters)
print("Estimated paramaters: ", estimated_parameters)
print("RMS residual: ", rms_final_residuals)
util.plot_residuals_estimation(
    tnw_final_residuals,
    observation_epochs,
    tles_batch_epochs,
    show_tles_epochs=True)
