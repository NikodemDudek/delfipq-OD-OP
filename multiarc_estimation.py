# Load standard modules
import time
import sys

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy.astro.time_conversion import DateTime

# Load functions
import utilities as util

"""
This file estimates multiple arcs of observations from SGP4-propagated TLEs. It can also estimate constant or arc-wise drag coefficient.
"""

# Start timer
timer_start = time.time()

# Load spice kernels
spice.load_standard_kernels()

# Import tle data
tles, tles_cartesian_states, tles_epochs = util.import_tles()

# Model parameters
simulation_start_epoch = DateTime(2022, 2, 1).epoch()
simulation_end_epoch = DateTime(2022, 12, 30).epoch()
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
drag_arc_length = int(sys.argv[1])  # use depends on the model setting

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

# Performing estimation
estimation_output, initial_parameters = util.perform_estimation(
    estimatable_parameters,
    estimator,
    observation_collection,
    no_iterations)

# Processing the results
estimated_parameters, cartesian_final_residuals, tnw_final_residuals, rms_final_residuals, estimated_cartesian_states, estimated_epochs, dependent_variables, overlap_errors, rms_overlap_errors = util.process_results_multiarc_estimation(
    estimation_output,
    estimatable_parameters,
    sgp4_cartesian_states)

# End timer
timer_end = time.time()
timer = timer_end - timer_start

# Save the results
n_darcs = drag_arcs_lengths.shape[0]
C_D_est = estimated_parameters[-n_darcs:]
C_D_est = ' '.join(map(str, C_D_est))
new_row = {
    "t_start": simulation_start_epoch,
    "t_end": simulation_end_epoch,
    "dt_SGP4": sgp4_time_step,
    "dt_int": integrator_time_step,
    "dt_obs": observation_time_step,
    "i": no_iterations,
    "EH_deg": earth_harmonics[0],
    "EH_ord": earth_harmonics[1],
    "A_D": reference_area_drag,
    "A_R": reference_area_radiation,
    "C_D": drag_coefficient,
    "C_R": radiation_pressure_coefficient,
    "l_arc": arc_length,
    "l_darc": drag_arc_length,
    "if_const_C_D": estimate_constant_drag_coefficient,
    "if_arc_C_D": estimate_arcwise_drag_coefficient,
    "timer": timer,
    "RMS_residual":rms_final_residuals,
    "RMS_overlap": rms_overlap_errors,
    "C_D_est": C_D_est
    }
util.save_to_csv(
    new_row,
    "csvs\multiarc_estimation-RMSxC_D_estxl_darc.csv",
    include_head_row=False)

# Show the results
print("Initial paramaters: ", initial_parameters)
print("Estimated paramaters: ", estimated_parameters)
print("RMS difference between arcs: ", rms_overlap_errors)
print("RMS residual: ", rms_final_residuals)
util.plot_residuals_estimation(
    tnw_final_residuals,
    observation_epochs,
    tles_batch_epochs,
    show_tles_epochs=True)
util.plot_histogram(tnw_final_residuals)
util.plot_accelerations(dependent_variables, estimated_epochs)