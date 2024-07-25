# Load required standard modules
import numpy as np
from copy import copy
import time

# Load required tudatpy modules
from tudatpy.interface import spice
from tudatpy.astro.time_conversion import DateTime

# Load functions
import utilities as util


# Model parameters
estimation_arc_length = 3600 * 24 * 2  # Testing
prediction_arc_length = 3600 * 24 * 2
drag_arc_length = 3600 * 24 * 2
simulation_start_epoch = DateTime(2022, 2, 1).epoch()
simulation_end_epoch = DateTime(2022, 12, 30).epoch()
sgp4_time_step = 250
no_iterations = 3
earth_harmonics = (10, 10)
reference_area_drag = 0.013
reference_area_radiation = 0.013
drag_coefficient = 1.9
radiation_pressure_coefficient = 1.3


def setup_drag_arcs(i, drag_arcs_sgp4_states, drag_arcs_epochs, drag_arcs_termination_epochs):
    
    # Environment setup
    bodies = util.setup_environment_singlearc(
        drag_arcs_sgp4_states[i],
        drag_arcs_epochs[i],
        drag_coefficient,
        reference_area_drag,
        radiation_pressure_coefficient,
        reference_area_radiation)

    # Propagation setup
    propagator_settings = util.setup_propagation(
        bodies,
        drag_arcs_sgp4_states[i, 0],
        drag_arcs_epochs[i, 0],
        drag_arcs_epochs[i, -1],
        integrator_time_step,
        earth_harmonics)

    # Observation setup
    observation_settings, observation_collection, observation_positions, observation_epochs = util.setup_observation(
        drag_arcs_epochs[i, 0],
        drag_arcs_termination_epochs[i], 
        observation_time_step, 
        bodies)

    # Estimation setup
    estimator, estimatable_parameters = util.setup_estimation_singlearc(
        propagator_settings,
        observation_settings,
        bodies,
        estimate_constant_drag_coefficient=True)

    # Performing estimation
    estimation_output, initial_parameters = util.perform_estimation(
        estimatable_parameters,
        estimator,
        observation_collection,
        no_iterations)

    # Process the results
    estimated_parameters = estimatable_parameters.parameter_vector
    estimated_drag_coefficient = estimated_parameters[-1]
    print("Estimated drag coefficient:", estimated_drag_coefficient)
    print("Drag arc established.", "\n")

    return estimated_drag_coefficient

def setup_estimation_arcs(i, estimation_arcs_sgp4_states, estimation_arcs_epochs, estimation_arcs_termination_epochs, estimated_drag_coefficient):
    
    # Environment setup
    bodies = util.setup_environment_singlearc(
        estimation_arcs_sgp4_states[i],
        estimation_arcs_epochs[i],
        estimated_drag_coefficient,
        reference_area_drag,
        radiation_pressure_coefficient,
        reference_area_radiation)

    # Propagation setup
    propagator_settings = util.setup_propagation(
        bodies,
        estimation_arcs_sgp4_states[i, 0],
        estimation_arcs_epochs[i, 0],
        estimation_arcs_epochs[i, -1],
        integrator_time_step,
        earth_harmonics)

    # Observation setup
    observation_settings, observation_collection, observation_positions, observation_epochs = util.setup_observation(
        estimation_arcs_epochs[i, 0],
        estimation_arcs_termination_epochs[i], 
        observation_time_step, 
        bodies)

    # Estimation setup
    estimator, estimatable_parameters = util.setup_estimation_singlearc(
        propagator_settings,
        observation_settings,
        bodies,
        estimate_constant_drag_coefficient=False)

    # Performing estimation
    estimation_output, initial_parameters = util.perform_estimation(
        estimatable_parameters,
        estimator,
        observation_collection,
        no_iterations)
    
    # Process the results
    estimated_parameters = estimatable_parameters.parameter_vector
    estimated_initial_state = copy(estimated_parameters)
    print("Estimated initial state:", estimated_initial_state)
    print("Estimation arc established.", "\n")

    return estimated_initial_state, bodies

def setup_prediction_arcs(i, bodies, estimated_initial_state, estimation_arcs_epochs, prediction_arcs_epochs):

    # Propagation setup
    prediction_propagator_settings = util.setup_propagation(
        bodies,
        estimated_initial_state,
        estimation_arcs_epochs[i, 0],
        prediction_arcs_epochs[i, -1],
        integrator_time_step,
        earth_harmonics)

    # Performing propagation with prediction
    propagation_cartesian_states, propagation_epochs, dependent_variables = util.perform_propagation(
        bodies,
        prediction_propagator_settings)
    
    # Process the results
    prediction_arc_prediction_states = propagation_cartesian_states[propagation_epochs >= prediction_arcs_epochs[i,0]]
    print("Prediction arc established.", "\n")

    return prediction_arc_prediction_states

def setup_conventional_arcs(i, tles_batch, tles_batch_epochs, prediction_arcs_epochs, prediction_arcs_termination_epochs):

    # Propagate last tle with sgp4 (conventional method)
    last_tle_index = np.where(tles_batch_epochs <= prediction_arcs_epochs[i,0])[0][-1]
    last_tle = tles_batch[last_tle_index]
    conventional_cartesian_states, conventional_epochs = util.sgp4_propagate_tle_forward(
        prediction_arcs_epochs[i,0],
        prediction_arcs_termination_epochs[i],
        sgp4_time_step,
        last_tle)
    
    # Process the results
    print("Conventional arc established.", "\n")

    return conventional_cartesian_states


# Start timer
timer_start = time.time()

# Load spice kernels
spice.load_standard_kernels()

# Import tle data
tles, tles_cartesian_states, tles_epochs = util.import_tles()

# Set simulation epochs
integrator_time_step = sgp4_time_step
observation_time_step = sgp4_time_step

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

# Divide SGP4 states into estimation, drag and prediction arcs
estimation_arcs_epochs, estimation_arcs_sgp4_states, estimation_arcs_termination_epochs, estimation_arcs_lengths, prediction_arcs_epochs, prediction_arcs_sgp4_states, prediction_arcs_termination_epochs, prediction_arcs_lengths, drag_arcs_epochs, drag_arcs_sgp4_states, drag_arcs_termination_epochs, drag_arcs_lengths = util.setup_arcs_multiarc_prediction(
    estimation_arc_length,
    prediction_arc_length,
    drag_arc_length,
    sgp4_cartesian_states,
    sgp4_epochs,
    sgp4_time_step)

# Iterate over the arcs
prediction_arcs_prediction_states = []
conventional_arcs_sgp4_states = []
for i, _ in enumerate(estimation_arcs_lengths):

    # Forming drag arc
    estimated_drag_coefficient = setup_drag_arcs(i, drag_arcs_sgp4_states, drag_arcs_epochs, drag_arcs_termination_epochs)

    # Forming estimation arc
    estimated_initial_state, bodies = setup_estimation_arcs(i, estimation_arcs_sgp4_states, estimation_arcs_epochs, estimation_arcs_termination_epochs, estimated_drag_coefficient)

    # Forming prediction arc
    prediction_arc_prediction_states = setup_prediction_arcs(i, bodies, estimated_initial_state, estimation_arcs_epochs, prediction_arcs_epochs)

    # Forming conventional arc
    conventional_cartesian_states = setup_conventional_arcs(i, tles_batch, tles_batch_epochs, prediction_arcs_epochs, prediction_arcs_termination_epochs)

    # Append states
    prediction_arcs_prediction_states.append(prediction_arc_prediction_states)
    conventional_arcs_sgp4_states.append(conventional_cartesian_states)

    print(f"Completed iteration {i}/{len(estimation_arcs_lengths)}.", "\n")

prediction_arcs_prediction_states = np.array(prediction_arcs_prediction_states)
conventional_arcs_sgp4_states = np.array(conventional_arcs_sgp4_states)

# Process results
cartesian_final_residuals_prediction = (prediction_arcs_sgp4_states - prediction_arcs_prediction_states)[:,:,:3]
cartesian_final_residuals_conventional = (prediction_arcs_sgp4_states - conventional_arcs_sgp4_states)[:,:,:3]
rms_final_residuals_prediction = np.sqrt(np.average(np.square(cartesian_final_residuals_prediction)))
rms_final_residuals_conventional = np.sqrt(np.average(np.square(cartesian_final_residuals_conventional)))
print("RMS prediction residual:", rms_final_residuals_prediction)
print("RMS conventional prediction residual:", rms_final_residuals_conventional)

# End timer
timer_end = time.time()
timer = timer_end - timer_start

# Save the results
new_row = {
    "t_start": simulation_start_epoch,
    "t_end": simulation_end_epoch,
    "dt_sgp4": sgp4_time_step,
    "i": no_iterations,
    "EH_deg": earth_harmonics[0],
    "EH_ord": earth_harmonics[1],
    "A_D": reference_area_drag,
    "A_R": reference_area_radiation,
    "C_D": drag_coefficient,
    "C_R": radiation_pressure_coefficient,
    "l_estm": estimation_arc_length,
    "l_pred": prediction_arc_length,
    "l_drag": drag_arc_length,
    "timer": timer,
    "RMS_pred": rms_final_residuals_prediction,
    "RMS_conv": rms_final_residuals_conventional
}
util.save_to_csv(
    new_row,
    "csvs\multiarc_prediction-RMS_predxl_estm.csv",
    include_head_row=True)
