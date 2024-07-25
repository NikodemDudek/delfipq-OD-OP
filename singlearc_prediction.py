# Load required standard modules
import numpy as np

# Load required tudatpy modules
from tudatpy.interface import spice
from tudatpy.astro.time_conversion import DateTime

# Load functions
import utilities as util

"""
This file estimates a single arc of observations from SGP4-propagated TLEs. Then it propagates this arc to form the prediction. The prediction is compared to the SGP4 propgatation of one last tle (this is called the conventional method).
"""

# Model parameters
estimation_arc_length = 3600 * 24 * 1
prediction_arc_length = 3600 * 24 * 10
prediction_start_epoch = DateTime(2022, 6, 12).epoch()  # 25/06/2022 is presentable
sgp4_time_step = 250
no_iterations = 3
earth_harmonics = (10, 10)
reference_area_drag = 0.013
reference_area_radiation = 0.013
drag_coefficient = 1.2494016021977434  # use depends on the model settings
radiation_pressure_coefficient = 1.3

# Model settings
estimate_constant_drag_coefficient = False


def setup_estimation_arc():

    # Environment setup
    bodies = util.setup_environment_singlearc(
        sgp4_cartesian_states_estimation,
        sgp4_epochs,
        drag_coefficient,
        reference_area_drag,
        radiation_pressure_coefficient,
        reference_area_radiation)

    # Propagation setup
    initial_state = sgp4_cartesian_states_estimation[0]
    propagator_settings = util.setup_propagation(
        bodies,
        initial_state,
        simulation_start_epoch,
        prediction_start_epoch,
        integrator_time_step,
        earth_harmonics)

    # Observation setup
    observation_settings, observation_collection, observation_positions, observation_epochs = util.setup_observation(
        simulation_start_epoch,
        prediction_start_epoch, 
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
    
    return estimatable_parameters, bodies
    
def setup_prediction_arc(estimated_initial_state, bodies):
    
    # New propagation setup
    prediction_propagator_settings = util.setup_propagation(
        bodies,
        estimated_initial_state,
        simulation_start_epoch,
        simulation_end_epoch,
        integrator_time_step,
        earth_harmonics)

    # Performing propagation with prediction
    propagation_cartesian_states, propagation_epochs, dependent_variables = util.perform_propagation(
        bodies,
        prediction_propagator_settings)
    
    return propagation_epochs, propagation_cartesian_states


# Load spice kernels
spice.load_standard_kernels()

# Import tle data
tles, tles_cartesian_states, tles_epochs = util.import_tles()

# Set simulation epochs
simulation_start_epoch = prediction_start_epoch - estimation_arc_length
simulation_end_epoch = prediction_start_epoch + prediction_arc_length
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

# Split SGP4 states into estimation and prediction arcs
sgp4_epochs_estimation = sgp4_epochs[sgp4_epochs < prediction_start_epoch]
sgp4_epochs_prediction = sgp4_epochs[sgp4_epochs >= prediction_start_epoch]
sgp4_cartesian_states_estimation = sgp4_cartesian_states[sgp4_epochs < prediction_start_epoch]
sgp4_cartesian_states_prediction = sgp4_cartesian_states[sgp4_epochs >= prediction_start_epoch]
prediction_start_epoch = sgp4_epochs_prediction[0]

# Forming estimation arc
estimatable_parameters, bodies = setup_estimation_arc()
estimated_initial_state = estimatable_parameters.parameter_vector[:6]

# Forming prediction arc
propagation_epochs, propagation_cartesian_states = setup_prediction_arc(estimated_initial_state, bodies)

# Forming conventional arc (propagate last tle with sgp4)
last_tle_index = np.where(tles_batch_epochs <= prediction_start_epoch)[0][-1]
last_tle = tles_batch[last_tle_index]
conventional_cartesian_states, conventional_epochs = util.sgp4_propagate_tle_forward(
    prediction_start_epoch,
    simulation_end_epoch,
    sgp4_time_step,
    last_tle)

# Split propagation states into estimation and prediction
propagation_cartesian_states_estimation = propagation_cartesian_states[propagation_epochs < prediction_start_epoch]
propagation_cartesian_states_prediction = propagation_cartesian_states[propagation_epochs >= prediction_start_epoch][:-1]  # Remove last epoch to comply with SGP4 epochs
propagation_epochs_estimation = propagation_epochs[propagation_epochs < prediction_start_epoch]
propagation_epochs_prediction = propagation_epochs[propagation_epochs >= prediction_start_epoch][:-1]

# Processing the results
cartesian_final_residuals_estimation, cartesian_final_residuals_prediction, cartesian_final_residuals_conventional, tnw_final_residuals_estimation, tnw_final_residuals_prediction, tnw_final_residuals_conventional, rms_final_residuals_estimation, rms_final_residuals_prediction, rms_final_residuals_conventional = util.process_results_singlearc_prediction(
    sgp4_cartesian_states_estimation,
    sgp4_cartesian_states_prediction,
    propagation_cartesian_states_estimation,
    propagation_cartesian_states_prediction,
    conventional_cartesian_states)

# Show the results
print("\nRMS estimation residual:", rms_final_residuals_estimation)
if estimate_constant_drag_coefficient:
    print("Estimated drag coefficient:", estimatable_parameters.parameter_vector[-1])
print("RMS prediction residual:", rms_final_residuals_prediction)
print("RMS conventional prediction residual:", rms_final_residuals_conventional)
util.plot_residuals_singlearc_prediction(
    np.concatenate((tnw_final_residuals_estimation, tnw_final_residuals_prediction)),
    tnw_final_residuals_conventional,
    conventional_epochs,
    sgp4_epochs,
    tles_batch_epochs,
    prediction_start_epoch,
    show_tles_epochs=True)
