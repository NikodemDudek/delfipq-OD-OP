# Load required standard modules
import numpy as np
from copy import copy

# Load required tudatpy modules
from tudatpy.interface import spice
from tudatpy.astro.time_conversion import DateTime

# Load functions
import utilities as util


def setup_estimation_arc(sgp4_epochs_estimation, sgp4_cartesian_states_estimation):

    # Environment setup
    bodies = util.setup_environment_singlearc(
        sgp4_cartesian_states_estimation,
        sgp4_epochs_estimation,
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
        time_step,
        earth_harmonics)

    # Observation setup
    observation_settings, observation_collection, observation_positions, observation_epochs = util.setup_observation(
        simulation_start_epoch,
        prediction_start_epoch, 
        time_step, 
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
    
    return estimatable_parameters
    
def setup_prediction_arc(sgp4_epochs, sgp4_cartesian_states, estimated_initial_state, drag_coefficient_prediction):

    # Environment setup
    bodies = util.setup_environment_singlearc(
        sgp4_cartesian_states,
        sgp4_epochs,
        drag_coefficient_prediction,
        reference_area_drag,
        radiation_pressure_coefficient,
        reference_area_radiation)
    
    # Propagation setup
    prediction_propagator_settings = util.setup_propagation(
        bodies,
        estimated_initial_state,
        simulation_start_epoch,
        simulation_end_epoch,
        time_step,
        earth_harmonics)

    # Performing propagation with prediction
    propagation_cartesian_states, propagation_epochs, dependent_variables = util.perform_propagation(
        bodies,
        prediction_propagator_settings)
    
    return propagation_epochs, propagation_cartesian_states


def model(estimation_arc_length, prediction_arc_length, prediction_start_epoch, time_step, no_iterations, earth_harmonics, reference_area_drag, reference_area_radiation, drag_coefficient, radiation_pressure_coefficient, estimate_constant_drag_coefficient, simulation_start_epoch, simulation_end_epoch):

    # Load spice kernels
    spice.load_standard_kernels()

    # Import tle data
    tles, tles_cartesian_states, tles_epochs = util.import_tles()

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
        time_step,
        tles_batch,
        tles_batch_epochs,
        tle_before_batch)

    # Split SGP4 states into estimation and prediction arcs
    sgp4_epochs_estimation = sgp4_epochs[sgp4_epochs < prediction_start_epoch]
    sgp4_epochs_prediction = sgp4_epochs[sgp4_epochs >= prediction_start_epoch]
    sgp4_cartesian_states_estimation = sgp4_cartesian_states[sgp4_epochs < prediction_start_epoch]
    sgp4_cartesian_states_prediction = sgp4_cartesian_states[sgp4_epochs >= prediction_start_epoch]
    prediction_start_epoch_updated = sgp4_epochs_prediction[0]

    # Forming estimation arc
    estimatable_parameters = setup_estimation_arc(sgp4_epochs_estimation, sgp4_cartesian_states_estimation)

    # Forming prediction arc
    estimated_initial_state = estimatable_parameters.parameter_vector[:6]
    if estimate_constant_drag_coefficient:
        drag_coefficient_prediction = estimatable_parameters.parameter_vector[-1]
    else:
        drag_coefficient_prediction = copy(drag_coefficient)
    propagation_epochs, propagation_cartesian_states = setup_prediction_arc(sgp4_epochs, sgp4_cartesian_states, estimated_initial_state, drag_coefficient_prediction)

    # Forming conventional arc (propagate last tle with sgp4)
    last_tle_index = np.where(tles_batch_epochs <= prediction_start_epoch_updated)[0][-1]
    last_tle = tles_batch[last_tle_index]
    conventional_cartesian_states, conventional_epochs = util.sgp4_propagate_tle_forward(
        prediction_start_epoch_updated,
        simulation_end_epoch,
        time_step,
        last_tle)

    # Split propagation states into estimation and prediction
    propagation_cartesian_states_estimation = propagation_cartesian_states[propagation_epochs < prediction_start_epoch_updated]
    propagation_cartesian_states_prediction = propagation_cartesian_states[propagation_epochs >= prediction_start_epoch_updated][:-1]  # Remove last epoch to comply with SGP4 epochs
    propagation_epochs_estimation = propagation_epochs[propagation_epochs < prediction_start_epoch_updated]
    propagation_epochs_prediction = propagation_epochs[propagation_epochs >= prediction_start_epoch_updated][:-1]

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
    print("Estimated initial state:", list(estimated_initial_state))
    print("RMS prediction residual:", rms_final_residuals_prediction)
    print("RMS conventional prediction residual:", rms_final_residuals_conventional, "\n")
    util.plot_residuals_singlearc_prediction(
        np.concatenate((tnw_final_residuals_estimation, tnw_final_residuals_prediction)),
        tnw_final_residuals_conventional,
        conventional_epochs,
        sgp4_epochs,
        tles_batch_epochs,
        prediction_start_epoch_updated,
        show_tles_epochs=True)

if __name__ == "__main__":

    # Model parameters
    estimation_arc_length = 3600 * 24 * 2
    prediction_arc_length = 3600 * 24 * 4
    prediction_start_epoch = DateTime(2022, 7, 20).epoch()
    time_step = 200
    no_iterations = 4
    earth_harmonics = (10, 10)
    reference_area_drag = 0.013
    reference_area_radiation = 0.013
    drag_coefficient = 1.7542180152006401
    radiation_pressure_coefficient = 1.3

    # Model settings
    estimate_constant_drag_coefficient = False

    # Simulation time
    simulation_start_epoch = prediction_start_epoch - estimation_arc_length
    simulation_end_epoch = prediction_start_epoch + prediction_arc_length

    # Run model
    model(
        estimation_arc_length,
        prediction_arc_length,
        prediction_start_epoch,
        time_step,
        no_iterations,
        earth_harmonics,
        reference_area_drag,
        reference_area_radiation,
        drag_coefficient,
        radiation_pressure_coefficient,
        estimate_constant_drag_coefficient,
        simulation_start_epoch,
        simulation_end_epoch
    )
