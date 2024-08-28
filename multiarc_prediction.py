# Load standard modules
import time
import sys
import numpy as np

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy.astro.time_conversion import DateTime

# Load functions
import utilities as util


def form_state_arcs_drag_arcs(sgp4_cartesian_states, sgp4_epochs, state_arcs_start_epochs, state_arcs_initial_states, state_arcs_termination_epochs, drag_arcs_start_epochs, drag_coefficient, reference_area_drag, radiation_pressure_coefficient, reference_area_radiation, time_step, earth_harmonics, simulation_start_epoch, simulation_end_epoch, no_iterations):
    
    # Environment setup
    bodies = util.setup_environment_multiarc(
        sgp4_cartesian_states, 
        sgp4_epochs,
        drag_coefficient,
        reference_area_drag,
        radiation_pressure_coefficient,
        reference_area_radiation)

    # Propagation setup
    multiarc_propagator_settings = util.setup_propagation_multiarc(
        bodies, 
        state_arcs_start_epochs, 
        state_arcs_initial_states, 
        state_arcs_termination_epochs, 
        time_step,
        earth_harmonics)

    # Observation setup
    observation_settings, observation_collection, observation_positions, observation_epochs = util.setup_observation(
        simulation_start_epoch,
        simulation_end_epoch, 
        time_step, 
        bodies)

    # Estimation setup
    estimator, estimatable_parameters = util.setup_estimation_multiarc(
        multiarc_propagator_settings, 
        observation_settings, 
        bodies,
        state_arcs_start_epochs,
        estimate_constant_drag_coefficient=False,
        estimate_arcwise_drag_coefficient=True,
        drag_arcs_start_epochs=drag_arcs_start_epochs)

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
    
    # Extracting estiamted initial states and drag coefficients
    state_arcs_estm_initial_state = np.array([estimated_parameters[6*i:6*i+6] for i, _ in enumerate(state_arcs_start_epochs)])
    state_arcs_estm_drag_coefficient = estimated_parameters[-len(drag_arcs_start_epochs):]

    return state_arcs_estm_initial_state, state_arcs_estm_drag_coefficient

def form_model_prediction_arc(estimated_initial_state, estimated_drag_coefficient, estimation_arc_epochs, prediction_arc_epochs, estimation_arc_sgp4_states, prediction_arc_sgp4_states, reference_area_drag, radiation_pressure_coefficient, reference_area_radiation, time_step, earth_harmonics):
    
    # Environment setup
    bodies = util.setup_environment_singlearc(
        np.concatenate((estimation_arc_sgp4_states, prediction_arc_sgp4_states)),
        np.concatenate((estimation_arc_epochs, prediction_arc_epochs)),
        estimated_drag_coefficient,
        reference_area_drag,
        radiation_pressure_coefficient,
        reference_area_radiation)

    # Propagation setup
    prediction_propagator_settings = util.setup_propagation(
        bodies,
        estimated_initial_state,
        estimation_arc_epochs[0],
        prediction_arc_epochs[-1],
        time_step,
        earth_harmonics)

    # Performing propagation with prediction
    propagation_cartesian_states, propagation_epochs, dependent_variables = util.perform_propagation(
        bodies,
        prediction_propagator_settings)
    
    # Process the results
    prediction_arc_prediction_states = propagation_cartesian_states[propagation_epochs >= prediction_arc_epochs[0]]

    return prediction_arc_prediction_states

def form_sgp4_prediction_arc(tles_batch, tles_batch_epochs, prediction_arc_epochs, prediction_arc_termination_epoch, time_step):

    # Propagate last tle with sgp4 (conventional method)
    last_tle_index = np.where(tles_batch_epochs <= prediction_arc_epochs[0])[0][-1]
    last_tle = tles_batch[last_tle_index]
    conventional_cartesian_states, conventional_epochs = util.sgp4_propagate_tle_forward(
        prediction_arc_epochs[0],
        prediction_arc_termination_epoch,
        time_step,
        last_tle)

    return conventional_cartesian_states


def model(simulation_start_epoch, simulation_end_epoch, time_step, state_arc_length, drag_arc_length, prediction_arc_length, no_iterations, earth_harmonics, reference_area_drag, reference_area_radiation, drag_coefficient, radiation_pressure_coefficient, save_results, file_name):

    # Start timer
    timer_start = time.time()

    # Load spice kernels
    spice.load_standard_kernels()

    # Import tle data
    tles, tles_states, tles_epochs = util.import_tles()

    # Get one batch of tles
    tles_batch, tles_batch_states, tles_batch_epochs, tle_before_batch = util.load_one_tle_batch(
        simulation_start_epoch,
        simulation_end_epoch,
        tles,
        tles_states,
        tles_epochs)

    # Use SGP4 to propagate one batch of tles
    sgp4_states, sgp4_epochs = util.sgp4_propagate_tle_batch_forward(
        simulation_start_epoch,
        simulation_end_epoch,
        time_step,
        tles_batch,
        tles_batch_epochs,
        tle_before_batch)

    # Splitting epochs into state arcs, drag arcs and prediction arcs
    if drag_arc_length % state_arc_length != 0:
        raise Exception(f"Drag arc length {drag_arc_length} s is not an integer multiple of state arc length {state_arc_length} s")
    state_arcs_start_epochs, state_arcs_termination_epochs, state_arcs_initial_states, state_arcs_lengths, state_arcs_epochs, state_arcs_sgp4_states = util.setup_arcs(
        state_arc_length, 
        sgp4_states, 
        sgp4_epochs,
        time_step)
    drag_arcs_start_epochs, drag_arcs_termination_epochs, drag_arcs_initial_states, drag_arcs_lengths, drag_arcs_epochs, drag_arcs_sgp4_states = util.setup_arcs(
        drag_arc_length, 
        sgp4_states, 
        sgp4_epochs,
        time_step)
    pred_arcs_epochs, pred_arcs_sgp4_states, pred_arcs_termination_epochs, pred_arcs_lengths = util.setup_prediction_arcs(
        sgp4_epochs,
        sgp4_states,
        drag_arcs_termination_epochs,
        prediction_arc_length,
        time_step)

    # Perform estimation on state arcs and drag arcs
    state_arcs_estm_initial_state, state_arcs_estm_drag_coefficient = form_state_arcs_drag_arcs(
        sgp4_states,
        sgp4_epochs,
        state_arcs_start_epochs,
        state_arcs_initial_states,
        state_arcs_termination_epochs,
        drag_arcs_start_epochs,
        drag_coefficient,
        reference_area_drag,
        radiation_pressure_coefficient,
        reference_area_radiation,
        time_step,
        earth_harmonics,
        simulation_start_epoch,
        simulation_end_epoch,
        no_iterations)

    # Drop unused state arcs
    n = int(drag_arc_length / state_arc_length)
    state_arcs_start_epochs = state_arcs_start_epochs[n-1::n]
    state_arcs_termination_epochs = state_arcs_termination_epochs[n-1::n]
    state_arcs_initial_states = state_arcs_initial_states[n-1::n]
    state_arcs_lengths = state_arcs_lengths[n-1::n]
    state_arcs_epochs = state_arcs_epochs[n-1::n]
    state_arcs_sgp4_states = state_arcs_sgp4_states[n-1::n]
    state_arcs_estm_initial_state = state_arcs_estm_initial_state[n-1::n]

    # Perform propagation on model prediction arcs and sgp4 prediction arcs
    pred_arcs_model_pred_states = []
    pred_arcs_sgp4_pred_states = []
    for i, _ in enumerate(pred_arcs_lengths):

        # Forming model prediction arc
        pred_arc_model_pred_states = form_model_prediction_arc(
            state_arcs_estm_initial_state[i],
            state_arcs_estm_drag_coefficient[i],
            state_arcs_epochs[i],
            pred_arcs_epochs[i],
            state_arcs_sgp4_states[i],
            pred_arcs_sgp4_states[i],
            reference_area_drag,
            radiation_pressure_coefficient,
            reference_area_radiation,
            time_step,
            earth_harmonics)

        # Forming SGP4 prediction arc
        pred_arc_sgp4_pred_states = form_sgp4_prediction_arc(
            tles_batch,
            tles_batch_epochs,
            pred_arcs_epochs[i],
            pred_arcs_termination_epochs[i],
            time_step)

        # Appending states
        pred_arcs_model_pred_states.append(pred_arc_model_pred_states)
        pred_arcs_sgp4_pred_states.append(pred_arc_sgp4_pred_states)

    pred_arcs_model_pred_states = np.array(pred_arcs_model_pred_states)
    pred_arcs_sgp4_pred_states = np.array(pred_arcs_sgp4_pred_states)

    # Process results
    n_drop = int(3600 * 24 * 1 / time_step)
    residuals_model_prediction = (pred_arcs_sgp4_states - pred_arcs_model_pred_states)[:,n_drop:,:3]
    residuals_sgp4_prediction = (pred_arcs_sgp4_states - pred_arcs_sgp4_pred_states)[:,n_drop:,:3]
    rms_residuals_model_prediction = np.sqrt(np.average(np.square(residuals_model_prediction)))
    rms_residuals_sgp4_prediction = np.sqrt(np.average(np.square(residuals_sgp4_prediction)))

    # End timer
    timer_end = time.time()
    timer = timer_end - timer_start

    # Show results
    print("RMS residual model prediction:", rms_residuals_model_prediction)
    print("RMS residual SGP4 prediction:", rms_residuals_sgp4_prediction)

    # Save results
    if save_results:
        res_model = residuals_model_prediction.reshape((-1))
        res_model = ' '.join(map(str, res_model))
        res_sgp4 = residuals_sgp4_prediction.reshape((-1))
        res_sgp4 = ' '.join(map(str, res_sgp4))
        new_row = {
            "t_start": simulation_start_epoch,
            "t_end": simulation_end_epoch,
            "dt": time_step,
            "i": no_iterations,
            "EH_deg": earth_harmonics[0],
            "EH_ord": earth_harmonics[1],
            "A_D": reference_area_drag,
            "A_R": reference_area_radiation,
            "C_D": drag_coefficient,
            "C_R": radiation_pressure_coefficient,
            "l_state": state_arc_length,
            "l_drag": drag_arc_length,
            "l_pred": prediction_arc_length,
            "timer": timer,
            "RMS_res_model": rms_residuals_model_prediction,
            "RMS_res_sgp4": rms_residuals_sgp4_prediction,
            "res_model": res_model,
            "res_sgp4": res_sgp4
            }
        util.save_to_csv(
            new_row,
            file_name,
            include_head_row=False)


if __name__ == "__main__":

    # Model parameters
    simulation_start_epoch = DateTime(2022, 2, 1).epoch()
    simulation_end_epoch = DateTime(2022, 12, 30).epoch()
    time_step = 200
    state_arc_length = 3600 * 24 * 2
    drag_arc_length = 3600 * 24 * 14
    prediction_arc_length = 3600 * 24 * 14
    no_iterations = 4
    earth_harmonics = (10, 10)
    reference_area_drag = 0.013
    reference_area_radiation = 0.013
    drag_coefficient = 1.9
    radiation_pressure_coefficient = 1.3

    # Model settings
    save_results = False
    file_name = "csvs/multiarc_prediction.csv"

    # Run model
    model(
        simulation_start_epoch,
        simulation_end_epoch,
        time_step,
        state_arc_length,
        drag_arc_length,
        prediction_arc_length,
        no_iterations,
        earth_harmonics,
        reference_area_drag,
        reference_area_radiation,
        drag_coefficient,
        radiation_pressure_coefficient,
        save_results,
        file_name)
