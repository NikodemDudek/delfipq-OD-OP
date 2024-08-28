# Load required standard modules
import numpy as np
from matplotlib import pyplot as plt
import csv
from copy import copy

# Load required tudatpy modules
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import environment
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import estimation_setup
from tudatpy.numerical_simulation import estimation
from tudatpy.numerical_simulation.estimation_setup import observation
from tudatpy.astro import element_conversion
from tudatpy.astro import time_conversion
from tudatpy.astro import frame_conversion
from tudatpy.util import result2array

"""
This file provides the functions used in other files.
"""

def transform_tles(original_array):
    result_array = []

    for i in range(0, len(original_array), 2):
        subarray = original_array[i:i+2]
        result_array.append(subarray)

    return result_array

def import_tles():
    
    # Import tles
    file = open("TLEs/tle.txt", "r")
    tle_lines = file.read().splitlines()
    file.close()

    # Transform tles to (n, 2) array
    tles = transform_tles(tle_lines)

    # Clean data by removing duplicates
    tles = np.unique(tles, axis=0)

    # Extract cartesian states from tles
    cartesian_states = []
    epochs = []
    time_scale_converter = time_conversion.default_time_scale_converter()
    for i, tle in enumerate(tles):
        year = int(tles[i][0][18:20]) + 2000
        seconds = (float(tles[i][0][20:32]) - 1) * 24 * 3600
        epoch = time_conversion.epoch_from_date_time_components(year, 1, 1, 0, 0, 0) + seconds
        epochs.append(epoch)
        
        Tle_object = environment.Tle(tles[i][0], tles[i][1])
        Tle_ephemeris = environment.TleEphemeris("Earth", "J2000", Tle_object, False)
        epoch_TDB = time_scale_converter.convert_time(
                input_scale=time_conversion.utc_scale,
                output_scale=time_conversion.tdb_scale,
                input_value=epoch)
        cartesian_state = Tle_ephemeris.cartesian_state(epoch_TDB)
        cartesian_states.append(cartesian_state)
    
    # Ensure that epochs are strictly increasing
    strictly_increasing = all(i < j for i, j in zip(epochs, epochs[1:]))
    if not strictly_increasing:
        raise ValueError('TLEs epochs are not strictly increasing!')

    # Transform lists into arrays
    cartesian_states = np.array(cartesian_states)
    epochs = np.array(epochs)
    tles = np.array(tles)

    return tles, cartesian_states, epochs

def load_one_tle_batch(start_epoch, end_epoch, tles, tles_cartesian_states, tles_epochs):

    batch_indices = np.where((tles_epochs >= start_epoch) & (tles_epochs <= end_epoch))
    
    tles_batch = tles[batch_indices]
    tles_batch_cartesian_states = tles_cartesian_states[batch_indices]
    tles_batch_epochs = tles_epochs[batch_indices]

    tle_before_batch = tles[batch_indices[0][0] - 1]
    if batch_indices[0][0] == 0:
        tle_before_batch = None
        print("WARNING: no TLE before the batch")

    return tles_batch, tles_batch_cartesian_states, tles_batch_epochs, tle_before_batch


def sgp4_propagate_tle_forward(simulation_start_epoch, simulation_end_epoch, sgp4_time_step, tle):

    sgp4_epochs = np.arange(simulation_start_epoch, simulation_end_epoch, sgp4_time_step)
    sgp4_cartesian_states = []
    time_scale_converter = time_conversion.default_time_scale_converter()

    for sgp4_epoch in sgp4_epochs:

        # Propagate TLE
        Tle_object = environment.Tle(tle[0], tle[1])
        Tle_ephemeris = environment.TleEphemeris("Earth", "J2000", Tle_object, False)
        
        # Append to the array
        epoch_TDB = time_scale_converter.convert_time(
            input_scale=time_conversion.utc_scale,
            output_scale=time_conversion.tdb_scale,
            input_value=sgp4_epoch)
        sgp4_cartesian_state = Tle_ephemeris.cartesian_state(epoch_TDB)
        sgp4_cartesian_states.append(sgp4_cartesian_state)
        
    sgp4_epochs = np.array(sgp4_epochs)
    sgp4_cartesian_states = np.array(sgp4_cartesian_states)

    return sgp4_cartesian_states, sgp4_epochs

def sgp4_propagate_tle_batch_forward(simulation_start_epoch, simulation_end_epoch, sgp4_time_step, tles_batch, tles_batch_epochs, tle_before_batch):
       
    """
    The epochs are in the a range [simulation_start_epoch, simulation_end_epoch) with equal interval sgp4_time_step; tles from the batch are rather not included
    """

    sgp4_epochs = np.arange(simulation_start_epoch, simulation_end_epoch, sgp4_time_step)
    sgp4_cartesian_states = []
    time_scale_converter = time_conversion.default_time_scale_converter()


    for i, sgp4_epoch in enumerate(sgp4_epochs):
        
        # Find the right TLE to propagate
        if sgp4_epoch < tles_batch_epochs[0]:
            tle = tle_before_batch
        else: 
            for j, tle_batch_epoch in enumerate(tles_batch_epochs):
                if tle_batch_epoch <= sgp4_epoch:
                    tle = tles_batch[j]    

        # Propagate TLE
        Tle_object = environment.Tle(tle[0], tle[1])
        Tle_ephemeris = environment.TleEphemeris("Earth", "J2000", Tle_object, False)
        
        # Append to the array
        epoch_TDB = time_scale_converter.convert_time(
            input_scale=time_conversion.utc_scale,
            output_scale=time_conversion.tdb_scale,
            input_value=sgp4_epoch)
        sgp4_cartesian_state = Tle_ephemeris.cartesian_state(epoch_TDB)
        sgp4_cartesian_states.append(sgp4_cartesian_state)
        
    sgp4_epochs = np.array(sgp4_epochs)
    sgp4_cartesian_states = np.array(sgp4_cartesian_states)

    return sgp4_cartesian_states, sgp4_epochs

def setup_arcs(arc_length, sgp4_states, sgp4_epochs, time_step):

    """
    Start epoch is the first epochs of an arc.
    Termination epoch is the first epoch of the following arc.
    Arc length is the time between the start and termination epochs.
    """
    
    arcs_start_epochs = [sgp4_epochs[0]]
    arcs_initial_states = [sgp4_states[0]]
    arcs_termination_epochs = []
    arcs_lengths = []

    arc_epoch = sgp4_epochs[0] + arc_length
    for i, _ in enumerate(sgp4_epochs):
        if sgp4_epochs[i] >= arc_epoch:
            
            arcs_start_epochs.append(sgp4_epochs[i])
            arcs_initial_states.append(sgp4_states[i])
            arcs_termination_epochs.append(sgp4_epochs[i])
            arc_epoch += arc_length
    
    arcs_termination_epochs.append(sgp4_epochs[-1])

    if (arcs_termination_epochs[-1] - arcs_start_epochs[-1]) < (0.3 * arc_length):
        arcs_start_epochs.pop()
        arcs_initial_states.pop()
        arcs_termination_epochs.pop(-2)

    arcs_start_epochs = np.array(arcs_start_epochs)
    arcs_termination_epochs = np.array(arcs_termination_epochs)
    arcs_initial_states = np.array(arcs_initial_states)
    arcs_lengths = arcs_termination_epochs - arcs_start_epochs

    # Split epochs into arcs
    arcs_epochs = [np.arange(arcs_start_epochs[i], arcs_termination_epochs[i], time_step) for i, _ in enumerate(arcs_lengths)]

    # Split states into arcs
    arcs_sgp4_states = []
    index = 0
    for sublist in arcs_epochs:
        length = len(sublist)
        arcs_sgp4_states.append(sgp4_states[index:index + length])
        index += length

    return arcs_start_epochs, arcs_termination_epochs, arcs_initial_states, arcs_lengths, arcs_epochs, arcs_sgp4_states

def setup_prediction_arcs(sgp4_epochs, sgp4_states, drag_arcs_termination_epochs, prediction_arc_length, time_step):
    
    # Get initial epochs
    pred_arcs_epochs_0 = [np.arange(drag_arcs_termination_epochs[i], drag_arcs_termination_epochs[i] + prediction_arc_length, time_step) for i, _ in enumerate(drag_arcs_termination_epochs)]

    # Remove epochs from outside SGP4 epochs
    pred_arcs_epochs = []
    for i, arc_epochs in enumerate(pred_arcs_epochs_0):
        if arc_epochs[-1] < sgp4_epochs[-1]:
            pred_arcs_epochs.append(arc_epochs)
    pred_arcs_epochs = np.array(pred_arcs_epochs)

    # Get SGP4 states
    pred_arcs_sgp4_states = np.array([sgp4_states[(sgp4_epochs >= arc_epochs[0]) & (sgp4_epochs <= arc_epochs[-1])] for arc_epochs in pred_arcs_epochs])

    # Get arcs termination epochs
    pred_arcs_termination_epochs = pred_arcs_epochs[:, -1] + time_step
    
    # Get arcs length
    pred_arcs_lengths = pred_arcs_termination_epochs - pred_arcs_epochs[:, 0]

    return pred_arcs_epochs, pred_arcs_sgp4_states, pred_arcs_termination_epochs, pred_arcs_lengths

def setup_arcs_multiarc_prediction(estimation_arc_length, prediction_arc_length, drag_arc_length, sgp4_cartesian_states, sgp4_epochs, sgp4_time_step):

    # Calulate number of data points for each arc type
    estimation_arc_size = int(estimation_arc_length / sgp4_time_step)
    prediction_arc_size = int(prediction_arc_length / sgp4_time_step)
    drag_arc_size = int(drag_arc_length / sgp4_time_step)

    # Choose starting points
    if estimation_arc_size > drag_arc_size:
        i_0 = 0
    else:
        i_0 = drag_arc_size - estimation_arc_size

    # Divide epochs
    estimation_arcs_epochs = np.array([sgp4_epochs[i:i+estimation_arc_size] for i in range(i_0, len(sgp4_epochs)-estimation_arc_size-prediction_arc_size-1, estimation_arc_size)])  # Iteration over first epochs of each estimation arc
    drag_arcs_epochs = np.array([sgp4_epochs[i+estimation_arc_size-drag_arc_size:i+estimation_arc_size] for i in range(i_0, len(sgp4_epochs)-estimation_arc_size-prediction_arc_size-1, estimation_arc_size)])
    prediction_arcs_epochs = np.array([sgp4_epochs[i+estimation_arc_size:i+estimation_arc_size+prediction_arc_size] for i in range(i_0, len(sgp4_epochs)-estimation_arc_size-prediction_arc_size-1, estimation_arc_size)])

    # Calculate other arrays
    estimation_arcs_states = np.array([sgp4_cartesian_states[i:i+estimation_arc_size] for i in range(i_0, len(sgp4_epochs)-estimation_arc_size-prediction_arc_size-1, estimation_arc_size)])
    estimation_arcs_start_epochs = np.array([arc_epochs[0] for arc_epochs in estimation_arcs_epochs])
    estimation_arcs_termination_epochs = np.array([arc_epochs[-1]+sgp4_time_step for arc_epochs in estimation_arcs_epochs])
    estimation_arcs_lengths = estimation_arcs_termination_epochs - estimation_arcs_start_epochs

    prediction_arcs_states = np.array([sgp4_cartesian_states[i+estimation_arc_size:i+estimation_arc_size+prediction_arc_size] for i in range(i_0, len(sgp4_epochs)-estimation_arc_size-prediction_arc_size-1, estimation_arc_size)])
    prediction_arcs_start_epochs = np.array([arc_epochs[0] for arc_epochs in prediction_arcs_epochs])
    prediction_arcs_termination_epochs = np.array([arc_epochs[-1]+sgp4_time_step for arc_epochs in prediction_arcs_epochs])
    prediction_arcs_lengths = prediction_arcs_termination_epochs - prediction_arcs_start_epochs

    drag_arcs_states = np.array([sgp4_cartesian_states[i+estimation_arc_size-drag_arc_size:i+estimation_arc_size] for i in range(i_0, len(sgp4_epochs)-estimation_arc_size-prediction_arc_size-1, estimation_arc_size)])
    drag_arcs_start_epochs = np.array([arc_epochs[0] for arc_epochs in drag_arcs_epochs])
    drag_arcs_termination_epochs = np.array([arc_epochs[-1]+sgp4_time_step for arc_epochs in drag_arcs_epochs])
    drag_arcs_lengths = drag_arcs_termination_epochs - drag_arcs_start_epochs

    return estimation_arcs_epochs, estimation_arcs_states, estimation_arcs_termination_epochs, estimation_arcs_lengths, prediction_arcs_epochs, prediction_arcs_states, prediction_arcs_termination_epochs, prediction_arcs_lengths, drag_arcs_epochs, drag_arcs_states, drag_arcs_termination_epochs, drag_arcs_lengths

def cartesian_states2dict(cartesian_states, epochs):
    state_history_dict = {}

    for epoch, state in zip(epochs, cartesian_states):
        state_history_dict[epoch] = state

    return state_history_dict

def setup_environment_singlearc(sgp4_cartesian_states, epochs, drag_coefficient, reference_area_drag, radiation_pressure_coefficient, reference_area_radiation):

    # Create default body settings for "Sun", "Earth", "Moon", "Mars", and "Venus"
    bodies_to_create = ["Sun", "Earth", "Moon", "Mars", "Venus", "Jupiter"]

    # Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create,
        global_frame_origin,
        global_frame_orientation)
    body_settings.add_empty_settings("Delfi-PQ")
    body_settings.get("Delfi-PQ").constant_mass = 0.6
    
    # Inerpolate cartesian states from SGP4 and add them as ephemeris to Delfi-PQ
    state_history_dict = cartesian_states2dict(sgp4_cartesian_states, epochs)
    delfi_ephemeris = environment_setup.ephemeris.tabulated(
        state_history_dict,
        global_frame_origin,
        global_frame_orientation
    )
    body_settings.get("Delfi-PQ").ephemeris_settings = delfi_ephemeris

    # Create system of bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Create aerodynamic coefficient interface settings
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area_drag,
        [drag_coefficient, 0.0, 0.0]
    )
    # Add the aerodynamic interface to the environment
    environment_setup.add_aerodynamic_coefficient_interface(bodies, "Delfi-PQ", aero_coefficient_settings)

    # Create radiation pressure settings
    occulting_bodies_dict = dict()
    occulting_bodies_dict["Sun"] = ["Earth"]
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
        reference_area_radiation,
        radiation_pressure_coefficient,
        occulting_bodies_dict
    )

    # Add the radiation pressure interface to the environment
    environment_setup.add_radiation_pressure_target_model(
        bodies,
        "Delfi-PQ",
        radiation_pressure_settings
    )

    return bodies

def setup_environment_multiarc(sgp4_cartesian_states, epochs, drag_coefficient, reference_area_drag, radiation_pressure_coefficient, reference_area_radiation):

    # Create default body settings for "Sun", "Earth", "Moon", "Mars", and "Venus"
    bodies_to_create = ["Sun", "Earth", "Moon", "Mars", "Venus", "Jupiter"]

    # Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create,
        global_frame_origin,
        global_frame_orientation)
    body_settings.add_empty_settings("Delfi-PQ")
    body_settings.get("Delfi-PQ").constant_mass = 0.6
    
    # Inerpolate cartesian states from SGP4 and add them as ephemeris to Delfi-PQ
    state_history_dict = cartesian_states2dict(sgp4_cartesian_states, epochs)
    delfi_ephemeris = environment_setup.ephemeris.tabulated(
        state_history_dict,
        global_frame_origin,
        global_frame_orientation
    )
    delfi_ephemeris.make_multi_arc_ephemeris = True
    body_settings.get("Delfi-PQ").ephemeris_settings = delfi_ephemeris

    # Create system of bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)


    # Create aerodynamic coefficient interface settings
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area_drag,
        [drag_coefficient, 0.0, 0.0]
    )
    # Add the aerodynamic interface to the environment
    environment_setup.add_aerodynamic_coefficient_interface(bodies, "Delfi-PQ", aero_coefficient_settings)

    # Create radiation pressure settings
    occulting_bodies_dict = dict()
    occulting_bodies_dict["Sun"] = ["Earth"]
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
        reference_area_radiation,
        radiation_pressure_coefficient,
        occulting_bodies_dict
    )

    # Add the radiation pressure interface to the environment
    environment_setup.add_radiation_pressure_target_model(
        bodies,
        "Delfi-PQ",
        radiation_pressure_settings
    )

    return bodies


def setup_propagation(bodies, initial_state, simulation_start_epoch, simulation_end_epoch, time_step, earth_harmonics):

    # Define bodies that are propagated
    bodies_to_propagate = ["Delfi-PQ"]

    # Define central bodies of propagation
    central_bodies = ["Earth"]

    # Define the accelerations acting on Delfi-PQ
    accelerations_settings = dict(
        Sun=[
            propagation_setup.acceleration.radiation_pressure(),
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Mars=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Venus=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Jupiter=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Moon=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Earth=[
            propagation_setup.acceleration.spherical_harmonic_gravity(earth_harmonics[0], earth_harmonics[1]),
            propagation_setup.acceleration.aerodynamic()
        ])

    # Create global accelerations dictionary
    acceleration_settings = {"Delfi-PQ": accelerations_settings}

    # Create acceleration model
    acceleration_model = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

    # Define list of dependent variables to save
    spherical_harmonic_terms = [(0, 0), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
    dependent_variables = [
        propagation_setup.dependent_variable.total_acceleration_norm("Delfi-PQ"),
        propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.point_mass_gravity_type, "Delfi-PQ", "Sun"),
        propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.point_mass_gravity_type, "Delfi-PQ", "Moon"),
        propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.point_mass_gravity_type, "Delfi-PQ", "Mars"),
        propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.point_mass_gravity_type, "Delfi-PQ", "Venus"),
        propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.point_mass_gravity_type, "Delfi-PQ", "Jupiter"),
        propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.spherical_harmonic_gravity_type, "Delfi-PQ", "Earth"),
        propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.aerodynamic_type, "Delfi-PQ", "Earth"),
        propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.cannonball_radiation_pressure_type, "Delfi-PQ", "Sun"),
        propagation_setup.dependent_variable.latitude("Delfi-PQ", "Earth"),
        propagation_setup.dependent_variable.longitude("Delfi-PQ", "Earth"),
        propagation_setup.dependent_variable.spherical_harmonic_terms_acceleration_norm("Delfi-PQ", "Earth", spherical_harmonic_terms)
    ]

    # Create numerical integrator settings
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
        initial_time_step=time_step,
        coefficient_set=propagation_setup.integrator.CoefficientSets.rkdp_87
    )

    # Create termination settings
    termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)

    # Create propagation settings
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_model,
        bodies_to_propagate,
        initial_state,
        simulation_start_epoch,
        integrator_settings,
        termination_condition,  # Stops propagation when the last generated state is after termination time
        output_variables=dependent_variables)

    return propagator_settings

def setup_propagation_multiarc(bodies, arcs_start_epochs, arcs_initial_states, arcs_termination_epochs, integrator_time_step, earth_harmonics):

    singlearc_propagators_settings = []
    for i, _ in enumerate(arcs_start_epochs):
        singlearc_propagator_settings = setup_propagation(
            bodies,
            arcs_initial_states[i],
            arcs_start_epochs[i],
            arcs_termination_epochs[i],
            integrator_time_step,
            earth_harmonics)
        singlearc_propagators_settings.append(singlearc_propagator_settings)

    multiarc_propagator_settings = propagation_setup.propagator.multi_arc(
        singlearc_propagators_settings,
        False) # Transfer state to next arc
    
    return multiarc_propagator_settings


def perform_propagation(bodies, propagator_settings):
    
    # Create simulation object and propagate the dynamics
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)

    # Extract the resulting state and depedent variable history and convert it to an ndarray
    states = dynamics_simulator.state_history
    states = result2array(states)
    epochs = states[:, 0]
    states = states[:, 1:]

    dependent_variables = dynamics_simulator.dependent_variable_history
    dependent_variables = result2array(dependent_variables)


    return states, epochs, dependent_variables

def setup_observation(simulation_start_epoch, simulation_end_epoch, time_step, bodies):
    

    # Create observation settings
    link_ends = dict()
    link_ends[observation.observed_body] = observation.body_origin_link_end_id("Delfi-PQ")
    link_definition = observation.LinkDefinition(link_ends)

    observation_settings = [observation.cartesian_position(link_definition)] 


    # Define epochs at which the ephemerides shall be checked
    observation_times = np.arange(simulation_start_epoch, simulation_end_epoch, time_step)

    # Create observation simulation settings
    observation_simulation_settings = observation.tabulated_simulation_settings(
        observation.position_observable_type,
        link_definition,
        observation_times,
        reference_link_end_type=observation.observed_body)
    observation_simulation_settings = [observation_simulation_settings]


    # Create observation simulator
    observation_simulator = estimation_setup.create_observation_simulators(observation_settings, bodies)

    # Get ephemeris states as ObservationCollection (ObservationCollection <=> cartesian_states)
    observation_collection = estimation.simulate_observations(
        observation_simulation_settings,
        observation_simulator,
        bodies
    )

    observation_epochs = np.array(observation_collection.concatenated_times)
    observation_positions = np.array(observation_collection.concatenated_observations)

    observation_epochs = observation_epochs[::3]
    observation_positions = observation_positions.reshape(-1, 3)
    

    return observation_settings, observation_collection, observation_positions, observation_epochs

def setup_estimation_singlearc(propagator_settings, observation_settings, bodies, estimate_constant_drag_coefficient):
    
    # Setup parameters settings to propagate the state transition matrix
    parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)

    # Add estimated parameters to the sensitivity matrix that will be propagated
    if estimate_constant_drag_coefficient:
        parameter_settings.append(estimation_setup.parameter.constant_drag_coefficient("Delfi-PQ"))

    # Create the parameters that will be estimated
    estimatable_parameters = estimation_setup.create_parameter_set(parameter_settings, bodies)

    # Create the estimator
    estimator = numerical_simulation.Estimator(
        bodies,
        estimatable_parameters,
        observation_settings,
        propagator_settings,
        True  # Integrate on creation
    )

    return estimator, estimatable_parameters

def setup_estimation_multiarc(propagator_settings, observation_settings, bodies, arcs_start_epochs, estimate_constant_drag_coefficient, estimate_arcwise_drag_coefficient, drag_arcs_start_epochs):
    
    # Setup parameters settings to propagate the state transition matrix
    parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies, arcs_start_epochs)

    # Add estimated parameters to the sensitivity matrix that will be propagated
    if estimate_constant_drag_coefficient:
        parameter_settings.append(estimation_setup.parameter.constant_drag_coefficient("Delfi-PQ"))
    if estimate_arcwise_drag_coefficient:
        parameter_settings.append(estimation_setup.parameter.arcwise_constant_drag_coefficient("Delfi-PQ", drag_arcs_start_epochs))

    # Create the parameters that will be estimated
    estimatable_parameters = estimation_setup.create_parameter_set(parameter_settings, bodies)

    # Create the estimator
    estimator = numerical_simulation.Estimator(
        bodies,
        estimatable_parameters,
        observation_settings,
        propagator_settings,
        True  # Integrate on creation
    )

    return estimator, estimatable_parameters

def perform_estimation(estimatable_parameters, estimator, observation_collection, no_iterations):

    # Save the true parameters to later analyse the error
    initial_paramaters = estimatable_parameters.parameter_vector

    # Create input object for the estimation
    convergence_checker = estimation.estimation_convergence_checker(maximum_iterations=no_iterations)
    estimation_input = estimation.EstimationInput(
        observation_collection,
        convergence_checker=convergence_checker,
        apply_final_parameter_correction=True)  # Default True

    # Set methodological options
    estimation_input.define_estimation_settings(
        reintegrate_variational_equations=True,
        save_state_history_per_iteration=True)

    # Perform the estimation
    estimation_output = estimator.perform_estimation(estimation_input)  # It prints current residuals as rms

    return estimation_output, initial_paramaters

def preform_covariance_analysis(estimator, observation_collection):

    # Create input object for covariance analysis
    covariance_input = estimation.CovarianceAnalysisInput(
        observation_collection)

    # Set methodological options
    covariance_input.define_covariance_settings(
        reintegrate_variational_equations=False,
        print_output_to_terminal=True)

    # Perform the covariance analysis
    covariance_output = estimator.compute_covariance(covariance_input)

    return covariance_output

def process_results_singlearc_estimation(estimation_output, estimatable_parameters, sgp4_cartesian_states):

    # Get estimated parameters
    estimated_parameters = estimatable_parameters.parameter_vector

    # Get residuals
    residual_history = estimation_output.residual_history
    
    final_residuals = estimation_output.final_residuals
    final_residuals = final_residuals.reshape(-1, 3)

    rms_final_residuals = np.sqrt(np.average(np.square(final_residuals)))

    # Transfrom residuals from cartesian to TNW
    tnw_final_residuals = []
    for i, cartesian_state in enumerate(sgp4_cartesian_states):
        rotation_matrix = frame_conversion.inertial_to_tnw_rotation_matrix(cartesian_state)
        tnw_final_residual = rotation_matrix.dot(final_residuals[i])
        tnw_final_residuals.append(tnw_final_residual)
    tnw_final_residuals = np.array(tnw_final_residuals)

    # Get state and dependent variable history
    simulator_object = estimation_output.simulation_results_per_iteration[-1]
    
    estimated_cartesian_states_dict = simulator_object.dynamics_results.state_history
    estimated_cartesian_states = np.array(list(estimated_cartesian_states_dict.values()))
    estimated_epochs = np.array(list(estimated_cartesian_states_dict.keys()))

    dependent_variables_dict = simulator_object.dynamics_results.dependent_variable_history
    dependent_variables = np.array(list(dependent_variables_dict.values()))
    
    return estimated_parameters, final_residuals, tnw_final_residuals, rms_final_residuals, estimated_cartesian_states, estimated_epochs, dependent_variables

"""
    Note about all types of epochs:
     - len(sgp4 epochs) = len(observation epochs) = len(estimated epochs) - 1 = len(propagation epochs) - 1
"""

def process_results_multiarc_estimation(estimation_output, estimatable_parameters, sgp4_cartesian_states):

    # Get estimated parameters
    estimated_parameters = estimatable_parameters.parameter_vector

    # Get residuals
    residual_history = estimation_output.residual_history
    
    final_residuals = estimation_output.final_residuals
    final_residuals = final_residuals.reshape(-1, 3)

    rms_final_residuals = np.sqrt(np.average(np.square(final_residuals)))

    # Transfrom residuals from cartesian to TNW
    tnw_final_residuals = []
    for i, cartesian_state in enumerate(sgp4_cartesian_states):
        rotation_matrix = frame_conversion.inertial_to_tnw_rotation_matrix(cartesian_state)
        tnw_final_residual = rotation_matrix.dot(final_residuals[i])
        tnw_final_residuals.append(tnw_final_residual)
    tnw_final_residuals = np.array(tnw_final_residuals)

    # Get states and dependent variable history
    estimated_cartesian_states = []
    estimated_epochs = []
    dependent_variables = []
    
    simulator_object = estimation_output.simulation_results_per_iteration[-1]
    
    for i, _ in enumerate(simulator_object.single_arc_results):
        
        singlearc_simulator_object = simulator_object.single_arc_results[i]
        
        estimated_cartesian_states_dict = singlearc_simulator_object.dynamics_results.state_history
        estimated_cartesian_states += list(estimated_cartesian_states_dict.values())
        estimated_epochs += list(estimated_cartesian_states_dict.keys())
        
        dependent_variables_dict = singlearc_simulator_object.dynamics_results.dependent_variable_history
        dependent_variables += list(dependent_variables_dict.values())

    
    estimated_cartesian_states = np.array(estimated_cartesian_states)
    estimated_epochs = np.array(estimated_epochs)
    dependent_variables = np.array(dependent_variables)
    
    # Get errors at arcs overlaps
    overlap_errors = []
    for i, epoch in enumerate(estimated_epochs):
        if i == 0:
            continue
        if (estimated_epochs[i] == estimated_epochs[i-1]):
            error = estimated_cartesian_states[i] - estimated_cartesian_states[i-1]
            overlap_errors.append(error)
    overlap_errors = np.array(overlap_errors)

    overlap_errors = overlap_errors[:, :3]  # Use only positions
    rms_overlap_errors = np.sqrt(np.average(np.square(overlap_errors)))
    
    
    return estimated_parameters, final_residuals, tnw_final_residuals, rms_final_residuals, estimated_cartesian_states, estimated_epochs, dependent_variables, overlap_errors, rms_overlap_errors

"""
    Note about all types of epochs:
     - len(sgp4 epochs) = len(observation epochs) = len(estimated epochs) - (len(arcs) - 1)
     - all arrays are the ranges [simulation_start_epoch, simulation_end epoch)
"""

def process_results_singlearc_prediction(sgp4_cartesian_states_estimation, sgp4_cartesian_states_prediction, propagation_cartesian_states_estimation, propagation_cartesian_states_prediction, conventional_cartesian_states):

    # Calculate final residuals
    cartesian_final_residuals_estimation = sgp4_cartesian_states_estimation[:,:3] - propagation_cartesian_states_estimation[:,:3]
    cartesian_final_residuals_prediction = sgp4_cartesian_states_prediction[:,:3] - propagation_cartesian_states_prediction[:,:3]
    cartesian_final_residuals_conventional = sgp4_cartesian_states_prediction[:,:3] - conventional_cartesian_states[:,:3]

    # Calculate RMS of final residuals
    rms_final_residuals_estimation = np.sqrt(np.average(np.square(cartesian_final_residuals_estimation)))
    rms_final_residuals_prediction = np.sqrt(np.average(np.square(cartesian_final_residuals_prediction)))
    rms_final_residuals_conventional = np.sqrt(np.average(np.square(cartesian_final_residuals_conventional)))


    # Transfrom residuals from cartesian to TNW
    tnw_final_residuals_estimation = []
    for i, cartesian_state in enumerate(sgp4_cartesian_states_estimation):
        rotation_matrix = frame_conversion.inertial_to_tnw_rotation_matrix(cartesian_state)
        tnw_final_residual = rotation_matrix.dot(cartesian_final_residuals_estimation[i])
        tnw_final_residuals_estimation.append(tnw_final_residual)
    tnw_final_residuals_estimation = np.array(tnw_final_residuals_estimation)
    
    tnw_final_residuals_prediction = []
    for i, cartesian_state in enumerate(sgp4_cartesian_states_prediction):
        rotation_matrix = frame_conversion.inertial_to_tnw_rotation_matrix(cartesian_state)
        tnw_final_residual = rotation_matrix.dot(cartesian_final_residuals_prediction[i])
        tnw_final_residuals_prediction.append(tnw_final_residual)
    tnw_final_residuals_prediction = np.array(tnw_final_residuals_prediction)

    tnw_final_residuals_conventional = []
    for i, cartesian_state in enumerate(sgp4_cartesian_states_prediction):
        rotation_matrix = frame_conversion.inertial_to_tnw_rotation_matrix(cartesian_state)
        tnw_final_residual = rotation_matrix.dot(cartesian_final_residuals_conventional[i])
        tnw_final_residuals_conventional.append(tnw_final_residual)
    tnw_final_residuals_conventional = np.array(tnw_final_residuals_conventional)

    return cartesian_final_residuals_estimation, cartesian_final_residuals_prediction, cartesian_final_residuals_conventional, tnw_final_residuals_estimation, tnw_final_residuals_prediction, tnw_final_residuals_conventional, rms_final_residuals_estimation, rms_final_residuals_prediction, rms_final_residuals_conventional

def plot_residuals_estimation(tnw_final_residuals, observation_epochs, tles_batch_epochs, show_tles_epochs):

    plt.figure(figsize=(13,3), dpi=100)

    plt.scatter((observation_epochs - observation_epochs[0])/3600/24, tnw_final_residuals[:, 0]/1000, color="blue", marker='.', label="t-residual")  
    plt.scatter((observation_epochs - observation_epochs[1])/3600/24, tnw_final_residuals[:, 1]/1000, color="green", marker='.', label="n-residual")
    plt.scatter((observation_epochs - observation_epochs[2])/3600/24, tnw_final_residuals[:, 2]/1000, color="red", marker='.', label="w-residual")

    if show_tles_epochs:
        for i, tle_epoch in enumerate((tles_batch_epochs - observation_epochs[0])/3600/24):
            if i == 1:
                plt.axvline(x=tle_epoch, color="yellow", label="TLE epoch")
            else:
                plt.axvline(x=tle_epoch, color="yellow")

    plt.ylabel("Residuals [km]")
    plt.xlabel("Epochs [days]")
    plt.grid()
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

    """
    Note about residuals:
    Residuals are defined as true_position - estimated_position
    """

def plot_residuals_multiarc_estimation(tnw_final_residuals, observation_epochs, tles_batch_epochs, show_tles_epochs):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 6), dpi=100)

    if show_tles_epochs:
        for i, tle_epoch in enumerate((tles_batch_epochs - observation_epochs[0])/3600/24):
            if i == 1:
                ax1.axvline(x=tle_epoch, color="yellow", label="TLE epochs", linewidth=0.75)
            else:
                ax1.axvline(x=tle_epoch, color="yellow", linewidth=0.75)
    ax1.scatter((observation_epochs - observation_epochs[0])/3600/24, tnw_final_residuals[:, 0]/1000, color="blue", marker='.', s=20)
    ax1.set_xticks(range(0, 11, 1))
    ax1.set_yticks(range(-4, 5, 2))
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.set_ylabel("t-residuals [km]", fontsize=15)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(-4, 4)
    ax1.grid()

    if show_tles_epochs:
        for i, tle_epoch in enumerate((tles_batch_epochs - observation_epochs[0])/3600/24):
            if i == 1:
                ax2.axvline(x=tle_epoch, color="yellow", label="TLE epoch", linewidth=0.75)
            else:
                ax2.axvline(x=tle_epoch, color="yellow", linewidth=0.75)
    ax2.scatter((observation_epochs - observation_epochs[1])/3600/24, tnw_final_residuals[:, 1]/1000, color="green", marker='.', label="n-residual", s=20)
    ax2.set_xticks(range(0, 11, 1))
    ax2.set_yticks(range(-4, 5, 2))
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_ylabel("n-residuals [km]", fontsize=15)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-4, 4)
    ax2.grid()

    if show_tles_epochs:
        for i, tle_epoch in enumerate((tles_batch_epochs - observation_epochs[0])/3600/24):
            if i == 1:
                ax3.axvline(x=tle_epoch, color="yellow", label="TLE epochs", linewidth=0.75)
            else:
                ax3.axvline(x=tle_epoch, color="yellow", linewidth=0.75)
    ax3.scatter((observation_epochs - observation_epochs[2])/3600/24, tnw_final_residuals[:, 2]/1000, color="red", marker='.', s=20)
    ax3.set_xticks(range(0, 11, 1))
    ax3.set_yticks(range(-4, 5, 2))
    ax3.tick_params(axis='both', which='major', labelsize=15)
    ax3.set_ylabel("w-residuals [km]", fontsize=15)
    ax3.set_xlabel("Epochs [days]", fontsize=15)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(-4, 4)
    ax3.legend(loc='upper left', fontsize=15)
    ax3.grid()

    plt.tight_layout()
    plt.show()

def plot_residuals_singlearc_prediction(tnw_final_residuals, tnw_final_residuals_conventional, conventional_epochs, observation_epochs, tles_batch_epochs, prediction_start_epoch, show_tles_epochs):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 6), dpi=100)

    if show_tles_epochs:
        for i, tle_epoch in enumerate((tles_batch_epochs - prediction_start_epoch)/3600/24):
            if i == 1:
                ax1.axvline(x=tle_epoch, color="yellow", label="TLE epochs", linewidth=0.75)
            else:
                ax1.axvline(x=tle_epoch, color="yellow", linewidth=0.75)  
    ax1.scatter((observation_epochs - prediction_start_epoch)/3600/24, tnw_final_residuals[:, 0]/1000, color="blue", marker='.', label="Model Prediction", s=20)
    ax1.scatter((conventional_epochs - prediction_start_epoch)/3600/24, tnw_final_residuals_conventional[:, 0]/1000, color="black", marker='.', label="SGP4 Prediction", s=20)
    ax1.set_xticks(range(-2, 5))
    ax1.set_yticks(range(-30, 11, 10))
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.set_xlim(-2, 4)
    ax1.set_ylim(-35, 10)
    ax1.set_ylabel("t-residuals [km]", fontsize=15)
    ax1.grid()
    ax1.legend(fontsize=10)

    if show_tles_epochs:
        for i, tle_epoch in enumerate((tles_batch_epochs - prediction_start_epoch)/3600/24):
            if i == 1:
                ax2.axvline(x=tle_epoch, color="yellow", label="TLE epochs", linewidth=0.75)
            else:
                ax2.axvline(x=tle_epoch, color="yellow", linewidth=0.75)  
    ax2.scatter((observation_epochs - prediction_start_epoch)/3600/24, tnw_final_residuals[:, 1]/1000, color="green", marker='.', label="Model Prediction", s=20)
    ax2.scatter((conventional_epochs - prediction_start_epoch)/3600/24, tnw_final_residuals_conventional[:, 1]/1000, color="#808080", marker='.', label="SGP4 Prediction", s=20)
    ax2.set_xticks(range(-2, 5))
    ax2.set_yticks(np.arange(-1, 1.5, 0.5))
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_xlim(-2, 4)
    ax2.set_ylim(-1, 1)
    ax2.set_ylabel("n-residuals [km]", fontsize=15)
    ax2.grid()
    ax2.legend(fontsize=10)

    if show_tles_epochs:
        for i, tle_epoch in enumerate((tles_batch_epochs - prediction_start_epoch)/3600/24):
            if i == 1:
                ax3.axvline(x=tle_epoch, color="yellow", label="TLE epochs", linewidth=0.75)
            else:
                ax3.axvline(x=tle_epoch, color="yellow", linewidth=0.75)  
    ax3.scatter((observation_epochs - prediction_start_epoch)/3600/24, tnw_final_residuals[:, 2]/1000, color="red", marker='.', label="Model Prediction", s=20)
    ax3.scatter((conventional_epochs - prediction_start_epoch)/3600/24, tnw_final_residuals_conventional[:, 2]/1000, color="#A9A9A9", marker='.', label="SGP4 Prediction", s=20)
    ax3.set_xticks(range(-2, 5))
    ax3.set_yticks(np.arange(-1, 1.5, 0.5))
    ax3.tick_params(axis='both', which='major', labelsize=15)
    ax3.set_xlim(-2, 4)
    ax3.set_ylim(-1, 1)
    ax3.set_ylabel("w-residuals [km]", fontsize=15)
    ax3.set_xlabel("Epochs [day]", fontsize=15)
    ax3.grid()
    ax3.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()

def plot_histogram(tnw_final_residuals):

    # Define bins
    min_edge = np.min(tnw_final_residuals) / 1000
    max_edge = np.max(tnw_final_residuals) / 1000
    bin_width = 0.05
    bins = np.arange(min_edge, max_edge + bin_width, bin_width)

    # Create figure
    plt.figure(figsize=(9, 6), dpi=100)
    plt.hist(tnw_final_residuals[:, 0]/1000, bins=bins, color="blue", alpha=0.5, label="t-residuals")
    plt.hist(tnw_final_residuals[:, 2]/1000, bins=bins, color="red", alpha=0.5, label="w-residuals")
    plt.hist(tnw_final_residuals[:, 1]/1000, bins=bins, color="green", alpha=0.5, label="n-residuals")
    
    # Set figure
    plt.xlabel('Residuals [km]', fontsize=15)
    plt.ylabel('Occurrences [-]', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(-3, 3)
    plt.ylim(0, 1600)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.grid()
    plt.show()    

def plot_3D_cartesian_states(epochs, cartesian_states):

    # Define a 3D figure using pyplot
    fig = plt.figure(figsize=(6,6), dpi=125)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'Delfi-PQ observations around Earth')

    # Plot the positional state history
    delfi_scatter = ax.scatter(cartesian_states[:, 0], cartesian_states[:, 1], cartesian_states[:, 2], c=epochs, cmap='viridis', label="Delfi-PQ", marker=".")
    earth_scatter = ax.scatter(0.0, 0.0, 0.0, label="Earth", marker='o', color='green')

    # Add the colorbar for reference
    cbar = fig.colorbar(delfi_scatter, ax=ax, label='Epoch')

    # Add the legend and labels, then show the plot
    ax.legend()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    plt.show()

def plot_keplerian_elements(epochs, cartesian_states):

    # Convert cartesian states to keplerian
    keplerian_states = []
    for i, cartesian_state in enumerate(cartesian_states):
        keplerian_states.append(element_conversion.cartesian_to_keplerian(cartesian_state, spice.get_body_gravitational_parameter("Earth")))
    keplerian_states = np.array(keplerian_states)

    # Plot Kepler elements as a function of time
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(9, 12))
    fig.suptitle('Evolution of Kepler elements')

    # Semi-major Axis
    semi_major_axis = keplerian_states[:, 0] / 1e3
    ax1.plot(epochs, semi_major_axis)
    ax1.set_ylabel('Semi-major axis [km]')

    # Eccentricity
    eccentricity = keplerian_states[:,1]
    ax2.plot(epochs, eccentricity)
    ax2.set_ylabel('Eccentricity [-]')

    # Inclination
    inclination = np.rad2deg(keplerian_states[:,2])
    ax3.plot(epochs, inclination)
    ax3.set_ylabel('Inclination [deg]')

    # Argument of Periapsis
    argument_of_periapsis = np.rad2deg(keplerian_states[:,3])
    ax4.plot(epochs, argument_of_periapsis)
    ax4.set_ylabel('Argument of Periapsis [deg]')

    # Right Ascension of the Ascending Node
    raan = np.rad2deg(keplerian_states[:,4])
    ax5.plot(epochs, raan)
    ax5.set_ylabel('RAAN [deg]')

    # True Anomaly
    true_anomaly = np.rad2deg(keplerian_states[:,5])
    ax6.scatter(epochs, true_anomaly, s=1)
    ax6.set_ylabel('True Anomaly [deg]')
    ax6.set_yticks(np.arange(0, 361, step=60))

    for ax in fig.get_axes():
        ax.set_xlabel('Epochs')
        ax.set_xlim([min(epochs), max(epochs)])
        ax.grid()
    plt.tight_layout()
    plt.show()

def plot_accelerations(dependent_variables, propagation_epochs):

    # Plot acelerations
    plt.figure(figsize=(9, 6))

    acceleration_norm_pm_sun = dependent_variables[:,1]
    plt.plot((propagation_epochs - propagation_epochs[0])/3600/24, acceleration_norm_pm_sun, label='PM Sun')

    acceleration_norm_pm_moon = dependent_variables[:,2]
    plt.plot((propagation_epochs - propagation_epochs[0])/3600/24, acceleration_norm_pm_moon, label='PM Moon')

    acceleration_norm_pm_mars = dependent_variables[:,3]
    plt.plot((propagation_epochs - propagation_epochs[0])/3600/24, acceleration_norm_pm_mars, label='PM Mars')

    acceleration_norm_pm_venus = dependent_variables[:,4]
    plt.plot((propagation_epochs - propagation_epochs[0])/3600/24, acceleration_norm_pm_venus, label='PM Venus')

    acceleration_norm_pm_jupiter = dependent_variables[:,5]
    plt.plot((propagation_epochs - propagation_epochs[0])/3600/24, acceleration_norm_pm_jupiter, label='PM Jupiter')

    acceleration_norm_sh_earth = dependent_variables[:,6]
    plt.plot((propagation_epochs - propagation_epochs[0])/3600/24, acceleration_norm_sh_earth, label='SH Earth')

    acceleration_norm_aero_earth = dependent_variables[:,7]
    plt.plot((propagation_epochs - propagation_epochs[0])/3600/24, acceleration_norm_aero_earth, label='Aerodynamic Earth')

    acceleration_norm_rp_sun = dependent_variables[:,8]
    plt.plot((propagation_epochs - propagation_epochs[0])/3600/24, acceleration_norm_rp_sun, label='Radiation Pressure Sun')

    # spherical_harmonic_terms = [(0, 0,) (), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3)]
    spherical_harmonic_terms = [(0, 0), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
    for i, eh in enumerate(spherical_harmonic_terms):
        acceleration_norm = dependent_variables[:,11+i]
        plt.plot((propagation_epochs - propagation_epochs[0])/3600/24, acceleration_norm, label=f'C_{eh[0]},{eh[1]}', linestyle="dotted")

    # plt.xlim([min(propagation_epochs/3600), max(propagation_epochs)])
    plt.xlabel('Time [days]')
    plt.ylabel('Acceleration Norm [m/s$^2$]')

    # plt.legend(bbox_to_anchor=(1.005, 1))
    # plt.suptitle("Accelerations norms on Delfi-PQ, distinguished by type and origin, over the course of propagation.")
    plt.legend()
    plt.yscale('log')
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(covariance_output):
    
    # Create figure
    plt.figure(figsize=(9, 7))
    plt.imshow(np.abs(covariance_output.correlations), aspect='equal', interpolation='none')
    
    # Set figure
    cbar = plt.colorbar()
    cbar.set_ticks(np.arange(0.0, 1.1, 0.1))
    cbar.ax.tick_params(labelsize=15)
    plt.xlabel("Estimated Parameter Index", fontsize=15)
    plt.ylabel("Estimated Parameter Index", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.show()

def save_to_csv(new_row, file_name, include_head_row):
    
    if include_head_row:
        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(list(new_row.keys()))
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list(new_row.values()))
