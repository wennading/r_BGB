# load CAPTAIN and dependencies
import copy
import os, csv
import glob
import sys
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, precision=3)
import captain as cn
from captain.utilities import empirical_data_parser as cn_util
from captain.utilities import sdm_utils as sdm_utils
from captain.biodivsim import SimGrid as cn_sim_grid
from captain.utilities.misc import parse_str, get_nn_params_from_file
import sparse
import argparse
import configparser

p = argparse.ArgumentParser()
p.add_argument('config_file', type=str, help='config file')
args = p.parse_args()
config_file = args.config_file


config = configparser.ConfigParser()
config.read(config_file)
SEED = parse_str(config["general"]["seed"])

#-------- SETTINGS ---------#

# policy settings (objectives and reward)
ext_risk_class = cn.ExtinctioRiskProtectedRangeFuture # cn.ExtinctioRiskProtectedRange
budget = parse_str(config["policy"]["budget"])  # if budget = None: np.sum(costs) * protection_fraction
reward = "pareto_mlt"

r_w = parse_str(config["policy"]["reward_weights"])
r_w = np.round(r_w / np.sum(r_w), 2)
reward_weights = {'carbon': r_w[0], 'species_risk': r_w[1], 'cost': r_w[2]}


# paths, output files
data_wd = config["files"]["data_dir"]
models_wd = config["files"]["models_dir"]
env_data_wd = os.path.join(data_wd, config["files"]["env_data_dir"])

results_wd = os.path.join(data_wd, config["files"]["result_dir"] + "_" + str(SEED))
try: os.mkdir(results_wd)
except FileExistsError: pass


#-------- LOAD DATA ---------#

# load present and future SDMs
sdms, species_names = sdm_utils.get_data_from_list(
    os.path.join(data_wd, config["files"]["sdm_data_dir"]),
    tag="/*.tif", max_species_cutoff=None)

sdms_f, species_names_f = sdm_utils.get_data_from_list(
    os.path.join(data_wd, config["files"]["future_sdm_data_dir"]),
    tag="/*", max_species_cutoff=None, zero_to_nan=True)

# load species traits (migration, sensitivity, growth, conservation status)
trait_tbl = pd.read_csv(os.path.join(data_wd, config["files"]["trait_tbl_file"]))
empirical_sensitivity = np.array(trait_tbl['sensitivity_disturbance'])
growth_rates = np.array(trait_tbl['growth_rate'])
conservation_status = 5 - np.array(trait_tbl['conservation_status'])
species_carbon = np.array(trait_tbl['biomass'])

# load disturbance and cost
disturbance_layer, _ = sdm_utils.load_map(os.path.join(env_data_wd, config["files"]["disturbance_file"]))
cost_layer, _ = sdm_utils.load_map(os.path.join(env_data_wd, config["files"]["cost_file"]))
if parse_str(config["files"]["selective_disturbance_file"]) is not None:
    selective_disturbance_layer, _ = sdm_utils.load_map(
        os.path.join(env_data_wd, config["files"]["selective_disturbance_file"]))
else:
    selective_disturbance_layer = None


#-------- ENVIRONMENT SETUP ---------#

# create env
sdms[sdms > parse_str(config["env_settings"]["full_suitability"])] = 1
sdms_f[sdms_f > parse_str(config["env_settings"]["full_suitability"])] = 1

# make it a graph without gaps
suitability = cn_util.get_habitat_suitability(sdms, parse_str(config["env_settings"]["prob_threshold"]))
original_grid_shape, reference_grid_pu, reference_grid_pu_nan, xy_coords, graph_coords = sdm_utils.get_graph(suitability)
max_species_richness = np.nanmax( np.nansum(suitability, axis=0))
n_species = sdms.shape[0]

# set up parameters
if parse_str(config["species_settings"]["use_empirical_dispersal_rates"]):
    emp_dispersal_rates = np.array(trait_tbl['dispersal_ability'])
    dispersal_rates = emp_dispersal_rates / np.nanmean(
        emp_dispersal_rates) * parse_str(config["species_settings"]["mean_dispersal_rate"])

elif parse_str(config["species_settings"]["variable_dispersal_rates"]):
    dispersal_rates = np.exp(np.linspace(-2, 2, None)) #[::-1]
    dispersal_rates = dispersal_rates / np.mean(
        dispersal_rates) * parse_str(config["species_settings"]["mean_dispersal_rate"])

else:
    dispersal_rates = parse_str(config["species_settings"]["mean_dispersal_rate"])


# dispersal threshold (in number of cells)
fname_sparse = "disp_probs_sp%s_th%s_%s.npz" % (n_species, config["species_settings"]["dispersal_threshold"], SEED)
try:
    dispersal_probs_sparse = sparse.load_npz(os.path.join(data_wd, fname_sparse))
except:
    dispersal_probs = cn_sim_grid.dispersalDistancesThresholdCoord(
        length=graph_coords.shape[1], lambda_0=1, lat=graph_coords[1], lon=graph_coords[0],
        threshold=parse_str(config["species_settings"]["dispersal_threshold"]))
    dispersal_probs_sparse = sparse.COO(dispersal_probs)
    sparse.save_npz(os.path.join(data_wd, fname_sparse), dispersal_probs_sparse)



# graph transform
# graph SDMs
graph_sdms, n_pus, grid_length = sdm_utils.grid_to_graph(sdms, reference_grid_pu)
graph_suitability = cn_util.get_habitat_suitability(
    graph_sdms, parse_str(config["env_settings"]["prob_threshold"]),
    integer=parse_str(config["env_settings"]["round_habitat_suitability"]))
graph_sdms_f , _, __ = sdm_utils.grid_to_graph(sdms_f, reference_grid_pu)
graph_future_suitability = cn_util.get_habitat_suitability(
    graph_sdms_f, parse_str(config["env_settings"]["prob_threshold"]),
    integer=parse_str(config["env_settings"]["round_habitat_suitability"]))
# how much the present suitability has to change per step to reach "future_suitability" in STEPS
# can be multiplied by e.g. 0.5 to go only halfway to "future_suitability" in STEPS
if parse_str(config["general"]["use_future_sdms"]):
    t = parse_str(config["general"]["time_to_future_suitability"])
    delta = (graph_future_suitability - graph_suitability) * 1 / t
    delta_suitability_per_step = {'delta': delta,
                                  'threshold': parse_str(config["env_settings"]["prob_threshold"])}
else:
    delta_suitability_per_step = {'delta': 0, 'threshold': parse_str(config["env_settings"]["prob_threshold"])}
    graph_sdms_f = graph_sdms + 0
    graph_future_suitability = graph_suitability + 0


if parse_str(config["policy"]["add_to_existing_protected_areas"]):
    mpas_layer_name = parse_str(config["files"]["protection_file"])
    if parse_str(config["files"]["protection_file"]) is not None:
        existing_mpas, _ = sdm_utils.load_map(os.path.join(env_data_wd, parse_str(config["files"]["protection_file"])))
        disturbance_in_protect = (existing_mpas > 0) * disturbance_layer
        # reset disturbance in MPAs to 0
        disturbance_layer[existing_mpas > 0] = 1 - parse_str(config["policy"]["max_protection_level"])
        graph_protection_matrix, _, __ = sdm_utils.grid_to_graph(existing_mpas > 0, reference_grid_pu, n_pus, nan_to_zero=True)
        existing_protected = sdm_utils.graph_to_grid(graph_protection_matrix, reference_grid_pu, zero_to_nan=True)
        existing_protection_fraction = np.sum(graph_protection_matrix[graph_protection_matrix > 0]) / n_pus
    else:
        sys.exit("protection_file not specified")
else:
    graph_protection_matrix = None
    existing_protection_fraction = 0
    existing_mpas = None

graph_disturbance, _, __ = sdm_utils.grid_to_graph(disturbance_layer, reference_grid_pu, n_pus, nan_to_zero=True)
if selective_disturbance_layer is not None:
    graph_selective_disturbance, _, __ = sdm_utils.grid_to_graph(selective_disturbance_layer, reference_grid_pu, n_pus,
                                                             nan_to_zero=True)
else:
    graph_selective_disturbance = graph_disturbance

if parse_str(config["general"]["use_cost"]):
    graph_cost, _, __ = sdm_utils.grid_to_graph(cost_layer, reference_grid_pu, n_pus, nan_to_zero=True)
else:
    graph_cost, _, __ = sdm_utils.grid_to_graph(np.ones(reference_grid_pu.shape) * 0.01, reference_grid_pu, n_pus, nan_to_zero=True)

protection_target = parse_str(config["general"]["protection_target"])
if budget is None:
    # budget set as a function of costs and target protection fraction
    if graph_protection_matrix is None:
        budget = np.sum(graph_cost) * protection_target
    else:
        budget = np.sum(graph_cost) * (protection_target - existing_protection_fraction)

if graph_protection_matrix is not None:
    protection_steps = protection_target * n_pus - int(np.sum(graph_protection_matrix))
else:
    protection_steps = protection_target * n_pus

protection_actions_per_step = int(np.floor((protection_steps / (
        parse_str(config["general"]["protection_time_steps"]) - 1)) / 100) * 100)
print("Protection actions per step:",protection_actions_per_step)



#-------- ANALYSIS ---------#



# env_init GENERATOR
def generate_init_env(config, seed):
    size_protection_unit = np.array([1, 1])  # only 1x1 makes sense in graph-grid!
    r_seeds_dict = {
        'rnd_sensitivity': 1 * seed,
        'rnd_growth': 2 * seed,
        'rnd_config_policy': 3 * seed,
        'rnd_k_species': 4 * seed,
        'rnd_dispersal': 5 * seed
    }

    if parse_str(config["env_settings"]["use_K_species"]):
        # species-specific carrying capacities
        rs_k = cn.get_rnd_gen(r_seeds_dict['rnd_k_species'])
        if parse_str(config["env_settings"]["species_k_rnd_uniform"]):
            K_species = 2 + rs_k.random(n_species) * 100
        else:
            K_species = 2 + rs_k.beta(0.6, 1.2, n_species) * 100
        h3d = K_species[:, np.newaxis, np.newaxis] * graph_suitability
    else:
        sys.exit("hd3 object not initialized (option use_K_species = False not available)")

    # create SimGrid object: initGrid with empirical 3d histogram
    stateInitializer = cn.EmpiricalStateInit(h3d)

    max_K_cell = np.max(h3d) * 2 # max K per cell set > than highest based on K species (only K species matters)
    K_max = (h3d.sum((0)) > 0) * max_K_cell

    # initialize grid to reach carrying capacity: no disturbance, no sensitivity
    env2d = cn.BioDivEnv(budget=0.1,
                         gridInitializer=stateInitializer,
                         length=grid_length,
                         n_species=n_species,
                         K_max=K_max,
                         disturbance_sensitivity=np.zeros(n_species),
                         disturbanceGenerator=cn.FixedEmpiricalDisturbanceGenerator(0),
                         # to fast forward to w 0 disturbance
                         dispersal_rate=5,
                         growth_rate=[2],
                         resolution=size_protection_unit,
                         habitat_suitability=graph_suitability,
                         cost_pu=graph_cost,
                         precomputed_dispersal_probs=dispersal_probs_sparse,
                         use_small_grid=True,
                         K_species=K_species
                         )

    # evolve system to reach K_max
    env2d.set_calc_reward(False)
    env2d.fast_forward(parse_str(config["env_settings"]["steps_fast_fw"]),
                       disturbance_effect_multiplier=0, verbose=False, skip_dispersal=False)

    if parse_str(config["general"]["do_plots"]):
        species_richness_init = sdm_utils.graph_to_grid(env2d.bioDivGrid.speciesPerCell(), reference_grid_pu)
        cn_util.plot_map(species_richness_init, z=reference_grid_pu_nan, nan_to_zero=False, vmax=max_species_richness,
                         cmap="YlGnBu", show=False, title="Species richness (Natural state - FWD)",
                         outfile=os.path.join(results_wd, "species_richness_natural_FWD.png"), dpi=250)

        pop_density = sdm_utils.graph_to_grid(env2d.bioDivGrid.individualsPerCell(), reference_grid_pu)
        cn_util.plot_map(pop_density, z=reference_grid_pu_nan, nan_to_zero=False,
                         cmap="Greens", show=False, title="Population density (Natural state - FWD)",
                         outfile=os.path.join(results_wd, "population_density_natural_FWD.png"), dpi=250)

    if parse_str(config["files"]["plot_env_states"]):
        cn.plot_env_state(env2d, wd=results_wd, species_list=[])

    # make empirically rare / threatened species sensitive
    sensitivities = {
        'disturbance_sensitivity': empirical_sensitivity,
        'selective_sensitivity': empirical_sensitivity,
        'climate_sensitivity': np.zeros(n_species)
    }

    # set extinction risks
    ext_risk_obj = ext_risk_class(
        natural_state=env2d.grid_obj_previous,
        current_state=env2d.bioDivGrid,
        starting_rl_status=conservation_status,
        evolve_status=parse_str(config["extinction_risk"]["evolve_rl_status"]),
        # relative_pop_thresholds=relative_pop_thresholds,
        epsilon=0.5,
        # eps=1: last change, eps=0.5: rolling average, eps<0.5: longer legacy of long-term change
        sufficient_protection=0.5,
        pop_decrease_threshold=parse_str(config["extinction_risk"]["pop_decrease_threshold"]),
        min_individuals_cell=parse_str(config["extinction_risk"]["min_individuals_cell"]),
        relative_protected_range_thresholds=parse_str(config["extinction_risk"]["relative_protected_range_thresholds"]),
        risk_weights=parse_str(config["extinction_risk"]["risk_weights"]),
        min_protected_cells=parse_str(config["extinction_risk"]["min_protected_cells"]))


    d = np.einsum('sxy->xy', h3d)
    protection_steps = np.round(
        (d[d > 1].size * parse_str(config["general"]["protection_target"])) / (
                size_protection_unit[0] * size_protection_unit[1])).astype(int)
    print("Protection steps:", protection_steps)

    mask_disturbance = (K_max > 0).astype(int)
    disturbanceGenerator = cn.FixedEmpiricalDisturbanceGenerator(0)
    selective_disturbanceGenerator = cn.FixedEmpiricalDisturbanceGenerator(0)
    init_disturbance = disturbanceGenerator.updateDisturbance(
        graph_disturbance * mask_disturbance * parse_str(config["env_settings"]["zero_disturbance"]))
    init_selective_disturbance = selective_disturbanceGenerator.updateDisturbance(
        graph_selective_disturbance * mask_disturbance * parse_str(config["env_settings"]["zero_selective_disturbance"]))

    # config simulation
    config = cn.ConfigOptimPolicy(rnd_seed=r_seeds_dict['rnd_config_policy'],
                                  obsMode=1,
                                  # 0: random, 1: full monitor, 2: citizen-science, 3: one-time, 4: value, 5: area
                                  feature_update_per_step=True,
                                  steps=protection_steps,
                                  simulations=1,
                                  observePolicy=1,  # 0: NO-OBSERVE-UPDATE 1: ORACLE 2: PROTECTATONCE
                                  disturbance=-1,
                                  degrade_steps=5,
                                  initial_disturbance=init_disturbance,  # set initial disturbance matrix
                                  initial_selective_disturbance=init_selective_disturbance,
                                  initial_protection_matrix=graph_protection_matrix,
                                  edge_effect=0,
                                  protection_cost=1,
                                  n_nodes=[2, 2],
                                  random_sim=0,
                                  # "0: fixed (replicable) simulations; 1: random; 2: fixed training, seq pickle"
                                  rewardMode=reward,
                                  obs_error=0,  # "Amount of error in species counts (feature extraction)"
                                  use_true_natural_state=True,
                                  resolution=size_protection_unit,
                                  grid_size=env2d.length,
                                  budget=budget,
                                  dispersal_rate=dispersal_rates,
                                  growth_rates=growth_rates,  # can be 1 values (list of 1 item) or or one value per species
                                  use_climate=0,  # "0: no climate change, 1: climate change, 2: climate disturbance,
                                  # 3: climate change + random variation"
                                  rnd_alpha=0,
                                  # (st.dev of sp.-specific fluctuation in mortality (if 'by_species' ==1 in SimpleGrid)
                                  outfile=config["files"]["outfile"] + config["general"]["seed"] + ".log",
                                  # model settings
                                  trained_model=None,
                                  temperature=1,
                                  deterministic_policy=1,  # 0: random policy (altered by temperature);
                                  # 1: deterministic policy (overrides temperature)
                                  sp_threshold_feature_extraction=0.001,
                                  start_protecting=1,
                                  plot_sim=False,
                                  plot_species=[],
                                  wd_output=results_wd,
                                  grid_h=env2d.bioDivGrid.h,  # 3D hist of species (e.g. empirical)
                                  distb_objects=[disturbanceGenerator, selective_disturbanceGenerator],
                                  # list of distb_obj, selectivedistb_obj
                                  return_env=True,
                                  ext_risk_obj=ext_risk_obj,
                                  # here set to 1 because env2d.bioDivGrid.h is already fast-evolved to carrying capcaity
                                  max_K_multiplier=1,
                                  suitability=graph_suitability,
                                  future_suitability=graph_future_suitability,
                                  delta_suitability_per_step=delta_suitability_per_step,
                                  cost_layer=graph_cost,
                                  actions_per_step=1,
                                  heuristic_policy="random",
                                  minimize_policy=False,
                                  use_empirical_setup=True,
                                  max_protection_level=parse_str(config["policy"]["max_protection_level"]),
                                  dynamic_print=True,
                                  precomputed_dispersal_probs=dispersal_probs_sparse,
                                  use_small_grid=True,
                                  sensitivities=sensitivities,
                                  K_species=K_species,
                                  pre_steps=10,
                                  sp_carbon=species_carbon,
                                  reference_grid_pu=reference_grid_pu
                                  )

    config_init = copy.deepcopy(config)
    config_init.steps = 5
    config_init.budget = 0
    config_init.actions_per_step = 1
    env_init = cn.run_restore_policy(config_init)
    env_init.set_budget(config.budget)
    return env_init, config_init


# train model
if config["general"]["run_mode"] == "train":
    envList = []
    for i in range(int(config["general"]["batch_size"])):
        print("Setup batch n. %s" % i)
        env_train, config_train = generate_init_env(config, SEED * i)
        env_train._verbose = i == 0 # only job 0 is verbose
        env_train.rewardMode = reward
        env_train.iterations = parse_str(config["general"]["steps"])
        # if budget is None:
        env_train.budget = budget
        actions_per_step = 1
        env_train.reset_init_values()
        env_train.set_calc_reward(True)
        env_train._reward_min_protection = None
        envList.append(copy.deepcopy(env_train))

    if parse_str(config["general"]["fine_tune"]):
        trained_model_file = os.path.join(models_wd, config["files"]["model_file_name"])
        wNN_params = get_nn_params_from_file(trained_model_file,
                                             load_best_epoch=False,
                                             sample_from_iteration=None,
                                             seed=SEED)
    else:
        wNN_params = None

    tmp_res_plot_class = cn_util.plot_map_class(z=reference_grid_pu_nan)

    env_tmp = cn.runBatchGeneticStrategyEmpirical(envList,
                                                  epochs=parse_str(config["general"]["epochs"]),
                                                  lr=0.5,
                                                  lr_adapt=0.01,
                                                  temperature=1,
                                                  max_workers=0,
                                                  outfile=config["files"]["outfile"] + config["general"]["seed"] + ".log",
                                                  obsMode=1,
                                                  observe_error=0,
                                                  running_reward_start=-1000,
                                                  eps_running_reward=0.5,
                                                  sigma=0.2,
                                                  wNN=wNN_params,
                                                  n_NN_nodes=[3, 2],
                                                  increase_temp=0,
                                                  deterministic=True,
                                                  resolution=np.array([1, 1]),
                                                  max_temperature=1000,
                                                  sp_threshold_feature_extraction=1,
                                                  wd_output=results_wd,
                                                  actions_per_step=1,
                                                  protection_per_step=protection_actions_per_step,
                                                  plot_res_class=tmp_res_plot_class,
                                                  reward_weights=reward_weights,
                                                  return_env=True
                                                  )





































