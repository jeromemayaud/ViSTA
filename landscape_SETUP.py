#Import modules from Python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
import sys
import operator
import random
import time
import math

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#                            CHANGEABLE PARAMETERS   
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

#Grid
Nr = 20 #Number of rows in grid (i.e. y axis)
Nc = 20 #Number of columns in grid (i.e. x axis)
cell_width = 1.0 #Width of a cell (m)
manual_initialisation = 'off' #'on' or 'off' - On means the veg and sand grids are manually specified in the 'startgrids_manual' function; off means grids are specified in the 'startgrids' functions
boundary_conditions = 'periodic' #'periodic' or 'open' - Periodic means sand that is lost from the edges is fed back in a Torus formation; 'open' means sand that falls off the edge of the grid is lost forever
start_grid_sand_mode = 'random' #'uniform', 'random' or 'circle' - Random means sand depth is random up to the depth defined by 'start_grid_sand'; 'uniform' means sand depth is uniform over the depth defined by 'start_grid_sand'; 'circle' means the entire landscape is bare apart from a central circle of radius 'sand_circle_radius' and height 'sand_circle_height'
start_grid_wall_mode = 'none' #'none', 'circle', 'square' or 'manual' - None means there are no walls; 'circle' means a central circular filled wall of radius 'wall_circle_or_square_radius_or_length' and height 'wall_circle_or_square_height'; 'square' means a central square filled wall of radius 'wall_circle_or_square_radius_or_length' and height 'wall_circle_or_square_height'; 'manual' means manually entered in the startgrids() routine
start_grid_age_mode = 'random' #'uniform' or 'random' - Random means age is random up to 'start_grid_age'; 'uniform' means age is uniform over 'start_grid_age'
start_grid_trunks_mode = 'uniform' #'uniform' or 'random' - Random means trunk proportion is random between 'min_trunk_proportion' and 'max_trunk_proportion'; 'uniform' means trunk proportion is uniform over 'start_grid_trunks'
start_grid_porosity_mode = 'uniform' #'uniform' or 'random' - Random means plant porosities are random between 'min_random_porosity' and 'max_random_porosity'; 'uniform' means plant porosity is uniform over 'start_grid_porosity'
start_grid_sand_upper = 10.0 #Highest depth of sediment when depths are randomly or uniformly assigned at start (m)
start_grid_sand_lower = 10.0 #Lowest depth of sediment when depths are randomly assigned at start (m)
start_grid_age = 360 #Age of plants when ages are randomly or uniformly assigned at start (months)
start_grid_trunks = 0.7 #Proportion of tree height taken up by trunk, for equal proportion of total tree height that trunks take up (e.g. 0.5 = 50% of total tree height is trunk; 1.0 = 100% of total height is trunk)
start_grid_porosity = 40 #Plant porosities, for equal porosity for all plants (%)
sand_circle_radius = 20 #Radius of sand circle at the centre (if activated in 'start_grid_sand_mode') (m)
sand_circle_height = 10 #Height of sand circle at the centre (if activated in 'start_grid_sand_mode') (m)
wall_circle_or_square_radius_or_length = 5 #Radius of wall circle or half-length of square at the centre (if activated in 'start_grid_wall_mode') (m)
wall_circle_or_square_height = 3. #Height of wall circle or square at the centre (if activated in 'start_grid_wall_mode') (m)
veg_distrib = 0.1 # % of surface covered in vegetation, min = 0, max = 1.0 (i.e. 0.9 means 90% of cells have veg)
grass_proportion = 0.4 #Proportion of the veg distribution that is grass (must sum to 1 with shrub and tree)
shrub_proportion = 0.55 #Proportion of the veg distribution that is shrub (must sum to 1 with grass and tree)
tree_proportion = 0.05 #Proportion of the veg distribution that is tree (must sum to 1 with grass and shrub)

#Time
model_iterations = 1 #Total number of iterations model is running
wind_event_frequency = 1 #Frequency at which wind event occurs (i.e. 5 = one wind event every 5 model iterations)
wind_resolution = 360 #Equivalent time for which wind is blowing during each wind event (MINUTES)
veg_update_frequency = 100000000 #Frequency at which veg is updated using the veg patterning module (i.e. 10 = one update every 10 model iterations)
veg_update_freq_equivalent = 12 #What the frequency of veg update represents in real MONTHS; must be divisible by 12 (1 = monthly (i.e. 12 updates a year), 6 = every 6 months (i.e. 2 updates a year), etc.)
moisture_update_frequency = 1000000 #Frequency at which moisture is updated in feedback (i.e. 5 = one moisture update every 5 model iterations)
moisture_update_frequency_equivalent = 1 #Only relevant for soil moisture feedback routine - it's what the moisture update frequency represents in real HOURS; technically should be compatible with veg update frequency and wind resolution

#Wind
wind_event_timeseries = 'constant' #Type of wind pattern imposed: can be 'constant', 'trend', 'v-shaped', 'weibull'
windspeed_variability = 'off' #'on' adds variability to the unobstructed windspeed at the start of each iteration (according to windspeed_var_normal_stdev) - currently only normal distribution(give a Weibull option too??);'off' removes this
windspeed_stochasticity = 'on' #'on' adds stochasticity to windspeed in each cell (according to the norm_dist_multiplier); 'off' removes this
wind_event_constant = 6. #For 'constant' (m/s) <--- Unobstructed windspeed entering grid for that timestep
wind_event_start = 2; wind_event_end = 10 #For 'trend', 'v-shaped' (m/s) <--- Unobstructed windspeed entering grid for that timestep
windspeed_threshold = 5.1 #Windspeed threshold over which sand transport occurs (m/s)
wind_angle_array = [132] #Angle of wind entering grid for that time step (degrees) (Array is repeated the number of wind iterations)
weibull_shape_parameter = 1.6 #Shape parameter for Weibull distribution if 'wind_event_timeseries' is set to 'weibull'
weibull_scale_parameter = 3.4 #Scale parameter for Weibull distribution if 'wind_event_timeseries' is set to 'weibull'

#Rainfall
rainfall_series_timeseries = 'seasonal' #Type of rainfall temporal pattern imposed: 'constant', 'trend', 'v-shaped', 'asym', 'red', 'seasonal'
rainfall_series_spatial = 'homogeneous' #Type of rainfall spatal pattern imposed: 'homogeneous', 'vertical' or 'corner'
rainfall_variability = 'off' #'on' adds variability to the rainfall at the start of each iteration; 'off' removes this
rainfall_series_constant = 400 #For constant stress on time series (mm, annual equivalent)
rainfall_series_constant_MAM = 57 #If rainfall_series_timeseries is 'seasonal'; March, April, May rainfall
rainfall_series_constant_JJA = 11 #If rainfall_series_timeseries is 'seasonal'; June, July, August rainfall
rainfall_series_constant_SON = 38 #If rainfall_series_timeseries is 'seasonal'; September, October, November rainfall
rainfall_series_constant_DJF = 103 #If rainfall_series_timeseries is 'seasonal'; December, January, February rainfall
rainfall_series_start = 200; rainfall_series_end = 0 #For trend, v-shaped, asym (mm, annual equivalent)
rainfall_series_switch = 5  #For asym
P = 5 #Approx period of red noise (P>1)
c0_start = 1.5; c0_end = -0.5 #Approx mean of red noise
beta_start = 0.1; beta_end = 0.1 #Red noise annual variation parameter
rainfall_series_gradient = 0.3 #For 'vertical' and 'corner' of spatial

#Vegetation
sed_balance_stress_switch = 'off' #'off' if sediment stress is always zero; 'on' if sediment stress is allowed to vary with sediment movement
recolonisation_dynamism = 'on' #'off' means grass/tree/shrub proportions remain the same at every run; 'on' means plant recolonisation is dynamically dependent on current proportions of grass/shrub/tree in the domain
max_height_grass = 1.0 #Maximum height a grass will ever reach if it has full biomass
max_height_shrub = 1.5 #Maximum height a shrub will ever reach if it has full biomass
max_height_tree = 6.0 #Maximum height a tree will ever reach if it has full biomass
max_trunk_proportion = 0.8 #Maximum proportion of a tree that can be made up by its trunk (e.g. 0.5 = 50% of its total height can be trunk; 1.0 = 100% of its total height can be trunk)
min_trunk_proportion = 0.2 #Minimum proportion of a tree that can be made up by its trunk
veg_threshold = 0.08 #Height over which you're interested in plotting population (m)

#Grazing and fire
fire_event_timeseries = 'none' #Type of fire regime imposed: 'none', 'single', 'periodic'
grazing_event_timeseries = 'none' #Type of grazing regime imposed: 'none', 'constant', 'periodic'
fire_event_single = 24 #For 'single' fire regime - month at which a single fire event occurs
fire_event_frequency = 12 #For 'periodic' fire regime - how often a fire happens - ideally, should be divisible by the veg_update_freq_equivalent (months)
grazing_event_frequency = 24 #For 'periodic' grazing frequency - how often a grazing event occurs - ideally, should be divisible by the veg_update_freq_equivalent (months)
stocking_rate = 0.005 #For grazing - number of livestock per hectare (typically up to ~0.06; above 0.06 counts as severely degraded)

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#                             SEMI-FIXED PARAMETERS   
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

#Grid
n_shells = 4 #maximum number of cells a sand slab can move downwind, i.e. size of the margins (4 is adequate)

#Wind/rainfall
windspeed_var_normal_stdev = 1 #Standard deviation for adding variability to unobstructed WINDSPEED at start of each iteration (as an ADDITION parameter)
rainfall_var_normal_stdev = 10 #Standard deviation for adding variability to RAINFALL at start of each iteration (as an ADDITION parameter)
normal_dist_stdev = 0.005 #Standard deviation for deriving a normal distribution in the WINDSPEED, to add stochasticity in each grid cell (as a MULITPLYING parameter)
min_trajectory = 0.1 #Minimum trajectory length required to go through a cell for that cell to be counted in the wind/deposition path (m)
max_downwind_height = 20. #Maximum height downwind of a plant at which flow is affected by that plant; further than this, and it's considered unobstructed flow (in plant heights, h)
min_veg_height_for_wind_effect = 0.10 #Minimum height a plant for it to have an aerodynamic effect on the flow (m)

#Vegetation
max_random_porosity = 95 #Maximum porosity a plant can be when the random porosity routine is run (%)
min_random_porosity = 0 #Maximum porosity a plant can be when the random porosity routine is run (%)
trunk_limit = 0.3 #Height of trunk above which the plant behaves aerodynamically like a tree (m)
peak_maturity_grass = 60 #Age of maturity of a grass (months)
peak_maturity_shrub = 300 #Age of maturity of a shrub (months)
peak_maturity_tree = 480 #Age of maturity of a tree (months)
no_effect_rainfall_grass = 180 #Precipitation midpoint for logistic T-squiggle pathway (for grasses)
no_effect_rainfall_shrub = 200 #Precipitation midpoint for logistic T-squiggle pathway (for shrubs)
no_effect_rainfall_tree = 500 #Precipitation midpoint for logistic T-squiggle pathway (for trees)
no_effect_t_squiggle = 0.0 #T-squiggle midpoint for logistic growth-unit pathway
t_squiggle_drought_threshold = 0.5 #Value of growth_unit below which it is considered the plant experiences a 'plant drought'
t_score_width_grass = 24 #Standard deviation of the normal distribution used to derive T-score, for grasses (months)
t_score_width_shrub = 96 #Standard deviation of the normal distribution used to derive T-score, for shrubs (months)
t_score_width_tree = 152 #Standard deviation of the normal distribution used to derive T-score, for trees (months)
biomass_exp_factor_grass = 0.14 #Curve steepness for logistic biomass-growth pathway, for grasses
biomass_exp_factor_shrub = 0.025 #Curve steepness for logistic biomass-growth pathway, for grasses
biomass_exp_factor_tree = 0.016 #Curve steepness for logistic biomass-growth pathway, for grasses
biomass_midpoint_growth_grass = 25 #Age midpoint for logistic biomass-growth pathway, for grasses (months)
biomass_midpoint_growth_shrub = 130 #Age midpoint for logistic biomass-growth pathway, for shrubs (months)
biomass_midpoint_growth_tree = 210 #Age midpoint for logistic biomass-growth pathway, for trees (months)
alpha_multiplier_grass = 1.0 #How much to multiply alpha score by to extend/restrict the range of rainfall across which patterning occurs (>1 is restricting, <1 is extending)
alpha_multiplier_shrub = 1.0 #How much to multiply alpha score by to extend/restrict the range of rainfall across which patterning occurs (>1 is restricting, <1 is extending)
alpha_multiplier_tree = 0.8 #How much to multiply alpha score by to extend/restrict the range of rainfall across which patterning occurs (>1 is restricting, <1 is extending)
max_biomass_grass = peak_maturity_grass #Number of biomass units a grass can grow to - linked directly to peak height
max_biomass_shrub = peak_maturity_shrub #Number of biomass units a grass can grow to - linked directly to peak height
max_biomass_tree = peak_maturity_tree #Number of biomass units a grass can grow to - linked directly to peak height
max_growth = 1.0 #Maximum growth unit increase of a plant per update period (unitless)
max_decline = 0.05 #0.01 #Maximum growth unit decline of a plant per update period (unitless) 
growthunit_midpoint_growth = 0.4 #T-squiggle midpoint for logistic growth unit
growthunit_exp_factor = 8 #Curve steepness for logistic growth-unit pathway
sed_balance_importance_factor = 0.2 #Value by which sediment balance stress is multiplied to contribute to total stress  
drought_importance_factor = 1.0 #Value by which drought stress is multiplied to contribute to total stress  
number_of_shells = 5 #Number of shells in the veg update neighbourhood (should be 5)
veg_dominance_rainfall = 600 #Rainfall level at which neither grasses nor trees theoretically dominate the recolonisation of a bare cell (mm/yr) - grasses dominate below this, trees dominate above
shell_weight_grass = np.array([1.2, 0.6, 0., -0.1, -0.2]) #Shell weightings for the neighbourhood stress calculation - GRASS - np.array([8., 4., 0.0, -1.5, -2.][2., 1., 0.0, -0.5, -1.]),([4., 2., 0.0, -0.5, -1.2])
shell_weight_shrub = np.array([2., 1., 0., -0.5, -1.]) #Shell weightings for the neighbourhood stress calculation - SHRUBS - np.array([2.5, 1.2, 0., -0.7, -1.2][2.2, 1.1, 0., -0.6, -1.1])
shell_weight_tree = np.array([2., 1., 0., -0.5, -1.]) #Shell weightings for the neighbourhood stress calculation - TREES

#Erosion/deposition
shadow_zone = 15. #Shadow zone limit, above which sand must be deposited (degrees)
prob_depos_bare = 0.4 #Prob of deposition on bare surface (i.e. bedrock)
prob_depos_sand = 0.7 #Prob of deposition on sand-covered surface
sand_density = 2000. # Density of sand across domain (kg/m^3)

#Avalanching
crit_repose = 30. #Critical slab height difference (in degrees) above which avalanching occurs, on sand
veg_crit_repose = 40. #Critical slab height difference (in degrees) above which avalanching occurs, on a vegetated cell
veg_effectiveness_threshold_crit_angle = 0.2 #Veg height at which critical angle of repose changes (m)
veg_effectiveness_threshold_prob_depos = 0.2 #Veg height at which probability of deposition changes (m)
max_neighbour_checks = 4 #Maximum number of rounds of neighbour checking for avalanching (to minimise time taken to run avalanching routine) 

#Airflow compression
max_air_compression_change = 1.5 #Maximum acceleration due to air compression ('1' = 100% more speed)
min_air_compression_change = -1.0 #Maximum deceleration due to air compression ('-1' = 100% less speed, i.e. zero)
max_slope = 0.1 #Lower number = higher acceleration (keep at ~0.1)
min_slope = -0.1 #Lower number = higher deceleration (keep at ~0.1)

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#                                FIXED PARAMETERS   
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

#Grid
Nrw = Nr + (2*n_shells) #Row dimension of wrapped grid
Ncw = Nc + (2*n_shells) #Column dimension of wrapped grid
grid_start = n_shells #Beginning of core grid, which is embedded in the centre of a larger grid with wrapping of n_shells around the edges
grid_end_r = Nr + n_shells #End row of core grid, which is embedded in the centre of a larger grid with wrapping of n_shells around the edges
grid_end_c = Nc + n_shells #End row of core grid, which is embedded in the centre of a larger grid with wrapping of n_shells around the edges

#Wind
wind_iterations = math.floor(model_iterations/wind_event_frequency) #Number of times wind update occurs, given the specification above - DON'T CHANGE FORMULA

#Vegetation
age_threshold = veg_update_freq_equivalent*2 #Age over which you're interested in plotting population (months)
spinup_time = 0; spinup_rainfall_series = -1; spinup_wind_series = -1 #Keep at 0 and -1 for the vegetation stress series
veg_iterations = math.floor(model_iterations/veg_update_frequency) #Number of times veg_update is called, given the specification above

#Graphics
saving_frequency_movie = 1 #Frequency at which frames are taken for movie, in wind iterations (i.e. 50 = one plot every 50 iterations)
saving_grid_sand = np.zeros(shape=[Nr*(math.ceil(model_iterations/saving_frequency_movie)), Nc]) #Create a grid that can accommodate sand heights frames for display as movie in Matlab
saving_grid_wind = np.zeros(shape=[Nr*(math.ceil(model_iterations/saving_frequency_movie)), Nc]) #Create a grid that can accommodate wind speed frames for display as movie in Matlab
saving_grid_veg = np.zeros(shape=[Nr*(math.ceil(model_iterations/saving_frequency_movie)), Nc]) #Create a grid that can accommodate veg frames for display as movie in Matlab
saving_grid_apparent_veg_type = np.zeros(shape=[Nr*(math.ceil(model_iterations/saving_frequency_movie)), Nc]) #Create a grid that can accommodate veg type frames for display as movie in Matlab
saving_grid_age = np.zeros(shape=[Nr*(math.ceil(model_iterations/saving_frequency_movie)), Nc]) #Create a grid that can accommodate age frames for display as movie in Matlab
saving_grid_moisture = np.zeros(shape=[Nr*(math.ceil(model_iterations/saving_frequency_movie)), Nc]) #Create a grid that can accommodate moisture for display as movie in Matlab
saving_loc = 0 #For the saving frames