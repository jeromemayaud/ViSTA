# --------------------------- IMPORT MODULES ----------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import operator
import random
import time
import math
import copy
import gc
import cProfile
from scipy.stats import norm, expon

from landscape_SETUP import *

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- SAVE GRIDS *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

def savegrids(sand_heights_grid, veg_grid, apparent_veg_type_grid, age_grid, windspeed_grid, w_soil_moisture_grid, saving_loc, t):    
    #----------------------- Function description -----------------------------
    #Save data to grids   
    
    #---------------------------------- Start ---------------------------------
    saving_grid_sand[(saving_loc*Nr):(saving_loc*Nr)+Nr, 0:Nc] = sand_heights_grid
    saving_grid_veg[(saving_loc*Nr):(saving_loc*Nr)+Nr, 0:Nc] = veg_grid
    saving_grid_apparent_veg_type[(saving_loc*Nr):(saving_loc*Nr)+Nr, 0:Nc] = apparent_veg_type_grid
    saving_grid_age[(saving_loc*Nr):(saving_loc*Nr)+Nr, 0:Nc] = age_grid
    if (t+1) % wind_event_frequency == 0:
        saving_grid_wind[(saving_loc*Nr):(saving_loc*Nr)+Nr, 0:Nc] = windspeed_grid
        saving_grid_moisture[(saving_loc*Nr):(saving_loc*Nr)+Nr, 0:Nc] = w_soil_moisture_grid[((2*n_shells)+1):(Nrw+1), ((2*n_shells)+1):(Ncw+1)]
    saving_loc += 1
    
    return (saving_grid_sand, saving_grid_veg, saving_grid_apparent_veg_type, saving_grid_age, saving_grid_wind, saving_grid_moisture, saving_loc)
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- START ARRAYS *-*-*-*-*-*-*-*-*-*-*-*-*-*-

def startarrays():
    #----------------------- Function description -----------------------------
    #Define initial arrays to fill out with data    
    
    #---------------------------------- Start ---------------------------------
    rainfall_days = np.zeros(model_iterations)  #Defines which days it has rained or not (1=rain, 0=not)
    total_sand_vol = np.zeros(model_iterations) #Total volumne of sand being transported at each iteration
    total_aval_vol = np.zeros(model_iterations) #Volume of sand avalanching at each iteration
    total_veg_pop = np.zeros(veg_iterations) #Total veg cover in the grid at each iteration
    average_age_table = np.zeros((veg_iterations, 3)) #Average age of each veg type at each iteration
    veg_proportions = np.zeros((veg_iterations, 3)) #Proportions of each veg type at each iteration
    exposed_wall_proportions = np.zeros(model_iterations) #% of walls that are fully exposed (i.e. aren't covered in sand) in the grid at each iteration    
    differences_grid = np.zeros((Nrw+2, Ncw+2)) #Grid to allow differences in erosion/deposition to be calculated
    avalanching_grid = np.zeros((Nrw, Ncw)) #Grid to form basis for avalanching grid calculations
    windspeed_grid = np.zeros((Nr, Nc)) #Grid to form basis for windspeed calculations
    interaction_field = np.zeros((Nr, Nc)) #Grid to form basis for neighbourhood interactions between plants
    
    return (rainfall_days, total_sand_vol, total_aval_vol, total_veg_pop, average_age_table, veg_proportions, exposed_wall_proportions, differences_grid, avalanching_grid, windspeed_grid, interaction_field)
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- START GRIDS *-*-*-*-*-*-*-*-*-*-*-*-*-*-*

def startgrids():    
    #----------------------- Function description -----------------------------
    #Define initial grid of sand heights and vegetation        
    
    #-------------------------- Function setup --------------------------------    
    walls_presence_grid = np.zeros((Nr, Nc)) #Grid of 1's and 0's to show where wall is present
    grid = np.zeros((Nr, Nc)) #Grid of 1's and 0's to turn neighbourhood contributions on or off
    drought_grid = np.zeros((Nr, Nc)) #Grid of zeros to add consecutive number of droughts each plant experiences
    veg_type_grid = np.zeros((Nr, Nc)) #Grid to fill with plant types - 0 (no plant), 1 (grass), 2 (shrub), 3 (tree)
    veg_grid = np.zeros((Nr, Nc)) #Grid to fill with plant heights
    cum_growth_grid = np.zeros((Nr, Nc)) #Grid to fill with cumulated plant growth
    actual_biomass_grid = np.zeros((Nr, Nc)) #Grid to fill with the actual biomass that the cumulated growth represents
    chance3 = np.random.random((Nr, Nc)) #Grid of random numbers for allocating plant type randomly to cells
    
    #---------------------------------- Start ---------------------------------
    #------------------------------- Sand heights -----------------------------
    #Define initial grid of sand heights    
    if start_grid_sand_mode == 'random': #Create grid of random sand heights
        sand_heights_grid = np.zeros((Nr, Nc))
        for r in range(Nr):
            for c in range(Nc):
                sand_heights_grid[r, c] = np.random.uniform(start_grid_sand_lower, start_grid_sand_upper) #Fill with a random number from 0 to desired max height
    elif start_grid_sand_mode == 'uniform': #Create grid of uniform heights as specified
        sand_heights_grid = (np.ones((Nr, Nc)))*start_grid_sand_upper
    elif start_grid_sand_mode == 'circle': #Create grid of bare cells with a circle of sand at the centre
        n_r = Nr-1; n_c = Nc-1; circle_centre_r = np.round(Nr/2); circle_centre_c = np.round(Nc/2)
        y, x = np.ogrid[-circle_centre_r:n_r-circle_centre_r, -circle_centre_c:n_c-circle_centre_c]; mask = x**2 + y**2 <= sand_circle_radius**2
        sand_heights_grid = np.ones((Nr, Nc))*0; sand_heights_grid[mask] = sand_circle_height
    else:
        print('WARNING: No start_grid_sand_mode defined!')    
    
    #------------------------------- Wall heights -----------------------------
    #Define initial grid of walls   
    if start_grid_wall_mode == 'none': #No walls in domain
        walls_grid = np.zeros((Nr, Nc))      
    elif start_grid_wall_mode == 'circle': #Circular filled wall at centre of grid 
        n_r = Nr-1; n_c = Nc-1; circle_centre_r = np.round(Nr/2); circle_centre_c = np.round(Nc/2)
        y, x = np.ogrid[-circle_centre_r:n_r-circle_centre_r, -circle_centre_c:n_c-circle_centre_c]; mask = x**2 + y**2 <= wall_circle_or_square_radius_or_length**2
        walls_grid = np.ones((Nr, Nc))*0; walls_grid[mask] = wall_circle_or_square_height
    elif start_grid_wall_mode == 'square': #Square filled wall at centre of grid 
        square_centre_r = np.round(Nr/2); square_centre_c = np.round(Nc/2)
        r_start = square_centre_r - wall_circle_or_square_radius_or_length; r_end = r_start + (2*wall_circle_or_square_radius_or_length)
        c_start = square_centre_c - wall_circle_or_square_radius_or_length; c_end = c_start + (2*wall_circle_or_square_radius_or_length)
        walls_grid = np.zeros((Nr, Nc)); walls_grid[r_start:r_end, c_start:c_end] = wall_circle_or_square_height
    elif start_grid_wall_mode == 'manual': #Manually defined walls using indexing
        walls_grid = np.zeros((Nr, Nc)); walls_grid[10:15, 10:15] = 3; walls_grid[20:25, 20:25] = 2.5 
    else:
        print('WARNING: No start_grid_wall_mode defined!')  
    
    walls_presence_grid[np.where(walls_grid > 0)] = 1 #Fill grid with 1's where walls are present 
    
    #---------------------------------- Ages ----------------------------------
    #Create an initial plant population of ages
    if start_grid_age_mode == 'random': #Random ages up to maximum random age (in months)
        age_grid_random = np.random.random((Nr, Nc))
        age_grid = np.zeros((Nr, Nc))
        age_grid[np.where(age_grid_random <= veg_distrib)] = 1.
        age_grid[np.where(age_grid_random > veg_distrib)] = 0.
        age_grid = (age_grid*(np.round(np.random.random((Nr, Nc))*start_grid_age))) + veg_update_freq_equivalent #Add 1 at the end to make every plant cell at least 1 month old (they won't have any height at 1 month
    elif start_grid_age_mode == 'uniform': #Every vegetated cell has same age    
        age_grid_random = np.random.random((Nr, Nc))
        age_grid = np.zeros((Nr, Nc))
        age_grid[np.where(age_grid_random <= veg_distrib)] = 1.
        age_grid[np.where(age_grid_random > veg_distrib)] = 0.
        age_grid = (age_grid*start_grid_age) + veg_update_freq_equivalent #Add veg_update_freq_equivalent at the end to make every plant cell at least 1 x veg_update_freq_equivalent   
    else:
        print('WARNING: No start_grid_age_mode defined!')
    
    #Update 'grid' for neighbourhood interactions and the occupation grid
    grid[np.where(age_grid > veg_update_freq_equivalent)] = 1 
    veg_occupation_grid = copy.copy(grid) #Grid of 1's and 0's to signify where vegetation exists    
    
    #---------------------------- Veg types and heights -----------------------
    #Define initial grids of veg type, veg height, growth units and biomass according to age
    if (grass_proportion + shrub_proportion + tree_proportion) != 1:
        print('WARNING: Grass/shrub/tree proportions do not sum to 1!') 
    
    for R in range(Nr):
        for C in range(Nc):
            cell_age = age_grid[R, C]
            if cell_age > veg_update_freq_equivalent: #If there is a plant in the cell
                if chance3[R, C] <= grass_proportion: #GRASS
                    veg_type_grid[R, C] = 1 #'1' means the vegetation type is a grass
                    starting_biomass =  max_biomass_grass/(1+(math.exp(-biomass_exp_factor_grass*(cell_age - biomass_midpoint_growth_grass)))) #Starting biomass is the theoretically perfect biomass for that age (taken from veg_update routine)             
                    actual_biomass_grid[R, C] = starting_biomass
                    veg_grid[R, C] = (starting_biomass/max_biomass_grass)*max_height_grass #Simple proportional conversion from biomass units to veg height
                    cum_growth_grid[R, C] = cell_age #Age and cumulative growth units are theoretically the same to start
                elif chance3[R, C] <= (grass_proportion + shrub_proportion): #SHRUB
                    veg_type_grid[R, C] = 2 #'2' means the vegetation type is a shrub
                    starting_biomass =  max_biomass_shrub/(1+(math.exp(-biomass_exp_factor_shrub*(cell_age - biomass_midpoint_growth_shrub)))) #Starting biomass is the theoretically perfect biomass for that age (taken from veg_update routine)             
                    actual_biomass_grid[R, C] = starting_biomass
                    veg_grid[R, C] = (starting_biomass/max_biomass_shrub)*max_height_shrub #Simple proportional conversion from biomass units to veg height
                    cum_growth_grid[R, C] = cell_age #Age and cumulative growth units are theoretically the same to start
                elif chance3[R, C] <= (grass_proportion + shrub_proportion + tree_proportion): #TREE
                    veg_type_grid[R, C] = 3 #'3' means the vegetation type is a tree
                    starting_biomass =  max_biomass_tree/(1+(math.exp(-biomass_exp_factor_tree*(cell_age - biomass_midpoint_growth_tree)))) #Starting biomass is the theoretically perfect biomass for that age (taken from veg_update routine)             
                    actual_biomass_grid[R, C] = starting_biomass
                    veg_grid[R, C] = (starting_biomass/max_biomass_tree)*max_height_tree #Simple proportional conversion from biomass units to veg height
                    cum_growth_grid[R, C] = cell_age #Age and cumulative growth units are theoretically the same to start
            else: #If there is no plant present in cell
                if chance3[R, C] <= grass_proportion: #GRASS
                    veg_type_grid[R, C] = 1 
                elif chance3[R, C] <= (grass_proportion + shrub_proportion): #SHRUB
                    veg_type_grid[R, C] = 2
                elif chance3[R, C] <= (grass_proportion + shrub_proportion + tree_proportion): #TREE
                    veg_type_grid[R, C] = 3
                cum_growth_grid[R, C] = veg_update_freq_equivalent #Minimum growth unit
                actual_biomass_grid[R, C] = veg_update_freq_equivalent #Minimum growth unit        
                veg_grid[R, C] = 0. #No veg height
    
    #Remove plants from walls, as they can grow in the same place as a wall!
    age_grid[np.where(walls_grid > 0)] = 0;
    veg_grid[np.where(walls_grid > 0)] = 0;
    
    return (sand_heights_grid, veg_grid, veg_type_grid, age_grid, cum_growth_grid, actual_biomass_grid, veg_occupation_grid, drought_grid, grid, walls_grid, walls_presence_grid)
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* START GRIDS 2 *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

def startgrids_2(veg_grid, veg_type_grid):    
    #----------------------- Function description -----------------------------
    #Define initial grids of vegetation properties: veg type, porosity, trunk height
    #'veg_grid' is grid of veg heights (Nr x Nc)
    #'veg_type_grid' is grid of veg types: grass, shrub or tree (Nr x Nc)

    #-------------------------- Function setup --------------------------------
    porosity_grid = np.zeros((Nr, Nc))
    trunks_grid = np.zeros((Nr, Nc))
    w_soil_moisture_grid = np.zeros((Nrw+2, Ncw+2))
    
    #---------------------------------- Start ---------------------------------
    #------------------------- Veg porosities ---------------------------------  
    porosity_grid[np.where(veg_grid > 0)] = 1.
    if start_grid_porosity_mode == 'random': #Allocate random porosities to veg
        porosity_grid = porosity_grid*(np.random.uniform(min_random_porosity, max_random_porosity, (Nr, Nc)))
    elif start_grid_porosity_mode == 'uniform': #Every plant has same porosity
        porosity_grid = porosity_grid*start_grid_porosity
    else:
        print('WARNING: No start_grid_porosity_mode defined!')
    
    #---------------------- Trunk heights for trees ---------------------------
    trunks_grid[np.where(veg_type_grid == 3)] = 1.
    if start_grid_trunks_mode == 'random': #Allocate random trunk proportions to veg
        trunks_grid = trunks_grid*veg_grid*(np.random.uniform(min_trunk_proportion, max_trunk_proportion, (Nr, Nc))) #Make the trunk height a proportion of the tree height (observing minimum and maximum proportions the trunk can be)
    elif start_grid_trunks_mode == 'uniform': #Every tree cell has same proportion of total height that is trunk
        trunks_grid = trunks_grid*veg_grid*start_grid_trunks
    else:
        print('WARNING: No start_grid_trunks_mode defined!')
    
    return (porosity_grid, trunks_grid, w_soil_moisture_grid)
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* START GRIDS 3 *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
def startgrids_3(sand_heights_grid, veg_grid, veg_type_grid):    
    #----------------------- Function description -----------------------------
    #Define initial grids for things like sediment balance calculations
    #'veg_grid' is grid of veg heights (Nr x Nc)
    #'veg_type_grid' is grid of veg types: grass, shrub or tree (Nr x Nc)
    
    #---------------------------------- Start ---------------------------------
    initial_sand_heights_grid = copy.copy(sand_heights_grid)
    rooting_heights_grid = copy.copy(sand_heights_grid) #Rooting heights assumed to be at the initial sand surface
    initial_veg_grid = copy.copy(veg_grid)
    initial_apparent_veg_type_grid = copy.copy(veg_type_grid); initial_apparent_veg_type_grid[np.where(initial_veg_grid == 0)] = 0 #It's APPARENT because all cells are attributed a potential veg type, but those that have zero height need to be seen as zero 
    apparent_veg_type_grid = veg_type_grid #Assign this ahead of routine in case veg isn't updated during the run
    sed_balance_moisture_grid = np.zeros((Nrw+2, Ncw+2)) #Just blank sediment balance to begin with
    
    return (initial_sand_heights_grid, rooting_heights_grid, initial_veg_grid, initial_apparent_veg_type_grid, apparent_veg_type_grid, sed_balance_moisture_grid)
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

#*-*-*-*-*-*-*-*-*-*-*-*- START GRIDS MANUAL -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
def startgrids_manual():
    #----------------------- Function description -----------------------------
    #Define initial grids manually (using specific matrix indexing, or by loading existing files)

    #---------------------------------- Start ---------------------------------
    #Define main grids manually    
    sand_heights_grid = np.ones((Nr, Nc))*1; #sand_heights_grid[20:40, 20:40] = 0.4    
    walls_grid = np.zeros((Nr, Nc)); walls_grid[10:15, 10:15] = 2.2; walls_grid[20:25, 20:25] = 4 #Grid of where solid walls are located        
    veg_grid = np.ones((Nr, Nc))*1; veg_grid[np.where(walls_grid > 0)] = 0; #veg_grid[10:40, 10:40] = 4
    age_grid = np.ones((Nr, Nc))*300; age_grid[np.where(walls_grid > 0)] = 0; #age_grid[10:40, 10:40] = 300
    veg_type_grid = np.ones((Nr, Nc)); #veg_type_grid[10:40, 10:40] = 2 #Beware - can't have any zeros in veg_type_grid (a veg type has to be ascribed)  
    
    #Define main grids from previous files
    #sand_heights_grid = np.loadtxt('Final_sand_grid_barchans1', delimiter=',')
    #veg_grid = np.loadtxt('/Users/jeromemayaud/Documents/University/Oxford/DPhil/RESULTS/Modelling/Python/MODEL/MODEL_v6/Veg_grid', delimiter=',')
    
    #Define secondary grids depending on the main grids    
    walls_presence_grid = np.zeros((Nr, Nc)); walls_presence_grid[np.where(walls_grid > 0)] = 1 #Fill grid with 1's where walls are present     
    cum_growth_grid = age_grid #Make the cumulated biomass same as initial age - i.e. plants are perfectly healthy
    actual_biomass_grid = age_grid #Ditto
    grid = np.zeros((Nr, Nc)); grid[np.where(age_grid > veg_update_freq_equivalent)] = 1 #For neighbourhood interactions
    veg_occupation_grid = copy.copy(grid) #Grid of 1's and 0's to signify where vegetation exists    
    drought_grid = np.zeros((Nr, Nc)) #Grid of zeros to add consecutive number of droughts each plant experiences
    
    return (sand_heights_grid, veg_grid, veg_type_grid, age_grid, cum_growth_grid, actual_biomass_grid, veg_occupation_grid, drought_grid, grid, walls_grid, walls_presence_grid)
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- 
