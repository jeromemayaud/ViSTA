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

#*-*-*-*-*-*-*-*-*-*- DEFINE STRESS TIME SERIES AND MODELS *-*-*-*-*-*-*-*-*-*-

def define_rainfall_series():
    #----------------------- Function description -----------------------------
    # Defines how the rainfall changes over time and across the grid (useful for simulating rainfall gradients, etc)
    
    #-------------------------- Function setup --------------------------------
    total_runs = veg_iterations + spinup_time
    rainfall_series = np.zeros(total_runs); rainfall_series[0:spinup_time] = spinup_rainfall_series #rainfall_series defines the stress on plants
    spare = np.arange(spinup_time, total_runs)
    stress_mask = np.zeros((Nr, Nc))
    
    #---------------------------------- Start ---------------------------------
    # Define stress time series
    if rainfall_series_timeseries == 'constant': #Constant rainfall at same value
        rainfall_series[spinup_time:total_runs] = rainfall_series_constant
    elif rainfall_series_timeseries == 'trend': #Constant increase or decrease in rainfall
        rainfall_series_rate = (rainfall_series_end-rainfall_series_start)/veg_iterations
        rainfall_series[spinup_time:total_runs] = rainfall_series_start+(spare+1-spinup_time)*rainfall_series_rate
    elif rainfall_series_timeseries == 'v-shaped': #Rainfall decreases then increases again (or vice versa)
        rainfall_series_rate = (rainfall_series_end-rainfall_series_start)/(0.5*veg_iterations)
        for spare in range(1, veg_iterations+1):
            if spare < (veg_iterations/2):
                rainfall_series[spare-1+spinup_time] = rainfall_series_start + spare*rainfall_series_rate
            else:
                rainfall_series[spare-1+spinup_time] = (rainfall_series_start+(veg_iterations/2)*rainfall_series_rate)-(spare-(veg_iterations/2))*rainfall_series_rate
    elif rainfall_series_timeseries == 'asym': #Asymmetrical rainfall
        leg1 = rainfall_series_switch-rainfall_series_start; leg2 = rainfall_series_end-rainfall_series_switch
        rainfall_series_rate = (abs(leg1)+abs(leg2))/veg_iterations
        runs1 = round(veg_iterations*(abs(leg1)/(abs(leg1)+abs(leg2))))
        if rainfall_series_start > rainfall_series_switch:
            spare2 = np.arange(0, runs1); n_spare2 = len(spare2)
            rainfall_series[spinup_time:spinup_time+n_spare2] = rainfall_series_start-spare2*rainfall_series_rate
            spare3 = np.arange(runs1, veg_iterations)
            rainfall_series[spinup_time+n_spare2:total_runs] = (rainfall_series_start-runs1*rainfall_series_rate)+((spare3-veg_iterations)*rainfall_series_rate)
        else:
            spare4 = np.arange(0, runs1); n_spare4 = runs1
            rainfall_series[spinup_time:spinup_time+runs1] = rainfall_series_start+spare4*rainfall_series_rate
            spare5 = np.arange(runs1, veg_iterations)
            rainfall_series[spinup_time+n_spare4:total_runs] = (rainfall_series_start+runs1*rainfall_series_rate)-((spare5-runs1)*rainfall_series_rate)
    elif rainfall_series_timeseries == 'red':
        Rnd = np.random.normal(0,1,veg_iterations)
        c0_rate = (c0_end-c0_start)/veg_iterations
        beta_rate = (beta_end-beta_start)/veg_iterations
        rainfall_series[spinup_time] = c0_start
        for i in range(1, veg_iterations):
            rainfall_series[i+spinup_time] = (1-(1/P))*(rainfall_series[spinup_time+i-1]-(c0_start+c0_rate*1))+(c0_start+c0_rate*i)+(beta_start+beta_rate*i)*Rnd(1)
    elif rainfall_series_timeseries == 'seasonal': #To enable each of the four seasons to have a different rainfall
        rainfall_series[0::4] = rainfall_series_constant_MAM #The first and then every 4th item is rainfall for March, April, May
        rainfall_series[1::4] = rainfall_series_constant_JJA #The second and then every 4th item is rainfall for June, July, August
        rainfall_series[2::4] = rainfall_series_constant_SON #The third and then every 4th item is rainfall for September, October, November
        rainfall_series[3::4] = rainfall_series_constant_DJF #The fourth and then every 4th item is rainfall for December, January, February
    else:
        print('WARNING: The stress timeseries model is not entered correctly')
    
    #Define spatial model - if spatial model is homogeneous, stress mask remains zero
    if rainfall_series_spatial == 'homogeneous':
        print('Spatial model is homogeneous')
    elif rainfall_series_spatial == 'vertical':
        for rows in range(Nr):
            for columns in range(Nc):
                stress_mask[rows, columns] = -(rainfall_series_gradient/2)+rows*(rainfall_series_gradient/Nr)
    elif rainfall_series_spatial == 'corner':
        for rows in range(Nr):
            for columns in range(Nc):
                stress_mask[rows, columns] = -(rainfall_series_gradient/2)+rows*(rainfall_series_gradient/Nr)+columns*(rainfall_series_gradient/Nc)
    else:
        print('WARNING: The stress timeseries model is not entered correctly')
        
    #If the rainfall time series has to be variable
    if rainfall_variability == 'on':
        variability_multipliers = np.random.normal(0, rainfall_var_normal_stdev, veg_iterations) #Create an array of normally-distributed multipliers
        for tr in range(veg_iterations):
            new_rainfall = rainfall_series[tr] + variability_multipliers[tr]
            if new_rainfall < 0:
                new_rainfall = 0
            rainfall_series[tr] = new_rainfall
            
    return(rainfall_series)
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

#*-*-*-*-*-*-*-*-*-*-*- DEFINE STRESS TIME SERIES AND MODELS *-*-*-*-*-*-*-*-*-

def define_fire_grazing_series():
    #----------------------- Function description -----------------------------
    #Defines how grazing and fire occurrence vary over time    
    
    #-------------------------- Function setup --------------------------------
    fire_series = np.zeros(veg_iterations) #Define fire series
    grazing_series = np.zeros(veg_iterations) #Define grazing series
    
    #---------------------------------- Start ---------------------------------
    if fire_event_timeseries == 'none':
        fire_series = fire_series
    elif fire_event_timeseries == 'single':
        xx = math.floor(fire_event_single/veg_update_freq_equivalent)
        fire_series[xx-1] = 1
    elif fire_event_timeseries == 'periodic':        
        xx = math.floor(fire_event_frequency/veg_update_freq_equivalent)
        yy = math.floor(veg_iterations/xx)
        for zz in range(1, yy):
            fire_series[(xx*zz)-1] = 1
    else:
        print('WARNING: The stress timeseries model is not entered correctly')
    
    if grazing_event_timeseries == 'none':
        grazing_series = grazing_series
    elif grazing_event_timeseries == 'constant':
        grazing_series[0:veg_iterations] = 1
    elif grazing_event_timeseries == 'periodic':
        xx = math.floor(grazing_event_frequency/veg_update_freq_equivalent)
        yy = math.floor(veg_iterations/xx)
        for zz in range(1, yy):
            grazing_series[(xx*zz)-1] = 1
    else:
        print('WARNING: The stress timeseries model is not entered correctly')
            
    return(fire_series, grazing_series)
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- VEG UPDATE *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

def veg_update(veg_grid, veg_type_grid, age_grid, sand_heights_grid, cum_growth_grid, actual_biomass_grid, veg_occupation_grid, rooting_heights_grid, trunks_grid, porosity_grid, drought_grid, grid, walls_grid, rainfall_series, veg_t, fire_event, grazing_event):
    #----------------------- Function description -----------------------------
    # Performs the full vegetation update routine, where veg grows and dies according to stresses
    # 'veg_grid' is the vertical height of plants (Nr x Nc)
    # 'veg_type_grid' is the type of veg: grass, shrub or tree (Nr x Nc)
    # 'age_grid' is the age of plants (Nr x Nc)    
    # 'sand_heights_grid' is the grid of sand heights (Nr x Nc)
    # 'cum_growth_grid' is the cumulative growth unit for each plant (Nr x Nc)
    # 'actual_biomass_grid' is the actual biomass of each plant from previous timestep, so that all biomasses can change in one go (Nr x Nc)    
    # 'veg_occupation_grid' is 0 or 1 depending on whether there is already plant there or it died in the previous timestep (Nr x Nc)     
    # 'rooting_heights_grid' is the rooting heights of each plant (Nr x Nc)
    # 'trunks_grid' is the vertical height of trunks under each plant (Nr x Nc)
    # 'porosity_grid' is the porosity of plants (Nr x Nc)
    # 'drought_grid' is cumulative number of 'plant droughts' each plant has experienced (Nr x Nc) 
    # 'grid' is filled with 1's and 0's to turn neighbourhood contributions on or off
    # 'walls_grid' is a grid where solid walls are located    
    # 'rainfall_series' is temporal series of rain levels
    # 'veg_t' is tick of how often veg has been updated
    # 'fire_event' is 1 or 0 depending on if a fire has occurred
    # 'grazing_event' is 1 or 0 depending on if grazing has occurred

    #-------------------------- Function setup --------------------------------
    period_rainfall = rainfall_series[veg_t]*(veg_update_freq_equivalent/12) #Take appropriate stress value from time series according to how many veg updates there have already been        
    total = np.array([0., 0., 0., 0., 0.])
    update = np.zeros((Nr, Nc))
    interaction_field = np.zeros((Nr, Nc))
    wrap_r = 0; wrap_c = 0
    average_age_array = np.zeros(3) #To fill with the average ages of grasses, shrubs and trees in this iteration
    current_grass_proportion = ((veg_type_grid == 1).sum())/(Nr*Nc) ##Find current proportion of grasses, of all the cells that are vegetated
    current_shrub_proportion = ((veg_type_grid == 2).sum())/(Nr*Nc) #Ditto for shrubs
    current_tree_proportion = ((veg_type_grid == 3).sum())/(Nr*Nc) #Ditto for trees
    chance = np.random.random((Nr, Nc)); chance2 = np.random.random((Nr, Nc)) #Create random chance matrices
    next_age = copy.copy(age_grid) #Replicate age matrix to update it with new ages
    next_biomass = copy.copy(actual_biomass_grid) #Replicate biomass matrix to update it with new biomass
    next_veg_type = copy.copy(veg_type_grid) #Replicate veg type matrix to update it with new veg type if plant dies
    
    #---------------------------------- Start ---------------------------------
    print("Grass %:", round(current_grass_proportion, 2), "Shrub %:", round(current_shrub_proportion, 2), "Tree %:", round(current_tree_proportion, 2))
    
    #-------------------------- Set up response rules (in months) --------------
    age_range = np.linspace(1, 4800, 4800) #Age range of 400 years (can make bigger, won't affect the curve definitions below)
    
    #GRASSES
    kc_grass = norm.pdf(age_range, peak_maturity_grass, t_score_width_grass)*60 #kc: Competition strength, age dependence
    kf_grass = norm.pdf(np.arange(1, peak_maturity_grass), peak_maturity_grass, t_score_width_grass)*60; upper_range = np.ones(900)*0.999; kf_grass = np.concatenate([kf_grass, upper_range]) #kf: Facilitation strength age dependence
    stress_sensitivity_grass = (1-norm.pdf(age_range, peak_maturity_grass, t_score_width_grass)*54) #sc: Sensitivity to competition
    sf_grass = np.ones(math.floor(max(age_range))) #sf: Sensitvity to facilitation
    for z in range(math.floor(max(age_range))):
        sf_grass[z] = 1+(2*math.exp(-0.208333*z))
    age_stress_grass = np.zeros(math.floor(max(age_range))); as_factor = 5 #Age-related stress for grasses
    age_stress_grass[0:peak_maturity_grass] = (0.2-norm.pdf(age_range[0:peak_maturity_grass], peak_maturity_grass, t_score_width_grass)*24)/as_factor
    age_stress_grass[peak_maturity_grass:len(age_range)] = (1-norm.pdf(age_range[peak_maturity_grass:len(age_range)], peak_maturity_shrub, (t_score_width_shrub/2))*120)/as_factor #NOTE - the peak maturity and t-score-width of the SHRUB is used on purpose here, because otherwise the grass goes up to max age stress too quickly
    
    #SHRUBS
    kc_shrub = norm.pdf(age_range, peak_maturity_shrub, t_score_width_shrub)*240 #kc: Competition strength, age dependence
    kf_shrub = norm.pdf(np.arange(1, peak_maturity_shrub), peak_maturity_shrub, t_score_width_shrub)*240; upper_range = np.ones(900)*0.999; kf_shrub = np.concatenate([kf_shrub, upper_range]) #kf: Facilitation strength age dependence
    stress_sensitivity_shrub = (1-norm.pdf(age_range, peak_maturity_shrub, t_score_width_shrub)*216) #sc: Sensitivity to competition
    sf_shrub = np.ones(math.floor(max(age_range))) #sf: Sensitvity to facilitation
    for z in range(math.floor(max(age_range))):
        sf_shrub[z] = 1+(2*math.exp(-0.04166666*z))
    age_stress_shrub = np.zeros(math.floor(max(age_range))); as_factor = 5 #Age-related stress for shrubs
    age_stress_shrub[0:peak_maturity_shrub] = (0.2-norm.pdf(age_range[0:peak_maturity_shrub], peak_maturity_shrub, t_score_width_shrub)*48)/as_factor
    age_stress_shrub[peak_maturity_shrub:len(age_range)] = (1-norm.pdf(age_range[peak_maturity_shrub:len(age_range)], peak_maturity_shrub, t_score_width_shrub)*240)/as_factor 
    
    #TREES    
    kc_tree = norm.pdf(age_range, peak_maturity_tree, t_score_width_tree)*380 #kc: Competition strength, age dependence
    kf_tree = norm.pdf(np.arange(1, peak_maturity_tree), peak_maturity_tree, t_score_width_tree)*380; upper_range = np.ones(900)*0.999; kf_tree = np.concatenate([kf_tree, upper_range]) #kf: Facilitation strength age dependence
    stress_sensitivity_tree = (1-norm.pdf(age_range, peak_maturity_tree, t_score_width_tree)*342) #sc: Sensitivity to competition
    sf_tree = np.ones(math.floor(max(age_range))) #sf: Sensitvity to facilitation
    for z in range(math.floor(max(age_range))):
        sf_tree[z] = 1+(2*math.exp(-0.02604*z))
    age_stress_tree = np.zeros(math.floor(max(age_range))); as_factor = 5 #Age-related stress for trees
    age_stress_tree[0:peak_maturity_tree] = (0.2-norm.pdf(age_range[0:peak_maturity_tree], peak_maturity_tree, t_score_width_tree)*76)/as_factor
    age_stress_tree[peak_maturity_tree:len(age_range)] = (1-norm.pdf(age_range[peak_maturity_tree:len(age_range)], peak_maturity_tree, (t_score_width_shrub*2))*480)/as_factor #NOTE - the t-score-width of the SHRUB is used on purpose here
    #--------------------------------------------------------------------------
    
    #--------------------------- Vegetation update ----------------------------
    for R in range(Nr): #Evaluate grid
        for C in range(Nc):
            actual_biomass_polled_cell = actual_biomass_grid[R, C].astype(int)
            age_polled_cell = age_grid[R, C].astype(int)
            veg_type_polled_cell = veg_type_grid[R, C].astype(int)
            
            #Define appropriate parameters of the vegetated cell depending on its type               
            if veg_type_polled_cell == 1:
                max_veg_height = max_height_grass
                shell_weight = shell_weight_grass
                sf = sf_grass
                stress_sensitivity = stress_sensitivity_grass
                biomass_exp_factor = biomass_exp_factor_grass
                biomass_midpoint_growth = biomass_midpoint_growth_grass
                max_biomass = max_biomass_grass
            elif veg_type_polled_cell == 2:
                max_veg_height = max_height_shrub
                shell_weight = shell_weight_shrub
                sf = sf_shrub
                stress_sensitivity = stress_sensitivity_shrub
                biomass_exp_factor = biomass_exp_factor_shrub
                biomass_midpoint_growth = biomass_midpoint_growth_shrub
                max_biomass = max_biomass_shrub
            elif veg_type_polled_cell == 3:
                max_veg_height = max_height_tree
                shell_weight = shell_weight_tree
                sf = sf_tree
                stress_sensitivity = stress_sensitivity_tree
                biomass_exp_factor = biomass_exp_factor_tree
                biomass_midpoint_growth = biomass_midpoint_growth_tree
                max_biomass = max_biomass_tree
                if start_grid_trunks_mode == 'uniform': #Only need to calculate ratio of trunk to total height if cell is a tree, and it has at least bit of height, and the trunks are randomly proportioned
                    if veg_grid[R, C] > 0:
                        trunk_ratio_polled_cell = trunks_grid[R, C]/veg_grid[R, C]
                    else:
                        trunk_ratio_polled_cell = 0
            else:
                max_veg_height = 0.1 #Make it 0.1 instead of zero just in case there's an issue further down?
                
            #Define neighbourhood
            for S in range(1, number_of_shells+1):
                total[S-1] = 0
                polled_shell_weight = shell_weight[S-1] #Define age of currently polled cell
                #Top row of neighbourhood
                for x in range (0,(2*S+1)):
                    if R-S < 0:
                        wrap_r = Nr-1
                    if C-S+x < 0:
                        wrap_c = Nc-1
                    if C-S+x > Nc-1:
                        wrap_c = -Nc+1
                    #Allocate the correct kf/kc arrays depending on the vegetation type of the plant that is currently being polled in the neighbourhood
                    if veg_type_grid[R-S+wrap_r, C-S+x+wrap_c].astype(int) == 1:
                        kf = kf_grass; kc = kc_grass
                    elif veg_type_grid[R-S+wrap_r, C-S+x+wrap_c].astype(int) == 2:
                        kf = kf_shrub; kc = kc_shrub
                    elif veg_type_grid[R-S+wrap_r, C-S+x+wrap_c].astype(int) == 3:
                        kf = kf_tree; kc = kc_tree
                    if S < 3: #Facilitation for shells 1, 2
                        total[S-1] = total[S-1]+grid[R-S+wrap_r, C-S+x+wrap_c]*(kf[actual_biomass_grid[R-S+wrap_r, C-S+x+wrap_c].astype(int)-1]*polled_shell_weight/(S*8))*sf[actual_biomass_polled_cell]
                    else: #Competition for shells 3, 4, 5
                        total[S-1] = total[S-1]+grid[R-S+wrap_r, C-S+x+wrap_c]*(kc[actual_biomass_grid[R-S+wrap_r, C-S+x+wrap_c].astype(int)-1]*polled_shell_weight/(S*8))*stress_sensitivity[actual_biomass_polled_cell-1]
                    wrap_r = 0; wrap_c = 0
                #Bottom row of neighbourhood
                for x in range (0,(2*S+1)):
                    if R+S > Nr-1:
                        wrap_r = -Nr+1
                    if R+S < 0:
                        wrap_r = Nr-1
                    if C-S+x < 0:
                        wrap_c = Nc-1
                    if C-S+x > Nc-1:
                        wrap_c = -Nc+1
                    #Allocate the correct kf/kc arrays depending on the vegetation type of the plant that is currently being polled in the neighbourhood
                    if veg_type_grid[R+S+wrap_r, C-S+x+wrap_c].astype(int) == 1:
                        kf = kf_grass; kc = kc_grass
                    elif veg_type_grid[R+S+wrap_r, C-S+x+wrap_c].astype(int) == 2:
                        kf = kf_shrub; kc = kc_shrub
                    elif veg_type_grid[R+S+wrap_r, C-S+x+wrap_c].astype(int) == 3:
                        kf = kf_tree; kc = kc_tree
                    if S < 3: #Facilitation for shells 1, 2
                        total[S-1] = total[S-1]+grid[R+S+wrap_r, C-S+x+wrap_c]*(kf[actual_biomass_grid[R+S+wrap_r, C-S+x+wrap_c].astype(int)-1]*polled_shell_weight/(S*8))*sf[actual_biomass_polled_cell]
                    else: #Competition for shells 3, 4, 5
                        total[S-1] = total[S-1]+grid[R+S+wrap_r, C-S+x+wrap_c]*(kc[actual_biomass_grid[R+S+wrap_r, C-S+x+wrap_c].astype(int)-1]*polled_shell_weight/(S*8))*stress_sensitivity[actual_biomass_polled_cell-1]
                    wrap_r = 0; wrap_c = 0
                #Left column of neighbourhood
                for x in range (0,(2*S-1)):
                    if R-S+1+x < 0:
                        wrap_r = Nr-1
                    if R-S+1+x > Nr-1:
                        wrap_r = -Nr+1
                    if C-S < 0:
                        wrap_c = Nc-1
                    if C-S > Nc-1:
                        wrap_c = -Nc+1      
                    #Allocate the correct kf/kc arrays depending on the vegetation type of the plant that is currently being polled in the neighbourhood
                    if veg_type_grid[R-S+1+x+wrap_r, C-S+wrap_c].astype(int) == 1:
                        kf = kf_grass; kc = kc_grass
                    elif veg_type_grid[R-S+1+x+wrap_r, C-S+wrap_c].astype(int) == 2:
                        kf = kf_shrub; kc = kc_shrub
                    elif veg_type_grid[R-S+1+x+wrap_r, C-S+wrap_c].astype(int) == 3:
                        kf = kf_tree; kc = kc_tree
                    if S < 3: #Facilitation for shells 1, 2
                        total[S-1] = total[S-1]+grid[R-S+1+x+wrap_r, C-S+wrap_c]*(kf[actual_biomass_grid[R-S+1+x+wrap_r, C-S+wrap_c].astype(int)-1]*polled_shell_weight/(S*8))*sf[actual_biomass_polled_cell]
                    else: #Competition for shells 3, 4, 5
                        total[S-1] = total[S-1]+grid[R-S+1+x+wrap_r, C-S+wrap_c]*(kc[actual_biomass_grid[R-S+1+x+wrap_r, C-S+wrap_c].astype(int)-1]*polled_shell_weight/(S*8))*stress_sensitivity[actual_biomass_polled_cell-1]
                    wrap_r = 0; wrap_c = 0
                #Right column of neighbourhood
                for x in range (0,(2*S-1)):
                    if R-S+1+x < 0:
                        wrap_r = Nr-1
                    if R-S+1+x > Nr-1:
                        wrap_r = -Nr+1
                    if C+S < 0:
                        wrap_c = Nc-1
                    if C+S > Nc-1:
                        wrap_c = -Nc+1
                    #Allocate the correct kf/kc arrays depending on the vegetation type of the plant that is currently being polled in the neighbourhood
                    if veg_type_grid[R-S+1+x+wrap_r, C+S+wrap_c].astype(int) == 1:
                        kf = kf_grass; kc = kc_grass
                    elif veg_type_grid[R-S+1+x+wrap_r, C+S+wrap_c].astype(int) == 2:
                        kf = kf_shrub; kc = kc_shrub
                    elif veg_type_grid[R-S+1+x+wrap_r, C+S+wrap_c].astype(int) == 3:
                        kf = kf_tree; kc = kc_tree
                    if S < 3: #Facilitation for shells 1, 2
                        total[S-1] = total[S-1]+grid[R-S+1+x+wrap_r, C+S+wrap_c]*(kf[actual_biomass_grid[R-S+1+x+wrap_r, C+S+wrap_c].astype(int)-1]*polled_shell_weight/(S*8))*sf[actual_biomass_polled_cell]
                    else: #Competition for shells 3, 4, 5
                        total[S-1] = total[S-1]+grid[R-S+1+x+wrap_r, C+S+wrap_c]*(kc[actual_biomass_grid[R-S+1+x+wrap_r, C+S+wrap_c].astype(int)-1]*polled_shell_weight/(S*8))*stress_sensitivity[actual_biomass_polled_cell-1]
                    wrap_r = 0; wrap_c = 0
            
            #Apply rules --> Rainfall stress is ADDED to the neighbourhood stress (like Bailey, 2011)
            if veg_type_polled_cell == 1:
                alpha = (((period_rainfall*(12/veg_update_freq_equivalent)) - no_effect_rainfall_grass)/no_effect_rainfall_grass)*alpha_multiplier_grass #Rainfall stress for grasses
            elif veg_type_polled_cell == 2:
                alpha = (((period_rainfall*(12/veg_update_freq_equivalent)) - no_effect_rainfall_shrub)/no_effect_rainfall_shrub)*alpha_multiplier_shrub #Rainfall stress for grasses
            elif veg_type_polled_cell == 3:
                alpha = (((period_rainfall*(12/veg_update_freq_equivalent)) - no_effect_rainfall_tree)/no_effect_rainfall_tree)*alpha_multiplier_tree #Rainfall stress for trees
            total_stress = (sum(total)) + alpha #This is the T-score
            neigh_score = ((1/(1+math.exp(total_stress*-5)))) #This is T-squiggle    
            interaction_field[R, C] = neigh_score 
            
            #Calculate the growth unit
            growth_unit = ((max_growth+max_decline)/(1+math.exp(-growthunit_exp_factor*(neigh_score - growthunit_midpoint_growth)))) - max_decline #Growth unit depends on t-squiggle - rainfall has no effect at value of 'no_effect_t_squiggle'
                        
            #Calculate cumulated growth
            cumulated_growth = cum_growth_grid[R, C] + (growth_unit*veg_update_freq_equivalent) #Combine previous growth units and add the effect from rain - multiply by veg_update_freq_equivalent because to be proportional to how many months have passed since last veg update                    
 
            #Update all grids based on the cumulated growth
            if cumulated_growth <= veg_update_freq_equivalent: #Have to maintain minimum time unit
                update[R, C] = 0
                next_age[R, C] = veg_update_freq_equivalent
                cumulated_growth = veg_update_freq_equivalent #Need to give it the minimum cum_growth possible
                cum_growth_grid[R, C] = cumulated_growth #Put the adapted cumulated growth into the grid for next update                                 
                actual_biomass = veg_update_freq_equivalent
                next_biomass[R, C] = actual_biomass #Actual biomass is not the same as cumulated growth                  
                veg_grid[R, C] = 0.
                
            else: #If cumulated growth is above minimum time unit
                update[R, C] = 1
                next_age[R, C] += veg_update_freq_equivalent
                cum_growth_grid[R, C] = cumulated_growth #Put the adapted cumulated growth into the grid for next update                
                actual_biomass = max_biomass/(1+(math.exp(-biomass_exp_factor*(cumulated_growth - biomass_midpoint_growth)))) #Find what the cumulated growth is equivalent to in real biomass (because medium-sized plants will accrue biomass more readily than v big or v small plants)
                next_biomass[R, C] = actual_biomass #Actual biomass is not the same as cumulated growth                  
                if (actual_biomass/max_biomass)*max_veg_height >= veg_threshold: #Below a threshold, biomass is just stored as seedling, the veg doesn't have any height
                    veg_grid[R, C] = (actual_biomass/max_biomass)*max_veg_height #Simple proportional conversion from biomass to veg height                  
                else:
                    veg_grid[R, C] = 0
            
            if veg_type_polled_cell == 3: #If cell is a tree, update its trunk height 
                if start_grid_trunks == -1: #Allocate previous trunk proportion to polled tree if it's random
                    trunks_grid[R, C] = trunk_ratio_polled_cell*veg_grid[R, C]
                else: #Allocate stated trunk proportion to polled tree
                    trunks_grid[R, C] = start_grid_trunks*veg_grid[R, C]
                    
            #------------------- Drought stress -------------------------------            
            if neigh_score < t_squiggle_drought_threshold: #Add drought units to drought grid if neighbourhood score is negative                                    
                drought_grid[R, C] += veg_update_freq_equivalent*((t_squiggle_drought_threshold - neigh_score)/t_squiggle_drought_threshold)
            else: #If drought series is broken, return to zero
                drought_grid[R, C] = 0
            consecutive_droughts = drought_grid[R, C]
            
            #Adapt exponential parameter depending on vegetation type
            if veg_type_polled_cell == 1: #GRASS
                b_parameter = ((-0.00026666666*actual_biomass_polled_cell) + 0.1)/1
            elif veg_type_polled_cell == 2: #SHRUB
                b_parameter = ((-0.00026666666*actual_biomass_polled_cell) + 0.1)/2
            elif veg_type_polled_cell == 3: #TREE
                b_parameter = ((-0.00026666666*actual_biomass_polled_cell) + 0.1)/5
            
            if consecutive_droughts > 0:
                drought_stress = (1 - math.exp(-b_parameter*consecutive_droughts))*drought_importance_factor
            else:
                drought_stress = 0                          
            
            #-------------- Sediment balance stress ---------------------------
            if sed_balance_stress_switch == 'on':                 
                sed_balance = sand_heights_grid[R, C] - rooting_heights_grid[R, C] #Sediment balance is the current sand height minus the height at which the plant rooted
                sed_balance_annual_equiv = sed_balance*(12/veg_update_freq_equivalent) #Adjust to annual equivalent 
                
                if veg_type_polled_cell == 1: #GRASS                   
                    if sed_balance_annual_equiv < -1.5:
                        sed_stress = -0.5
                    elif sed_balance_annual_equiv >= -1.5 and sed_balance_annual_equiv < 0.1:
                        sed_stress = (sed_balance_annual_equiv*0.9375)-0.09375
                    elif sed_balance_annual_equiv >= 0.1 and sed_balance_annual_equiv < 0.6:
                        sed_stress = sed_balance_annual_equiv-0.1
                    elif sed_balance_annual_equiv >= 0.6 and sed_balance_annual_equiv < 2.0:
                        sed_stress = (sed_balance_annual_equiv*-0.35714286)+0.71428571
                    elif sed_balance_annual_equiv >= 2.0:
                        sed_stress = 0.
                elif veg_type_polled_cell == 2: #SHRUB                  
                    if sed_balance_annual_equiv < -1.4:
                        sed_stress = -0.25
                    elif sed_balance_annual_equiv >= -1.4 and sed_balance_annual_equiv < -0.6:
                        sed_stress = (sed_balance_annual_equiv*0.3125)+0.1875
                    elif sed_balance_annual_equiv >= -0.6 and sed_balance_annual_equiv < 0:
                        sed_stress = (sed_balance_annual_equiv*0.166666)+0.1
                    elif sed_balance_annual_equiv >= 0 and sed_balance_annual_equiv < 2.75:
                        sed_stress = (sed_balance_annual_equiv*-0.12727)+0.1
                    elif sed_balance_annual_equiv >= 2.75:
                        sed_stress = -0.25
                elif veg_type_polled_cell == 3: #TREE              
                    if sed_balance_annual_equiv < -0.5:
                        sed_stress = -0.15
                    elif sed_balance_annual_equiv >= -0.5 and sed_balance_annual_equiv < 0.5:
                        sed_stress = (sed_balance_annual_equiv*0.3)
                    elif sed_balance_annual_equiv >= 0.5 and sed_balance_annual_equiv < 2.0:
                        sed_stress = (sed_balance_annual_equiv*-0.1)+0.2
                    elif sed_balance_annual_equiv >= 2.0:
                        sed_stress = 0.              
            else: #Set sediment stress to zero if you don't want sediment balance to have an effect on veg growth
                sed_stress = 0           
            
            #------------------- Grazing stress -------------------------------             
            if grazing_event == 1: #If there is a grazing event            
                if veg_type_polled_cell == 1: #If cell is a grass, it can be grazed
                    if stocking_rate <= 0.06: #Below this value, the grazing stress varies with stocking rate (LSU per hectare)      
                        grazing_stress = stocking_rate*11
                    else: #Above the value, the land will have reached max degradation
                        grazing_stress = 0.66
                else:
                    grazing_stress = 0.01 #If cell is a shrub or tree, it can't be grazed, but can be trampled
            else: #No grazing occurs
                grazing_stress = 0.0
            
            #---------------------- Fire stress -------------------------------            
            if fire_event == 1: #If there is a fire
                if veg_type_polled_cell == 1: #There is a constant grass mortality for grasses
                    fire_stress = 0.66
                elif veg_type_polled_cell == 2:
                    fire_stress = 0.6*(math.exp(-0.01*age_polled_cell))
                elif veg_type_polled_cell == 3:
                    fire_stress = 0.4*(math.exp(-0.01*age_polled_cell)) 
            else: #If there is no fire
                fire_stress = 0
            
            #---------------------- Age stress --------------------------------
            if veg_type_polled_cell == 1:
                age_stress_current = age_stress_grass[age_polled_cell-1]
            elif veg_type_polled_cell == 2:
                age_stress_current = age_stress_shrub[age_polled_cell-1]
            elif veg_type_polled_cell == 3:
                age_stress_current = age_stress_tree[age_polled_cell-1]
            else:
                print('WARNING: No veg type detected in cell!')
                
            #----------- Probability of survival (i.e. Total stress) ----------
            prob_survival = 1 + (sed_stress - drought_stress - age_stress_current - grazing_stress - fire_stress)          
            
            if chance[R, C] >= prob_survival or walls_grid[R, C] > 0: #If plant dies, or if it's a wall, reset all grids to zero
                if recolonisation_dynamism == 'off': #Recolonisation occurs as a fixed proportion of grass/shrub/tree during all runs
                    if chance2[R, C] <= grass_proportion: #GRASS
                        next_veg_type[R, C] = 1 
                    elif chance2[R, C] <= (grass_proportion + shrub_proportion): #SHRUB
                        next_veg_type[R, C] = 2
                    elif chance2[R, C] <= (grass_proportion + shrub_proportion + tree_proportion): #TREE
                        next_veg_type[R, C] = 3
                    else:
                        print('WARNING: No veg type allocated to cell!')
                elif recolonisation_dynamism == 'on':
                    if (period_rainfall*(12/veg_update_freq_equivalent)) <= veg_dominance_rainfall: #For dry conditions
                        alpha_grass = np.random.normal(0.5 - ((period_rainfall*(12/veg_update_freq_equivalent))/1200), 0.1) #Multiplier to increase or decrease grass proportion depending on precipitation
                        alpha_shrub = np.random.normal(0.1 - ((period_rainfall*(12/veg_update_freq_equivalent))/3000), 0.1) #Multiplier to increase or decrease grass proportion depending on precipitation
                        alpha_tree = np.random.normal(-0.25 + ((period_rainfall*(12/veg_update_freq_equivalent))/2400), 0.1) #Multiplier to increase or decrease tree proportion depending on precipitation
                    elif (period_rainfall*(12/veg_update_freq_equivalent)) <= (veg_dominance_rainfall*2): #Below 500 mm/yr, grass dominates, from 500 mm/yr onwards, tree dominates
                        alpha_grass = np.random.normal(0.0, 0.1) #The multiplier can't be less than 0%
                        alpha_shrub = np.random.normal(-0.1, 0.1) #The multiplier can't be greater than 10% decrease
                        alpha_tree = np.random.normal(-0.25 + ((period_rainfall*(12/veg_update_freq_equivalent))/2400), 0.1) #Multiplier to increase or decrease tree proportion depending on precipitation 
                    elif (period_rainfall*(12/veg_update_freq_equivalent)) > (2*veg_dominance_rainfall): #For very moist conditions
                        alpha_grass = np.random.normal(0.0, 0.1) #The multiplier can't be greater than 25% decrease
                        alpha_shrub = np.random.normal(-0.1, 0.1) #The multiplier can't be greater than 10% decrease
                        alpha_tree = np.random.normal(0.25, 0.1) #The multiplier can't be greater than 25% increase
                    else:
                        print('WARNING: No alphas allocated to cells!')
                    #Add relevant alpha parameters to the current grass/shrub/tree proportions, to get the adapted probabilities
                    grass_recolonisation_prob = current_grass_proportion+alpha_grass #GRASS
                    if grass_recolonisation_prob < 0: #Probability can't be less than zero
                        grass_recolonisation_prob = 0
                    shrub_recolonisation_prob = current_shrub_proportion+alpha_shrub #SHRUB
                    if shrub_recolonisation_prob < 0:
                        shrub_recolonisation_prob = 0
                    tree_recolonisation_prob = current_tree_proportion+alpha_tree #TREE
                    if tree_recolonisation_prob < 0:
                        tree_recolonisation_prob = 0
                    #Attribute veg types to newly bare cell
                    total_probabilities_summed = grass_recolonisation_prob + shrub_recolonisation_prob + tree_recolonisation_prob #Sum of all probabilities (as they don't necessarily add to 1)
                    if total_probabilities_summed > 0: #As long as one plant has a probability greater than zero, can do this routine 
                        chance_number = np.random.uniform(0, total_probabilities_summed)
                        if chance_number <= grass_recolonisation_prob: #GRASS 
                            next_veg_type[R, C] = 1 
                        elif chance_number <= (grass_recolonisation_prob + shrub_recolonisation_prob): #SHRUB
                            next_veg_type[R, C] = 2
                        elif chance_number <= (grass_recolonisation_prob + shrub_recolonisation_prob + tree_recolonisation_prob): #TREE
                            next_veg_type[R, C] = 3
                        else:
                            print('WARNING: No veg type allocated to cell!')
                    else: #If no plant type has a chance above zero (e.g. can happen at zero rainfall)
                        chance_number = round(np.random.uniform(0.50000001, 3.4999999)) #Attribute veg types randomly equally (i.e. 1 in 3 chance)
                        if chance_number == 1: #GRASS 
                            next_veg_type[R, C] = 1 
                        elif chance_number == 2: #SHRUB
                            next_veg_type[R, C] = 2
                        elif chance_number == 3: #TREE
                            next_veg_type[R, C] = 3
                        else:
                            print('WARNING: No veg type allocated to cell!')
                else:
                    print('WARNING: Plant recolonisation dynamics are unspecified')
                update[R, C] = 0
                next_age[R, C] = veg_update_freq_equivalent
                cumulated_growth = veg_update_freq_equivalent #Need to give it the minimum cum_growth possible
                cum_growth_grid[R, C] = cumulated_growth #Put the adapted cumulated growth into the grid for next update                
                actual_biomass = veg_update_freq_equivalent
                next_biomass[R, C] = actual_biomass #Actual biomass is not the same as cumulated growth                  
                veg_grid[R, C] = 0. #Reset veg height to zero
                drought_grid[R, C] = 0 #Reset the drought grid, otherwise dead plants have no chance!
                trunks_grid[R, C] = 0 #Reset the trunks grid
                #Reset porosity
                if start_grid_porosity == -1:  
                    porosity_grid[R, C] = np.random.uniform(min_random_porosity, max_random_porosity)
                else:        
                    porosity_grid[R, C] = start_grid_porosity
            #Set the rooting height again, if this is a new plant
            if update[R, C] == 1:
                if veg_occupation_grid[R, C] == 0: #If it's a new plant (i.e. the same cell was empty at the last veg update)
                    rooting_heights_grid[R, C] = sand_heights_grid[R, C] #Set the new rooting height
                veg_occupation_grid[R, C] = 1 #Tell the grid that the cell is now occupied
            else:
                veg_occupation_grid[R, C] = 0 #Tell the grid that the cell is now unoccupied
                
    #Update all grids
    grid = copy.copy(update); age_grid = copy.copy(next_age); actual_biomass_grid = copy.copy(next_biomass); veg_type_grid = copy.copy(next_veg_type)
    #z1  = np.where(veg_grid>=veg_threshold) #Only count cells with veg height>threshold
    z1  = np.where(age_grid>=age_threshold) #Only count cells with age>threshold
    veg_population = sum(grid[z1])
    average_age_array[0] = (sum(age_grid[np.where(veg_type_grid == 1)]))/((veg_type_grid == 1).sum()+1); average_age_array[1] = (sum(age_grid[np.where(veg_type_grid == 2)]))/((veg_type_grid == 2).sum()+1); average_age_array[2] = (sum(age_grid[np.where(veg_type_grid == 3)]))/((veg_type_grid == 3).sum()+1) #Calculate average age of each plant type
        
    return (veg_grid, veg_type_grid, age_grid, cum_growth_grid, actual_biomass_grid, veg_occupation_grid, rooting_heights_grid, trunks_grid, porosity_grid, drought_grid, grid, interaction_field, veg_population, average_age_array, current_grass_proportion, current_shrub_proportion, current_tree_proportion)
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    
