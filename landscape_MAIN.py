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

from landscape_INCLUDE_list import * #Import the parameters from the all-inclusive list

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- SETUP *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
gc.collect() #Run garbage collector

#Define initial grids
if manual_initialisation == 'off': #Definition of grids according to parameters (using 'startgrids' functions)
    sand_heights_grid, veg_grid, veg_type_grid, age_grid, cum_growth_grid, actual_biomass_grid, veg_occupation_grid, drought_grid, grid, walls_grid, walls_presence_grid = startgrids()
elif manual_initialisation == 'on': #Definition of grids done manually (through 'manual_initialisation)
    sand_heights_grid, veg_grid, veg_type_grid, age_grid, cum_growth_grid, actual_biomass_grid, veg_occupation_grid, drought_grid, grid, walls_grid, walls_presence_grid = startgrids_manual() 

#Define initial grids for porosity, trunks, soil moisture
porosity_grid, trunks_grid, w_soil_moisture_grid = startgrids_2(veg_grid, veg_type_grid)

#Define initial grids for sediment balance calculations, apparent veg etc.
initial_sand_heights_grid, rooting_heights_grid, initial_veg_grid, initial_apparent_veg_type_grid, apparent_veg_type_grid, sed_balance_moisture_grid = startgrids_3(sand_heights_grid, veg_grid, veg_type_grid)

#Define arrays needed to fill with data
rainfall_days, total_sand_vol, total_aval_vol, total_veg_pop, average_age_table, veg_proportions, exposed_wall_proportions, differences_grid, avalanching_grid, windspeed_grid, interaction_field = startarrays()

#Define stress series (i.e. rainfall, windspeed, fire, grazing)
rainfall_series = define_rainfall_series() #Define rainfall time series for veg update routine
windspeed_dataset, wind_angle_dataset = define_wind_series() #Define wind speed and angle dataset
fire_series, grazing_series = define_fire_grazing_series() #Define fire and grazing events series

#Others
wind_t = 0; veg_t = 0 #To tick along how many times veg and wind have been updated
soil_moisture_presence = 24 #Reset the moisture presence to none (ie 24 hours at least since last rain)

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- START *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
start_time = time.time()
print("Starting...")
    
for t in range(model_iterations):
    print("Iteration number: ", t+1)
    gc.collect() # Run garbage collector
    
    #------------------------------ WIND EVENT? -------------------------------
    if (t+1) % wind_event_frequency == 0:
        #------------- GET WINDSPEED, ANGLE, RAINFALL DATA FROM SERIES --------
        unobstructed_windspeed = windspeed_dataset[wind_t];
        wind_angle = wind_angle_dataset[wind_t]; wind_angle = windangle_adaptor(wind_angle)
        rainfall_presence = rainfall_days[t] #Check whether rain fell during the previous timestep (important for sand movement routine)
        
        #----------------------- WIND ABOVE THRESHOLD? ------------------------
        if unobstructed_windspeed >= windspeed_threshold:   
            
            #------------------ CALCULATE WIND SPEED FOR EACH CELL ------------
            windspeed_grid, path_cell_location, path = windspeedcalculator(sand_heights_grid, veg_grid, porosity_grid, trunks_grid, rooting_heights_grid, walls_grid, wind_angle, unobstructed_windspeed)
            
            #------- WRAP SAND, VEG GRIDS FOR SAND TRANSPORT (Nw+2 x Nw+2) ----
            if boundary_conditions == 'periodic': #Depending on boundary conditions, wrap the differences_grid or just lose sand out of the edges        
                w_sand_heights_grid = wrap_grid(sand_heights_grid, 1)
                w_veg_grid = wrap_grid(veg_grid, 1)
                w_porosity_grid = wrap_grid(porosity_grid, 1)
                w_rooting_heights_grid = wrap_grid(rooting_heights_grid, 1)
                w_walls_grid = wrap_grid(walls_grid, 1)
            elif boundary_conditions == 'open':
                w_sand_heights_grid = wrap_grid_2(sand_heights_grid, 1)
                w_veg_grid = wrap_grid_2(veg_grid, 1)
                w_porosity_grid = wrap_grid_2(porosity_grid, 1)
                w_rooting_heights_grid = wrap_grid_2(rooting_heights_grid, 1)
                w_walls_grid = wrap_grid_2(walls_grid, 1)
            
            #------ CALCULATE DEPOSITION TRAJECTORIES FOR EACH TARGET CELL ----
            path_cell_location_depos, path_depos = depositiontrajectories(wind_angle)
               
            #-------------------------- SAND MOVEMENT -------------------------
            differences_grid, w_soil_moisture_grid, sed_balance_moisture_grid, soil_moisture_presence = sand_movement(w_sand_heights_grid, w_veg_grid, w_rooting_heights_grid, w_soil_moisture_grid, w_porosity_grid, w_walls_grid, windspeed_grid, sed_balance_moisture_grid, path_cell_location, path_cell_location_depos, path_depos, prob_depos_bare, prob_depos_sand, rainfall_presence, soil_moisture_presence, t)    
            if boundary_conditions == 'periodic': #Depending on boundary conditions, unwrap the differences_grid or just lose sand out of the edges     
                differences_grid = unwrap_grid(differences_grid, 1) #Unwrap the grid from (Nrw+2 x Ncw+2) to (Nr x Nc) 
            elif boundary_conditions == 'open':
                differences_grid = unwrap_grid_2(differences_grid, 1) #Only take the core of the differences grid (losing all the sand lost to the edge)
            sand_heights_grid += differences_grid #Add the height changes to original sand heights at start of interval                   
            
            #-------- WRAP SAND AND VEG GRIDS FOR AVALANCHING (Nrw x Ncw) -------
            w_sand_heights_grid = wrap_grid(sand_heights_grid, 0)
            w_veg_grid = wrap_grid(veg_grid, 0)
            w_rooting_heights_grid = wrap_grid(rooting_heights_grid, 0)
            w_walls_grid = wrap_grid(walls_grid, 0)            
            
            #--------------------------- AVALANCHING --------------------------
            avalanching_grid = avalanche(w_sand_heights_grid, w_veg_grid, w_rooting_heights_grid, w_walls_grid)
            if boundary_conditions == 'periodic': #Depending on boundary conditions, unwrap the avalanching grid or just lose sand out of the edges
                avalanching_grid = unwrap_grid(avalanching_grid, 0); avalanching_grid[np.isnan(avalanching_grid)] = 0 #Unwrap grid from (Nrw x Ncw) to (Nr x Nc)
            elif boundary_conditions == 'open':
                avalanching_grid = unwrap_grid_2(avalanching_grid, 0); avalanching_grid[np.isnan(avalanching_grid)] = 0
            sand_heights_grid += avalanching_grid #Add the height changes to modified sand heights
        
        wind_t += 1
        print("Wind event occurred. Windspeed =", round(unobstructed_windspeed, 2), ", Wind angle =", round(wind_angle, 2))
        
    #------------------------ UPDATE VEGETATION? ------------------------------
    if (t+1) % veg_update_frequency == 0:
        fire_event = fire_series[veg_t]; grazing_event = grazing_series[veg_t];
        veg_grid, veg_type_grid, age_grid, cum_growth_grid, actual_biomass_grid, veg_occupation_grid, rooting_heights_grid, trunks_grid, porosity_grid, drought_grid, grid, interaction_field, veg_population, average_age_array, current_grass_proportion, current_shrub_proportion, current_tree_proportion = veg_update(veg_grid, veg_type_grid, age_grid, sand_heights_grid, cum_growth_grid, actual_biomass_grid, veg_occupation_grid, rooting_heights_grid, trunks_grid, porosity_grid, drought_grid, grid, walls_grid, rainfall_series, veg_t, fire_event, grazing_event)
        total_veg_pop[veg_t] = veg_population
        average_age_table[veg_t, 0] = average_age_array[0]; average_age_table[veg_t, 1] = average_age_array[1]; average_age_table[veg_t, 2] = average_age_array[2]
        veg_proportions[veg_t, 0] = current_grass_proportion; veg_proportions[veg_t, 1] = current_shrub_proportion; veg_proportions[veg_t, 2] = current_tree_proportion        
        apparent_veg_type_grid = copy.copy(veg_type_grid); apparent_veg_type_grid[np.where(veg_grid == 0)] = 0 #It's APPARENT because all cells are attributed a potential veg type, but those that have zero height need to be seen as zero        
        veg_t += 1        
        print("Veg is updated. Veg density =", round(veg_population/(Nr*Nc), 2))
        
    #------------------------- STORE DATA TO GRID? ----------------------------
    if (t+1) % saving_frequency_movie == 0:
        saving_grid_sand, saving_grid_veg, saving_grid_apparent_veg_type, saving_grid_age, saving_grid_wind, saving_grid_moisture, saving_loc = savegrids(sand_heights_grid, veg_grid, apparent_veg_type_grid, age_grid, windspeed_grid, w_soil_moisture_grid, saving_loc, t)
    
    #------------------------ DO FINAL CALCULATIONS  --------------------------
    sum_diffs_grid = copy.copy(differences_grid); sum_diffs_grid[np.where(differences_grid <= 0)] = 0; sum_diffs_grid[np.isnan(sum_diffs_grid)] = 0; total_sand_vol[t] = sum(map(sum, sum_diffs_grid))*cell_width #Volume (m^3)  #Only check negative numbers so they don't cancel out
    sum_aval_events = copy.copy(avalanching_grid); sum_aval_events[np.where(avalanching_grid <= 0)] = 0; total_aval_vol[t] = sum(map(sum, sum_aval_events))*cell_width #Volume (m^3)
    sum_exposed_walls = (np.greater(walls_grid, sand_heights_grid))*1
    if (sum(map(sum, walls_presence_grid))) > 0:
        exposed_wall_proportions[t] = ((sum(map(sum, sum_exposed_walls)))/(sum(map(sum, walls_presence_grid)))) #Proportion (0 to 1) of walls that are covered by sand because sand has come over the top
    else:
        exposed_wall_proportions[t] = 0
    differences_grid = np.zeros((Nrw+2, Ncw+2)); avalanching_grid = np.zeros((Nrw, Ncw)) #Re-zero the differences and avalanching grid
         
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- WIND STOPS BLOWING *-*-*-*-*-*-*-*-*-*-*-*-*-*-
elapsed = time.time() - start_time
print("Total sand height START =", sum(map(sum, initial_sand_heights_grid))); print("Total sand height END =", sum(map(sum, sand_heights_grid))); print("Time elapsed =", elapsed, "seconds")

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- PLOTTING *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
plot_wind_figures(windspeed_dataset, total_sand_vol, total_aval_vol, initial_sand_heights_grid, sand_heights_grid, windspeed_grid, w_soil_moisture_grid)
plot_veg_figures(initial_veg_grid, veg_grid, initial_apparent_veg_type_grid, apparent_veg_type_grid, porosity_grid, age_grid, walls_grid, interaction_field, total_veg_pop, average_age_table, veg_proportions, rainfall_series, exposed_wall_proportions)
plt.show

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* DATA STORAGE *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
#np.savetxt('Frames_sand_height', saving_grid_sand, delimiter=',')
#np.savetxt('Frames_veg_height', saving_grid_veg, delimiter=',')
#np.savetxt('Frames_age', saving_grid_age, delimiter=',')
#np.savetxt('Frames_veg_type', saving_grid_apparent_veg_type, delimiter=',')
#np.savetxt('Frames_moisture', saving_grid_moisture, delimiter=',')
#np.savetxt('Final_neighbourhood_stress', interaction_field, delimiter=',')
np.savetxt('Final_veg_grid', veg_grid, delimiter=',')
np.savetxt('Final_age_grid', age_grid, delimiter=',')
np.savetxt('Final_veg_type_grid', apparent_veg_type_grid, delimiter=',')
#np.savetxt('Final_wind_grid', windspeed_grid, delimiter=',')
#np.savetxt('Final_sand_grid', sand_heights_grid, delimiter=',')
#np.savetxt('Wind_frames', saving_grid_wind, delimiter=',')
#np.savetxt('Initial_veg_grid', initial_veg_grid, delimiter=',')
#np.savetxt('Initial_sand_grid', initial_sand_heights_grid, delimiter=',')
#np.savetxt('Final_soil_moisture_grid', w_soil_moisture_grid[((2*n_shells)+1):(Nrw+1), ((2*n_shells)+1):(Ncw+1)], delimiter=',')
#np.savetxt('Population_density', (total_veg_pop/(Nr*Nc)), delimiter=',')
#np.savetxt('Veg_proportions', veg_proportions, delimiter=',')
#np.savetxt('Average_ages', average_age_table, delimiter=',')
#np.savetxt('Total_sand_vol', total_sand_vol, delimiter=',')
#np.savetxt('Total_aval_vol', total_aval_vol, delimiter=',')
#np.savetxt('Exposed_wall_proportions', exposed_wall_proportions, delimiter=',')
#np.savetxt('Rainfall_series', rainfall_series, delimiter=',')
#np.savetxt('Windspeed series', windspeed_dataset, delimiter=',')
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-