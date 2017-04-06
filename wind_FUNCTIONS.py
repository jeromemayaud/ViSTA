import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import operator
import random
import time
import math
import copy
from decimal import Decimal

from landscape_SETUP import *

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- AVALANCHING *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

def avalanche(w_sand_heights_grid, w_veg_grid, w_rooting_heights_grid, w_walls_grid):
    #----------------------- Function description -----------------------------    
    #Performs the avalanching routine, to make sure there are no unrealistically steep slopes
    #'w_sand_heights_grid' is the wrapped sand_heights_grid (Nrw x Ncw)
    #'w_veg_grid' is the wrapped veg_grid (Nrw x Ncw)
    #'w_rooting_heights_grid' is the wrapped rooting_heights_grid (Nrw x Ncw)
    #'w_walls_grid' is the wrapped walls_grid (Nrw x Ncw)
    
    #-------------------------- Function setup --------------------------------    
    avalanching_grid = np.zeros((Nrw, Ncw)) #Blank avalanching grid to fill    
    w_surface_heights_grid = copy.copy(w_sand_heights_grid) #Surface heights grid taken from sand_heights_grid, to then adapt with walls if they are present
    
    #---------------------------------- Start ---------------------------------
    #Create two arrays of i and j to choose a cell to be randomly polled
    i_random = np.zeros((Nr*Nc)); j_random = np.zeros((Nr*Nc))
    for a in range(Nr):
        for b in range(Nc):
            i_random[a*Nc + b] = a        
    for a in range(Nr):
        for b in range(Nc):
            j_random[a*Nc + b] = b                
    n = np.random.permutation(Nr*Nc) #Randomise the order in which cells will be polled
    
    for a in range(Nrw):
        for b in range(Ncw):
            if w_walls_grid[a, b] > w_sand_heights_grid[a, b]: #If the wall is exposed, surface height is the height of the wall
                w_surface_heights_grid[a, b] = w_walls_grid[a, b]
    
    for q in range(Nr*Nc):
        #Determine cell to be polled, define i and j accordingly for each iteration   
        poll_location = n[q]; i = i_random[poll_location]; j = j_random[poll_location] 
        if w_sand_heights_grid[i+n_shells, j+n_shells] > 0 and w_walls_grid[i+n_shells, j+n_shells] <= 0: #Only perform avalanching routine if there is actually sand in the polled cell, and there is no wall in that cell
            #Define von Neuman neighbourhood, which will be used to avalanche
            a = w_surface_heights_grid[i+n_shells, j+n_shells] #This is the polled cell
            b = w_surface_heights_grid[i-1+n_shells, j+n_shells]
            c = w_surface_heights_grid[i+n_shells, j-1+n_shells]
            d = w_surface_heights_grid[i+n_shells, j+1+n_shells]
            f = w_surface_heights_grid[i+1+n_shells, j+n_shells]
            #Create a random number to determine what order you check neighbours for avalanching
            random_avalanching_order = np.random.permutation(4)         
            #Create a temporary empty neighbourhood to update        
            updated_neighbourhood = np.zeros((3,3))
            #Define an array of the 4 neighbours of the polled cell, in a random order
            neighbours_array = np.zeros(4)
            neighbours_array[np.where(random_avalanching_order == 0)] = b
            neighbours_array[np.where(random_avalanching_order == 1)] = c
            neighbours_array[np.where(random_avalanching_order == 2)] = d
            neighbours_array[np.where(random_avalanching_order == 3)] = f
            #Find minimum value and its index in the neighbours array
            min_neighbour_location, min_neighbour = min(enumerate(neighbours_array), key=operator.itemgetter(1))
            #Assign the value of the polled minimum neighbour for the calculation
            if min_neighbour_location == 0:
                min_neighbour_polled = neighbours_array[0]
            elif min_neighbour_location == 1:
                min_neighbour_polled = neighbours_array[1]
            elif min_neighbour_location == 2:
                min_neighbour_polled = neighbours_array[2]
            else:
                min_neighbour_polled = neighbours_array[3]
                    
            #If (exposed) veg exceeds a critical height in polled cell, angle of repose increases
            exposed_veg_height = (w_rooting_heights_grid[i+n_shells, j+n_shells] + w_veg_grid[i+n_shells, j+n_shells]) - a    
            if exposed_veg_height >= veg_effectiveness_threshold_crit_angle:
                crit_repose_temp = veg_crit_repose
            else:
                crit_repose_temp = crit_repose        
            
            #Find max height difference allowed for angle of repose not to be violated
            max_height_diff = abs(cell_width*(np.tan(np.radians(crit_repose_temp))))
            
            kk = 0 #Reset to zero to break the 'while' loop
            #While angle of repose is violated, avalanche the difference to the minimum neighbour           
            while (a - min_neighbour_polled) > max_height_diff: #Only perform avalanching if polled cell is higher than minimum neighbour          
                kk += 1
                if kk == max_neighbour_checks: #To break the while loop if too many neighbours are low
                    break
                #Define a slab to be removed - divide by two because it has to come off the polled cell and onto the lowest neighbour
                difference_slab = np.around(((a - min_neighbour_polled - max_height_diff)/2), 3) #Round to 3dp (0.1cm resolution)
                #Remove the difference slab from polled cell            
                updated_neighbourhood[1, 1] -= difference_slab
                
                #Create a 'temporary locator' of what cell in the random avalanching order corresponds to the minimum
                if min_neighbour_polled == neighbours_array[0]: 
                    temp_locator = random_avalanching_order[0]
                elif min_neighbour_polled == neighbours_array[1]: 
                    temp_locator = random_avalanching_order[1]
                elif min_neighbour_polled == neighbours_array[2]: 
                    temp_locator = random_avalanching_order[2]
                else:
                    temp_locator = random_avalanching_order[3]
                #Add the difference slab to the appropriate cell
                if temp_locator == 0:
                    updated_neighbourhood[0, 1] += difference_slab
                elif temp_locator == 1:
                    updated_neighbourhood[1, 0] += difference_slab
                elif temp_locator == 2:
                    updated_neighbourhood[1, 2] += difference_slab
                else:
                    updated_neighbourhood[2, 1] += difference_slab
    
                #Add updated neighbourhood to sand heights to mimic the avalanching already having happened
                if sum(map(sum, updated_neighbourhood)) != 0:  
                    updated_neighbourhood[1,1] -= sum(map(sum, updated_neighbourhood)) #Add any difference to target cell
                a = updated_neighbourhood[1, 1] + w_surface_heights_grid[i+n_shells, j+n_shells]
                b = updated_neighbourhood[0, 1] + w_surface_heights_grid[i-1+n_shells, j+n_shells]
                c = updated_neighbourhood[1, 0] + w_surface_heights_grid[i+n_shells, j-1+n_shells]
                d = updated_neighbourhood[1, 2] + w_surface_heights_grid[i+n_shells, j+1+n_shells]
                f = updated_neighbourhood[2, 1] + w_surface_heights_grid[i+1+n_shells, j+n_shells]
                
                #Define a new array of the 4 neighbours of the polled cell, in the same random order as above
                neighbours_array = np.zeros(4)
                neighbours_array[np.where(random_avalanching_order == 0)] = b
                neighbours_array[np.where(random_avalanching_order == 1)] = c
                neighbours_array[np.where(random_avalanching_order == 2)] = d
                neighbours_array[np.where(random_avalanching_order == 3)] = f                        
                #Find minimum value and its index in the neighbours array
                min_neighbour_location, min_neighbour = min(enumerate(neighbours_array), key=operator.itemgetter(1))
                #Assign the value of the polled minimum neighbour for the calculation
                if min_neighbour_location == 0:
                    min_neighbour_polled = neighbours_array[0]
                elif min_neighbour_location == 1:
                    min_neighbour_polled = neighbours_array[1]
                elif min_neighbour_location == 2:
                    min_neighbour_polled = neighbours_array[2]
                else:
                    min_neighbour_polled = neighbours_array[3]
            
            if sum(map(sum, updated_neighbourhood)) != 0:  
                updated_neighbourhood[1,1] -= sum(map(sum, updated_neighbourhood)) #Add any difference to target cell      
            avalanching_grid[i+n_shells,j+n_shells] += updated_neighbourhood[1,1]
            avalanching_grid[i-1+n_shells,j+n_shells] += updated_neighbourhood[0,1]
            avalanching_grid[i+n_shells,j-1+n_shells] += updated_neighbourhood[1,0]
            avalanching_grid[i+n_shells,j+1+n_shells] += updated_neighbourhood[1,2]
            avalanching_grid[i+1+n_shells,j+n_shells] += updated_neighbourhood[2,1]
        
    avalanching_sum = sum(map(sum, avalanching_grid))    
    #Verify whether there are any continuity errors in the avalanching due to rounding errors
    if avalanching_sum != 0:
        avalanching_grid[i+n_shells, j+n_shells] -= avalanching_sum #Add any difference to last target cell
    return (avalanching_grid)
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

#*-*-*-*-*-*-*--*-*-*- DEFINE STRESS TIME SERIES AND MODELS *-*-*-*-*-*-*-*-*-*

def define_wind_series():
    #----------------------- Function description -----------------------------
    #Create a time series of wind according to the specified parameters    
    
    #-------------------------- Function setup -------------------------------- 
    total_runs = wind_iterations + spinup_time
    windspeed_dataset = np.zeros(total_runs); windspeed_dataset[0:spinup_time] = spinup_wind_series
    wind_angle_dataset = np.resize(wind_angle_array, total_runs)
    spare = np.arange(spinup_time, total_runs)
    
    #---------------------------------- Start ---------------------------------
    # Define wind event time series
    if wind_event_timeseries == 'constant': #Constant wind at same speed
        windspeed_dataset[spinup_time:total_runs] = wind_event_constant
    elif wind_event_timeseries == 'trend': #Constant increase or decrease in wind speed
        wind_event_rate = (wind_event_end-wind_event_start)/wind_iterations
        windspeed_dataset[spinup_time:total_runs] = wind_event_start+(spare+1-spinup_time)*wind_event_rate
    elif wind_event_timeseries == 'v-shaped': #'V-shaped' wind series, going from high to low then back again (or vice versa)
        wind_event_rate = (wind_event_end-wind_event_start)/(0.5*wind_iterations)
        for spare in range(1, wind_iterations+1):
            if spare < (wind_iterations/2):
                windspeed_dataset[spare-1+spinup_time] = wind_event_start + spare*wind_event_rate
            else:
                windspeed_dataset[spare-1+spinup_time] = (wind_event_start+(wind_iterations/2)*wind_event_rate)-(spare-(wind_iterations/2))*wind_event_rate
    elif wind_event_timeseries == 'weibull': #Weibull distribution of wind speeds (realistic in deserts)
        for tr in range(wind_iterations):
            windspeed_dataset[tr] = round(weibull_scale_parameter*(np.random.weibull(weibull_shape_parameter)), 2) #These Weibull parameters are an average throughout the year
        
    #If the wind event time series has to be variable, and distributed such that only a few events are above-threshold
    if windspeed_variability == 'on':
        variability_additions = np.random.normal(0, windspeed_var_normal_stdev, wind_iterations) #Create an array of normally-distributed additions
        for tr in range(wind_iterations):
            new_windspeed = windspeed_dataset[tr] + variability_additions[tr]
            if new_windspeed < 0:
                new_windspeed = 0
            windspeed_dataset[tr] = new_windspeed
    
    return windspeed_dataset, wind_angle_dataset
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    
#*-*-*-*-*-*-*-*-*-*-*-*-*-* DEPOSITION TRAJECTORIES *-*-*-*-*-*-*-*-*-*-*-*-*-

def depositiontrajectories(wind_angle):
    #----------------------- Function description -----------------------------
    # Calculate the path along which sand being eroded from each cell will be deposited
    # 'wind_angle' is the wind angle entering the grid

    #-------------------------- Function setup -------------------------------- 
    Nrww = Nrw + 2 #Grid dimensions for rows
    Ncww = Ncw + 2 #Grid dimensions for columns
    section_lengths = np.zeros((Nrww, Ncww)) #lengths of each section of wind path leading to the target cell    
    path_depos = np.zeros((Nrww*Ncww, 3*(max(Nrww, Ncww)))) #Array of all the section lengths for each deposition path; 2*N because max cells a wind path can go through is √2N, so round up to 2 (or 3 for 180 degrees?!)  
    path_cell_location_depos = np.zeros((Nrww*Ncww, 3*(max(Nrww, Ncww)))) #Position of each cell in each deposition path leading to each target cell; 2*N because max cells a wind path can go through is √2N, so round up to 2 (or 3 for 180 degrees?!)  
    
    #---------------------------------- Start ---------------------------------
    #Reverse the wind angle, to get the trajectory FROM the target cell downwind, rather than TO it    
    if wind_angle < 180: 
        reverse_wind_angle = wind_angle + 180
    else:
        reverse_wind_angle = wind_angle - 180
    m = np.tan(np.radians(reverse_wind_angle)) #Tan of the reverse wind angle, in degrees
    if m < 0.001 and m > -0.001:
        m = 0
     
    for i in range(Nrww):
        for j in range(Ncww):
            target_row = i; target_col = j #The row/col for which wind is being calculated
            section_lengths = np.zeros((Nrww, Ncww))
            checked_cells = np.zeros((Nrww*Ncww*3, 2)) #Multiply by 3 because want one cell either side of the checked cell
            path_counter = -1       
            
            #--------------------- Define wind path intercept -----------------
            # 0deg is left-right; 90deg is top-bottom; 180deg is right-left; 270deg is bottom-top
            if reverse_wind_angle >= 0 and reverse_wind_angle < 90:
                start_r = 0; increment_r = 1; end_r = target_row+1
                start_c = 0; increment_c = 1; end_c = target_col+1
            elif reverse_wind_angle >= 90 and reverse_wind_angle < 180:
                start_r = 0; increment_r = 1; end_r = target_row+1
                start_c = Ncww-1; increment_c = -1; end_c = target_col-1
            elif reverse_wind_angle >= 180 and reverse_wind_angle < 270:
                start_r = Nrww-1; increment_r = -1; end_r = target_row-1
                start_c = Ncww-1; increment_c = -1; end_c = target_col-1
            elif reverse_wind_angle >= 270 and reverse_wind_angle <= 360:
                start_r = Nrww-1; increment_r = -1; end_r = target_row-1
                start_c = 0; increment_c = 1; end_c = target_col+1             
            
            intercept = target_row - (m*target_col) #Create the intercept for the wind path
            check_count = 0
            
            #Define which cells to look at for path (to limit how many cells are checked to only 1 either side of a potential row/column)            
            if (reverse_wind_angle < 45) or (reverse_wind_angle >= 135 and reverse_wind_angle < 225) or (reverse_wind_angle >= 315): #These angles have slope gradients (m) < ±1, so must be defined in terms of columns
                for rrr in range(start_c, end_c, increment_c): 
                    checked_row = int(m*rrr + intercept) #Equivalent to y = mx + c                
                    checked_cells[check_count, 0] = rrr; checked_cells[check_count+1, 0] = rrr; checked_cells[check_count+2, 0] = rrr          
                    checked_cells[check_count, 1] = checked_row; checked_cells[check_count+1, 1] = checked_row-1; checked_cells[check_count+2, 1] = checked_row+1
                    check_count += 3  
            else: #Any angles that have slope gradients (m) > ±1, so must be defined in terms of rows
                for rrr in range(start_r, end_r, increment_r): #Define which cells to look at for path (to limit how many cells are checked)           
                    checked_column = int((rrr - intercept)/m) #Equivalent to rearranged y = mx + c                
                    checked_cells[check_count, 1] = rrr; checked_cells[check_count+1, 1] = rrr; checked_cells[check_count+2, 1] = rrr          
                    checked_cells[check_count, 0] = checked_column; checked_cells[check_count+1, 0] = checked_column-1; checked_cells[check_count+2, 0] = checked_column+1
                    check_count += 3
            
            for rrr in range(check_count): #Calculate whether the chosen grid cells actually intersect with the wind path to the target cell
                r = checked_cells[rrr, 1]
                c = checked_cells[rrr, 0]
                
                if r >= 0 and r <= Nrww-1 and c >= 0 and c <= Ncww-1:  #To avoid referencing outside the grid                     
                    x1 = c-1; x2 = c; y1 = r-1; y2 = r   
                    if m!= 0:                                             
                        x1_star = (y1-intercept)/m; x2_star = (y2-intercept)/m
                        y1_star = m*x1 + intercept; y2_star = m*x2 + intercept
                    else: #For 0 deg and 180 deg, when m = 0
                        x1_star = 0; x2_star = 0
                        y1_star = intercept; y2_star = intercept
                        
                    if wind_angle > 360:
                        print("Warning: Angle > 360 degrees")
                    #Fill up section_lengths with Pythagoras calculation of wind path length going through each cell to reach target cell
                    if m > 0.001: #Positive gradient
                        if (x1_star < x1 and x2_star <= x1) or (x1_star >= x2 and x2_star > x2): #misses, bottom right and top right
                            section_lengths[r, c] = 0
                        elif (x1_star >= x1 and x2_star >= x2): #1A (see diagram)
                            test = math.sqrt(((x2-x1_star)**2) + ((y2_star-y1)**2))
                            if test >= min_trajectory: #check whether the path through the cell is significant enough to be counted
                                section_lengths[r, c] = test; path_counter += 1
                        elif (x2_star <= x2 and x1_star <= x1): #2A
                            test = math.sqrt(((x2_star-x1)**2) + ((y2-y1_star)**2))
                            if test >= min_trajectory:
                                section_lengths[r, c] = test; path_counter += 1
                        elif (x1_star > x1 and x2_star < x2): #3A
                            test = math.sqrt(((x1_star-x1)**2) + ((y2-y1)**2))
                            if test >= min_trajectory:
                                section_lengths[r, c] = test; path_counter += 1
                        elif (y1_star > y1 and y2_star < y2): #4A
                            test = math.sqrt(((x2-x1)**2) + ((y2_star-y1_star)**2))
                            if test >= min_trajectory:
                                section_lengths[r, c] = test; path_counter += 1
                    elif m < -0.001: #Negative gradient
                        if (x1_star <= x1 and x2_star <= x1) or (x1_star > x2 and x2_star >= x2): #misses, bottom left and top right
                            section_lengths[r, c] = 0
                        elif (x1_star <= x2 and x2_star <= x1): #1B (see diagram)
                            test = math.sqrt(((x1_star-x1)**2) + ((y1_star-y1)**2))
                            if test >= min_trajectory: #check whether the path through the cell is significant enough to be counted
                                section_lengths[r, c] = test; path_counter += 1
                        elif (x1_star >= x2 and x2_star >= x1): #2B
                            test = math.sqrt(((x2-x2_star)**2) + ((y2-y2_star)**2))
                            if test >= min_trajectory:
                                section_lengths[r, c] = test; path_counter += 1
                        elif (x1_star < x2 and x2_star > x1): #3B
                            test = math.sqrt(((x1_star-x2_star)**2) + ((y2-y1)**2))
                            if test >= min_trajectory:
                                section_lengths[r, c] = test; path_counter += 1
                        elif (y1_star < y2 and y2_star > y1): #4B
                            test = math.sqrt(((y1_star-y2_star)**2) + ((x2-x1)**2))
                            if test >= min_trajectory:
                                section_lengths[r, c] = test; path_counter += 1
                    else: #For when m is very low, i.e. 0deg and 180deg                   
                        if r == target_row: #If the rows are aligned, then count as path
                            section_lengths[r, c] = 1                            
                            path_counter += 1
                        else:
                            section_lengths[r, c] = 0
                    #Fill up 'path' with the lengths of each section of wind path leading away from the target cell (if minimum path resolution is exceeded)
                    if section_lengths[r, c] > 0:
                        path_depos[((target_row)*Ncww) + target_col, path_counter] = section_lengths[r, c]
                        path_cell_location_depos[((target_row)*Ncww) + target_col, path_counter] = (r*Ncww) + c #Give each path cell its locator along the path
                    
    return(path_cell_location_depos, path_depos)
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- 

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- SAND MOVEMENT *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

def sand_movement(w_sand_heights_grid, w_veg_grid, w_rooting_heights_grid, w_soil_moisture_grid, w_porosity_grid, w_walls_grid, windspeed_grid, sed_balance_moisture_grid, path_cell_location, path_cell_location_depos, path_depos, prob_depos_bare, prob_depos_sand, rainfall_presence, soil_moisture_presence, t):
    #----------------------- Function description -----------------------------
    #'w_sand_heights_grid' is wrapped sand heights grid (Nrw+2 x Ncw+2)
    #'w_veg_grid' is wrapped veg grid (Nrw+2 x Ncw+2)
    #'w_rooting_heights_grid' is wrapped rooting heights grid (Nrw+2 x Ncw+2)
    #'w_soil_moisture_grid' is wrapped soil moisture grid (Nrw+2 x Ncw+2)
    #'w_porosity_grid' is wrapped porosity grid (Nrw+2 x Ncw+2)
    #'windspeed_grid' is the matrix of windspeeds for that time step, taken from the windspeedcalculator function (Nr x Nc)
    #'sed_balance_moisture_grid' is a matrix of the sediment balance (i.e. differences grid) from previous iteration to calculate sediment feedback function (Nrw+2 x Ncw+2)
    #'path_cell_location' is the matrix of path cell locations for the windspeed trajectories, taken from the windspeedcalculator function (Nr x Nc)
    #'path_cell_location_depos' is the matrix of path cell locations for the deposition trajectories, taken from the depositiontrajectories function (Nrw+2 x Ncw+2)
    #'path_depos' is the sector lengths leading to each cell in the deposition trajectories, taken from the depositiontrajectories function (Nrw+2 x Ncw+2)
    #'prob_depos_bare' is probability of deposition on bare rock
    #'prob_depos_sand' is probability of deposition on a sandy surface
    #'rainfall_presence' is presence (0 or 1) of rainfall in previous timestep, as this affects sediment movement if it occurred not long ago
    #'soil_moisture_presence' is whether soil is moist from previous timestep, as this affects sediment movement
    #'t' is the iteration number

    #-------------------------- Function setup --------------------------------    
    w_sand_heights_reshaped = np.reshape(w_sand_heights_grid, np.size(w_sand_heights_grid)) #Reshaped matrix of sand heights
    w_veg_reshaped = np.reshape(w_veg_grid, np.size(w_veg_grid)) #Reshaped matrix of wrapped veg heights
    w_rooting_reshaped = np.reshape(w_rooting_heights_grid, np.size(w_rooting_heights_grid)) #Reshaped matrix of wrapped rooting heights
    w_soil_moisture_reshaped = np.reshape(w_soil_moisture_grid, np.size(w_soil_moisture_grid)) #Reshaped matrix of soil mosture
    w_porosity_reshaped = np.reshape(w_porosity_grid, np.size(w_porosity_grid)) #Reshaped matrix of plant porosities
    w_walls_reshaped = np.reshape(w_walls_grid, np.size(w_walls_grid)) #Reshaped matrix of wall heights
    w_surface_heights = np.zeros(np.size(w_sand_heights_grid)) #Matrix to fill with surface heights (i.e. sand heights or exposed walls)  
    w_walls_presence = np.zeros(np.size(w_sand_heights_grid)) #Matrix to fill with '1's where a wall exists
    sed_balance_moisture_reshaped = np.reshape(sed_balance_moisture_grid, np.size(sed_balance_moisture_grid)) #Reshaped matrix of moisture for sediment balance routine
    differences_array = np.zeros((Nrw+2)*(Ncw+2)) #Blank matrix to fill with changes in sand height
    shadow_zones_array = np.zeros((Nrw+2)*(Ncw+2)) #Blank matrix to fill with non-zeros if the cell is in a shadow zone
    
    #---------------------------------- Start ---------------------------------
    #----------- Update soil moisture based on evaporation/infiltration -------
    if rainfall_presence == 1 or (t+1) % moisture_update_frequency == 0: #If it's just rained, or if the moisture update is scheduled to occur
        if rainfall_presence == 1: #If it rained during the interval
            w_soil_moisture_reshaped = np.ones(np.size(w_soil_moisture_grid)) #Make all grid cells have a moisture effectiveness of 1 immediately after rain
            soil_moisture_presence = moisture_update_frequency_equivalent #Tell the next iteration that the time since rain starts from this point
        else: #If it didn't rain during the last interval
            if soil_moisture_presence < 24: #Still within the first 24 hours after rain, so perform feedback
                w_soil_moisture_reshaped += sed_balance_moisture_reshaped #Add or remove any sediment balance feedback
                w_soil_moisture_reshaped[np.where(w_soil_moisture_reshaped > 1.)] = 1.
                w_soil_moisture_reshaped -= (moisture_update_frequency_equivalent/24) #Remove moisture effectiveness as a function of time since rain
                w_soil_moisture_reshaped[np.where(w_soil_moisture_reshaped < 0.)] = 0.
                soil_moisture_presence += moisture_update_frequency_equivalent
            else: #More than 24 hours since last rain
                w_soil_moisture_reshaped = np.zeros(np.size(w_soil_moisture_grid))
                soil_moisture_presence = 24
        print('rainfall presence = ', rainfall_presence)
    
    for r in range(np.size(w_surface_heights)): #Fill array of ground surface height for the shadow zone routine below
        if w_walls_reshaped[r] > w_sand_heights_reshaped[r]: #If the wall is exposed, surface height is the wall
            w_surface_heights[r] = w_walls_reshaped[r]
            w_walls_presence[r] = 1 #Record the presence of some kind of wall in this cell, for deposition process
        else:
            w_surface_heights[r] = w_sand_heights_reshaped[r] #If there is no wall or it is buried, surface height is the sand height
   
    #----------------------- Calculate shadow zones ---------------------------    
    for r in range(Nr): 
        for c in range(Nc):
            current_target_cell_index = ((r + n_shells + 1)*(Ncw+2)) + c + n_shells + 1
            number_path_cells = np.size((np.where(path_depos[current_target_cell_index, :] > 0.))) #Pinpoints location of polled cell
            temp_path_cell_location = np.zeros(number_path_cells) #To fill with location of each downwind cell, in the reshaped array
            for z in range(number_path_cells): #Fill in locations of path cells
                temp_path_cell_location[z] = path_cell_location_depos[current_target_cell_index, number_path_cells - z - 2]
            z = 0
            #Find all cells downwind of target cell that are in its shadow zone
            if number_path_cells > 0:      
                while (w_surface_heights[current_target_cell_index] - w_surface_heights[temp_path_cell_location[z]])/(cell_width*(z+1)) > np.tan(np.radians(shadow_zone)):
                    shadow_zones_array[temp_path_cell_location[z]] += 1 #Make cells within the shadow zone non-zero
                    if z == number_path_cells-1: #Break if shadow zone reaches edge of wrapped grid
                        break
                    z += 1

    #--------- Calculate flux (q) and height change at each cell --------------  
    #Create two arrays of i and j to choose a cell to be randomly polled
    i_random = np.zeros((Nr*Nc)); j_random = np.zeros((Nr*Nc))
    for a in range(Nr):
        for b in range(Nc):
            i_random[a*Nc + b] = a        
    for a in range(Nr):
        for b in range(Nc):
            j_random[a*Nc + b] = b                
    n = np.random.permutation(Nr*Nc) #Randomise the order in which cells will be polled
    for q in range(Nr*Nc):
        #Determine cell to be polled, define i and j accordingly for each iteration   
        poll_location = n[q]; r = i_random[poll_location]; c = j_random[poll_location]
        current_target_cell_index = ((r + n_shells + 1)*(Ncw+2)) + c + n_shells + 1
        cell_windspeed = windspeed_grid[r, c] #Get windspeed for polled cell
        
        #--------------------- Erosion/deposition probabilities ---------------
        cell_moisture = w_soil_moisture_reshaped[current_target_cell_index] #Get moisture effectiveness for polled cell
        if cell_moisture < 0:
            cell_moisture = 0
        eros_multiplier = 1 - cell_moisture #The number by which to multiply the erosion from a cell
        exposed_wall_height = w_walls_reshaped[current_target_cell_index] - w_sand_heights_reshaped[current_target_cell_index]       
        
        # ----------------------------- Erosion -------------------------------
        #Only erode if target cell is NOT IN SHADOW ZONE and NOT TOO MOIST and A WALL ISN'T EXPOSED
        if shadow_zones_array[current_target_cell_index] == 0 and eros_multiplier > 0 and exposed_wall_height <= 0:      
            if cell_windspeed >= windspeed_threshold: #Flux only above threshold
                cell_flux = eros_multiplier*(0.00354*((1-(windspeed_threshold/cell_windspeed))**2)*(1.25/9.81)*(cell_windspeed**3)) #Dong et al.(2003) formulation for flux
            else: 
                cell_flux = 0
            #Only perform movement/deposition if flux from the cell is greater than zero
            if cell_flux > 0:
                #Change in height represented by the mass flux; multiply by 60 because flux formula is in kg/m/s
                dh = (((cell_flux*wind_resolution*60.)/sand_density)/cell_width)                               
                if dh >= w_sand_heights_reshaped[current_target_cell_index] and w_sand_heights_reshaped[current_target_cell_index] > 0: #Can only erode as much sediment as there is available in the cell
                    dh = w_sand_heights_reshaped[current_target_cell_index]
                elif w_sand_heights_reshaped[current_target_cell_index] <= 0: #Adjust sand heights to keep >= 0
                    dh = 0
                if dh < 0:
                    print("WARNING: dh is less than zero")
                differences_array[current_target_cell_index] -= dh #Remove sand height from target cell
                
                # -------------------------- Deposition -----------------------
                if dh > 0: #Only do deposition routine if sand has left the polled cell
                    prob_depos_array = np.zeros(n_shells) #To fill with deposition probabilities
                    number_path_cells = np.size((np.where(path_depos[current_target_cell_index, :] > 0.))) #Pinpoints location of polled cell
                    temp_path_cell_location = np.zeros(n_shells) #To fill with location of each downwind cell, in the reshaped array
                    temp_path_lengths = np.zeros(n_shells) #To fill with path lengths going through each cell downwind of polled cell
                    a = 0
                    for z in range(n_shells):
                        temp_path_cell_location[z] = path_cell_location_depos[current_target_cell_index, number_path_cells - z - 2]
                        temp_path_lengths[z] = path_depos[current_target_cell_index, number_path_cells - z - 2]
                    for z in range(n_shells):
                        veg_porosity = w_porosity_reshaped[temp_path_cell_location[z]]
                        exposed_veg_height = (w_rooting_reshaped[temp_path_cell_location[z]] + w_veg_reshaped[temp_path_cell_location[z]]) - w_sand_heights_reshaped[temp_path_cell_location[z]]
                        current_cell_moisture = w_soil_moisture_reshaped[temp_path_cell_location[z]]
                        
                        if z == n_shells-1: #If cell is last in the transport limit, full deposition occurs
                            prob_depos = 1. 
                        elif w_walls_presence[temp_path_cell_location[z+1]] > 0: #If next cell has a wall, deposit sand in this cell
                            prob_depos = 1.
                        elif shadow_zones_array[temp_path_cell_location[z]] > 0: #If cell is in shadow zone, full deposition occurs
                            prob_depos = 1.
                        elif w_sand_heights_reshaped[temp_path_cell_location[z]] < 0.001: #If downwind cell is bare rock (rounding error means sand heights are never quite 0, so make limit 1 mm)                                                         
                            #Moisture effects                                
                            if current_cell_moisture < 0.3: #Minimal moisture doesn't change the probability of deposition
                                prob_depos_bare_temp = prob_depos_bare
                            elif current_cell_moisture < 0.8: #Intermediate deposition results in rebound effect
                                prob_depos_bare_temp = 0.1
                            else: #If it's really moist, full deposition occurs
                                prob_depos_bare_temp = 1.0
                            #Vegetation effects  
                            if exposed_veg_height > veg_effectiveness_threshold_prob_depos: #If veg is exposed
                                prob_depos_bare_temp += (1-prob_depos_bare)*((100-veg_porosity)/100) #Account for veg effectiveness at trapping  
                            if prob_depos_bare_temp > 1: #Can't let probability of deposition be greater than 1
                                prob_depos_bare_temp = 1.
                            prob_depos = prob_depos_bare_temp*(temp_path_lengths[z]); #Make proportional to how much of path is going through a cell   
                        else: #If downwind cell is sand
                            #Moisture effects                                
                            if current_cell_moisture < 0.3: #Minimal moisture doesn't change the probability of deposition
                                prob_depos_sand_temp = prob_depos_sand
                            elif current_cell_moisture < 0.8: #Intermediate deposition results in rebound effect
                                prob_depos_sand_temp = 0.3
                            else: #If it's really moist, full deposition occurs
                                prob_depos_sand_temp = 1.0
                            #Vegetation effects  
                            if exposed_veg_height > veg_effectiveness_threshold_prob_depos: #If veg is exposed
                                prob_depos_sand_temp += (1-prob_depos_sand)*((100-veg_porosity)/100) #Account for veg effectiveness at trapping
                            if prob_depos_sand_temp > 1: #Can't let probability of deposition be greater than 1
                                prob_depos_sand_temp = 1.
                            prob_depos = prob_depos_sand_temp*(temp_path_lengths[z]) #Make proportional to how much of path is going through a cell
                        #Calculate cumulative probability of being deposited at cell 'z'
                        b = (prob_depos - (a*prob_depos))
                        if b < 0:
                            b = 0
                        if b > 1:
                            b = 1
                        a += b; prob_depos_array[z] = b #Put the appropriate proportion of the cumulative probability into array
                
                    check = -dh #Check the summing of neighbourhood distribution
                    for z in range(n_shells):
                        if check < 0: #Only keep depositing if there's still sand left over
                            if z == n_shells-1:
                                differences_array[temp_path_cell_location[z]] -= check; check -= check
                            else:
                                dh_in = dh*prob_depos_array[z] #Divide up the change in height from the original upwind cell into proportional depositions for each downwind cell
                                if (check + dh_in) <= 0:
                                    differences_array[temp_path_cell_location[z]] += dh_in #Move sand to appropriate downwind cell
                                    check += dh_in
                                else:
                                    differences_array[temp_path_cell_location[z]] -= check; check -= check

    #Reshape differences and soil moisture grids to return them to main module
    differences_grid = np.reshape(differences_array, (Nrw+2, Ncw+2))            
    w_soil_moisture_grid = np.reshape(w_soil_moisture_reshaped, (Nrw+2, Ncw+2))    
    if (sum(map(sum, differences_grid)) > 0.001) or (sum(map(sum, differences_grid)) < -0.001):
        print("WARNING: Differences do not add up to zero:", (sum(map(sum, differences_grid))))
     
    #Create the sediment balance grid for moisture feedback
    if (t+1) % moisture_update_frequency == 0:
        for r in range(Nr): 
            for c in range(Nc):
                current_target_cell_index = ((r + n_shells + 1)*(Nrw+2)) + c + n_shells + 1
                sed_bal = differences_array[current_target_cell_index]
                if sed_bal < -0.01: 
                    sed_balance_moisture_reshaped[current_target_cell_index] = 0.55
                elif sed_bal < 0:
                    sed_balance_moisture_reshaped[current_target_cell_index] = (-0.3*sed_bal) + 0.25
                elif sed_bal < 0.01:
                    sed_balance_moisture_reshaped[current_target_cell_index] = (-2.5*sed_bal) + 0.25
                else:
                    sed_balance_moisture_reshaped[current_target_cell_index] = -1
    
    sed_balance_moisture_grid = np.reshape(sed_balance_moisture_reshaped, (Nrw+2, Ncw+2))

    return (differences_grid, w_soil_moisture_grid, sed_balance_moisture_grid, soil_moisture_presence)
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- UNWRAP GRID #1 -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

def unwrap_grid(input_grid, extra_margin):
    #----------------------- Function description -----------------------------
    #Adds the values in the right-hand margin of the larger (input) grid to the left-hand side of the smaller grid, left to right, top to bottom, bottom to top
    #'input_grid' is the grid that must be unwrapped
    #'extra_margin' is any extra margin added on top of n_shells, per side (i.e. '1' if input grid's dimensions are Nrw+2 x Ncw+2)
    
    #-------------------------- Function setup --------------------------------   
    unwrapped_grid = np.zeros((Nr, Nc))
    unwrapped_grid = input_grid[grid_start+extra_margin:grid_end_r+extra_margin, grid_start+extra_margin:grid_end_c+extra_margin]
    
    #---------------------------------- Start ---------------------------------
    #Top
    unwrapped_grid[0:n_shells+extra_margin, 0:Nc] += input_grid[grid_end_r+extra_margin:grid_end_r+n_shells+2*extra_margin, n_shells+extra_margin:grid_end_c+extra_margin]
    #Bottom
    unwrapped_grid[Nr-n_shells-extra_margin:Nr, 0:Nc] += input_grid[0:n_shells+extra_margin, grid_start+extra_margin:grid_end_c+extra_margin]
    #Right
    unwrapped_grid[0:Nr, Nc-n_shells-extra_margin:Nc] += input_grid[grid_start+extra_margin:grid_end_r+extra_margin, 0:n_shells+extra_margin]
    #Left
    unwrapped_grid[0:Nr, 0:n_shells+extra_margin] += input_grid[grid_start+extra_margin:grid_end_r+extra_margin, grid_end_c+extra_margin:grid_end_c+n_shells+2*extra_margin]
    #Top left
    unwrapped_grid[0:n_shells+extra_margin, 0:n_shells+extra_margin] += input_grid[grid_end_r+extra_margin:grid_end_r+n_shells+2*extra_margin, grid_end_c+extra_margin:grid_end_c+n_shells+2*extra_margin]
    #Top right
    unwrapped_grid[0:n_shells+extra_margin, Nc-n_shells-extra_margin:Nc] += input_grid[grid_end_r+extra_margin:grid_end_r+n_shells+2*extra_margin, 0:n_shells+extra_margin]
    #Bottom left
    unwrapped_grid[Nr-n_shells-extra_margin:Nr, 0:n_shells+extra_margin] += input_grid[0:n_shells+extra_margin, grid_end_c+extra_margin:grid_end_c+n_shells+2*extra_margin]
    #Bottom right
    unwrapped_grid[Nr-n_shells-extra_margin:Nr, Nc-n_shells-extra_margin:Nc] += input_grid[0:n_shells+extra_margin, 0:n_shells+extra_margin]
    
    return(unwrapped_grid)
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* UNWRAP GRID #2 -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

def unwrap_grid_2(input_grid, extra_margin):
    #----------------------- Function description -----------------------------
    #Alternative unwrapping (for open boundary conditions) that lets sediment be lost outside the system
    #'input_grid' is the grid that must be unwrapped
    #'extra_margin' is any extra margin added on top of n_shells, per side (i.e. '1' if input grid's dimensions are Nrw+2 x Ncw+2)
    
    #---------------------------------- Start ---------------------------------
    unwrapped_grid = input_grid[grid_start+extra_margin:grid_end_r+extra_margin, grid_start+extra_margin:grid_end_c+extra_margin]
    
    return(unwrapped_grid)
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- WRAP GRID #1 *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

def wrap_grid(input_grid, extra_margin):
    #----------------------- Function description -----------------------------
    #Produces 'wrapping borders' around the input grid, of width=n_shells
    #'input_grid' is the grid that must be wrapped
    #'extra_margin' is any extra margin added on top of n_shells, per side (i.e. '1' if input grid's dimensions are Nrw+2 x Ncw+2)
    
    #-------------------------- Function setup --------------------------------     
    wrapped_grid = np.zeros((Nrw+(2*extra_margin), Ncw+(2*extra_margin)))
    wrapped_grid[grid_start+extra_margin:grid_end_r+extra_margin, grid_start+extra_margin:grid_end_c+extra_margin] = input_grid
    
    #---------------------------------- Start ---------------------------------
    #Bottom
    wrapped_grid[grid_end_r+extra_margin:grid_end_r+n_shells+2*extra_margin, grid_start+extra_margin:grid_end_c+extra_margin] = input_grid[0:n_shells+extra_margin, 0:Nc]
    #Top
    wrapped_grid[0:n_shells+extra_margin, grid_start+extra_margin:grid_end_c+extra_margin] = input_grid[Nr-n_shells-extra_margin:Nr, 0:Nc]
    #Left
    wrapped_grid[grid_start+extra_margin:grid_end_r+extra_margin, 0:n_shells+extra_margin] = input_grid[0:Nr, Nc-n_shells-extra_margin:Nc]
    #Right
    wrapped_grid[grid_start+extra_margin:grid_end_r+extra_margin, grid_end_c+extra_margin:grid_end_c+n_shells+2*extra_margin] = input_grid[0:Nr, 0:n_shells+extra_margin]
    #Top left corner
    wrapped_grid[0:n_shells+extra_margin, 0:n_shells+extra_margin] = input_grid[Nr-n_shells-extra_margin:Nr, Nc-n_shells-extra_margin:Nc]
    #Top right corner
    wrapped_grid[0:n_shells+extra_margin, grid_end_c+extra_margin:grid_end_c+n_shells+2*extra_margin] = input_grid[Nr-n_shells-extra_margin:Nr, 0:n_shells+extra_margin]
    #Bottom left corner
    wrapped_grid[grid_end_r+extra_margin:grid_end_r+n_shells+2*extra_margin, 0:n_shells+extra_margin] = input_grid[0:n_shells+extra_margin, Nc-n_shells-extra_margin:Nc]
    #Bottom right corner
    wrapped_grid[grid_end_r+extra_margin:grid_end_r+n_shells+2*extra_margin, grid_end_c+extra_margin:grid_end_c+n_shells+2*extra_margin] = input_grid[0:n_shells+extra_margin,0:n_shells+extra_margin]
    
    return(wrapped_grid)
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- WRAP GRID #2 *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

def wrap_grid_2(input_grid, extra_margin):
    #----------------------- Function description -----------------------------
    #Alternative wrapping (for open boundary conditions)
    #'input_grid' is the grid that must be wrapped
    #'extra_margin' is any extra margin added on top of n_shells, per side (i.e. '1' if input grid's dimensions are Nrw+2 x Ncw+2)
    
    #-------------------------- Function setup --------------------------------     
    wrapped_grid = np.zeros((Nrw+(2*extra_margin), Ncw+(2*extra_margin)))
    wrapped_grid[grid_start+extra_margin:grid_end_r+extra_margin, grid_start+extra_margin:grid_end_c+extra_margin] = input_grid
    
    #---------------------------------- Start ---------------------------------
    #Bottom
    wrapped_grid[grid_end_r+extra_margin:grid_end_r+n_shells+2*extra_margin, grid_start+extra_margin:grid_end_c+extra_margin] = np.zeros((n_shells+extra_margin, Nc))
    #Top
    wrapped_grid[0:n_shells+extra_margin, grid_start+extra_margin:grid_end_c+extra_margin] = np.zeros((n_shells+extra_margin, Nc))
    #Left
    wrapped_grid[grid_start+extra_margin:grid_end_r+extra_margin, 0:n_shells+extra_margin] = np.zeros((Nr, n_shells+extra_margin))
    #Right
    wrapped_grid[grid_start+extra_margin:grid_end_r+extra_margin, grid_end_c+extra_margin:grid_end_c+n_shells+2*extra_margin] = np.zeros((Nr, n_shells+extra_margin))
    #Top left corner
    wrapped_grid[0:n_shells+extra_margin, 0:n_shells+extra_margin] = np.zeros((n_shells+extra_margin, n_shells+extra_margin))
    #Top right corner
    wrapped_grid[0:n_shells+extra_margin, grid_end_c+extra_margin:grid_end_c+n_shells+2*extra_margin] = np.zeros((n_shells+extra_margin, n_shells+extra_margin))
    #Bottom left corner
    wrapped_grid[grid_end_r+extra_margin:grid_end_r+n_shells+2*extra_margin, 0:n_shells+extra_margin] = np.zeros((n_shells+extra_margin, n_shells+extra_margin))
    #Bottom right corner
    wrapped_grid[grid_end_r+extra_margin:grid_end_r+n_shells+2*extra_margin, grid_end_c+extra_margin:grid_end_c+n_shells+2*extra_margin] = np.zeros((n_shells+extra_margin, n_shells+extra_margin))
    
    return(wrapped_grid)
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- WIND ANGLE ADAPTOR *-*-*-*-*-*-*-*-*-*-*-*-*-

def windangle_adaptor(wind_angle):
    #----------------------- Function description -----------------------------
    #Make sure that wind approach angles of 90 degree and 270 degrees are slightly changed, because tan(90) = infinity

    #---------------------------------- Start ---------------------------------    
    if wind_angle == 90 or wind_angle == 270:
        wind_angle += 0.00000001
    if wind_angle == 360:
        wind_angle -= 0.00000001
        
    return(wind_angle)
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

#*-*-*-*-*-*-*-*-*-*-*-*-*- WIND SPEED CALCULATOR *-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    
def windspeedcalculator(sand_heights_grid, veg_grid, porosity_grid, trunks_grid, rooting_heights_grid, walls_grid, wind_angle, unobstructed_windspeed): 
    #----------------------- Function description -----------------------------
    #Calculates the wind speed over the entire grid at each timestep
    # 'sand_heights_grid' is the grid of sand heights (Nr x Nc)
    # 'veg_grid' is the vertical height of plants (Nr x Nc)
    # 'porosity_grid' is the porosity of plants (Nr x Nc)
    # 'trunks_grid' is the vertical height of trunks under each plant (Nr x Nc)
    # 'rooting_heights_grid' is the rooting heights of each plant (Nr x Nc)
    # 'walls_grid' is the heights of walls in the domain (Nr x Nc)    
    # 'wind_angle' is the wind angle entering the grid
    # 'unobstructed_windspeed' is the unobstructed windspeed coming onto the domain for this iteration

    #-------------------------- Function setup --------------------------------     
    section_lengths = np.zeros((Nr, Nc)) #lengths of each section of wind path leading to the target cell
    path = np.zeros((Nr*Nc, 3*(max(Nr, Nc)))) #Array of all the section lengths for each path; 2*N because max cells a wind path can go through is √2N, so round up to 2
    path_cell_location = np.zeros((Nr*Nc, 3*(max(Nr, Nc)))) #Position of each cell in each path leading to each target cell; 2*N because max cells a wind path can go through is √2N, so round up to 2       
    windspeed_grid = np.zeros((Nr, Nc)) #Grid of windspeed for each cell that is passed through to main module
    surface_windspeed = np.zeros(3*(max(Nr, Nc))) #Array for the actual surface windspeeds (e.g. affected by vegetation)
    L = np.zeros(3*(max(Nr, Nc))) #To fill with lengths of each cell in path, at each iteration
    g = np.zeros(3*(max(Nr, Nc))) #To fill with sand heights of each cell in path, at each iteration
    h = np.zeros(3*(max(Nr, Nc))) #To fill with veg heights of each cell in path, at each iteration
    p = np.zeros(3*(max(Nr, Nc))) #To fill with veg porosities of each cell in path, at each iteration
    tr = np.zeros(3*(max(Nr, Nc))) #To fill with trunk heights of each cell in path, at each iteration
    root = np.zeros(3*(max(Nr, Nc))) #To fill with rooting heights of each cell in path, at each iteration
    exp_wall = np.zeros(3*(max(Nr, Nc))) #To fill with exposed wall heights of each cell in path, at each iteration
    G = np.reshape(sand_heights_grid, np.size(sand_heights_grid)) #Reshaped matrix of sand heights
    H = np.reshape(veg_grid, np.size(veg_grid)) #Reshaped matrix of veg heights
    P = np.reshape(porosity_grid, np.size(porosity_grid)) #Reshaped matrix of veg heights
    TR = np.reshape(trunks_grid, np.size(trunks_grid)) #Reshaped matrix of trunk heights
    ROOT = np.reshape(rooting_heights_grid, np.size(rooting_heights_grid)) #Reshaped matrix of rooting heights
    WALL = np.reshape(walls_grid, np.size(walls_grid)) #Reshaped matrix of wall heights
    normal_dist_multiplier = np.random.normal(0,normal_dist_stdev, Nr*Nc) #Create array of standard(?) normal distribution, to probabilistically determine the windspeed at each timestep
    m = np.tan(np.radians(wind_angle)) #Tan of the wind angle, in degrees
    if m < 0.001 and m > -0.001: #If m is really small, make it zero, because otherwise divisions get messy
        m = 0
        
    #---------------------------------- Start ---------------------------------
    for i in range(Nr):
        for j in range(Nc):
            target_row = i; target_col = j #The row/col for which wind is being calculated
            section_lengths = np.zeros((Nr, Nc))
            checked_cells = np.zeros((Nr*Nc*3, 2)) #Multiply by 3 because want one cell either side of the checked cell
            path_counter = -1        
    
            #------------------------- Define wind path intercept -------------
            # 0deg is wind from left to right; 90deg is top-bottom; 180deg is right-left; 270deg is bottom-top
            if wind_angle >= 0 and wind_angle < 90:
                start_r = 0; increment_r = 1; end_r = target_row+1
                start_c = 0; increment_c = 1; end_c = target_col+1
            elif wind_angle >= 90 and wind_angle < 180:
                start_r = 0; increment_r = 1; end_r = target_row+1
                start_c = Nc-1; increment_c = -1; end_c = target_col-1
            elif wind_angle >= 180 and wind_angle < 270:
                start_r = Nr-1; increment_r = -1; end_r = target_row-1
                start_c = Nc-1; increment_c = -1; end_c = target_col-1
            elif wind_angle >= 270 and wind_angle <= 360:
                start_r = Nr-1; increment_r = -1; end_r = target_row-1
                start_c = 0; increment_c = 1; end_c = target_col+1
            
            intercept = target_row - (m*target_col) #Create the intercept for the wind path
            check_count = 0 #Reset to zero
            
            #Define which cells to look at for path (to limit how many cells are checked to only 1 either side of a potential row/column)            
            if (wind_angle < 45) or (wind_angle >= 135 and wind_angle < 225) or (wind_angle >= 315): #These angles have slope gradients (m) < ±1, so must be defined in terms of columns
                for rrr in range(start_c, end_c, increment_c): 
                    checked_row = int(m*rrr + intercept) #Equivalent to y = mx + c                
                    checked_cells[check_count, 0] = rrr; checked_cells[check_count+1, 0] = rrr; checked_cells[check_count+2, 0] = rrr          
                    checked_cells[check_count, 1] = checked_row; checked_cells[check_count+1, 1] = checked_row-1; checked_cells[check_count+2, 1] = checked_row+1
                    check_count += 3  
            else: #Any angles that have slope gradients (m) > ±1, so must be defined in terms of rows
                for rrr in range(start_r, end_r, increment_r): #Define which cells to look at for path (to limit how many cells are checked)           
                    checked_column = int((rrr - intercept)/m) #Equivalent to rearranged y = mx + c                
                    checked_cells[check_count, 1] = rrr; checked_cells[check_count+1, 1] = rrr; checked_cells[check_count+2, 1] = rrr          
                    checked_cells[check_count, 0] = checked_column; checked_cells[check_count+1, 0] = checked_column-1; checked_cells[check_count+2, 0] = checked_column+1
                    check_count += 3
            
            for rrr in range(check_count): #Calculate whether the chosen grid cells actually intersect with the wind path to the target cell
                r = checked_cells[rrr, 1]
                c = checked_cells[rrr, 0]
                
                if r >= 0 and r <= Nr-1 and c >= 0 and c <= Nc-1:  #To avoid referencing outside the grid
                    x1 = c-1; x2 = c; y1 = r-1; y2 = r                        
                    if m!= 0:                                             
                        x1_star = (y1-intercept)/m; x2_star = (y2-intercept)/m
                        y1_star = m*x1 + intercept; y2_star = m*x2 + intercept
                    else: #For 0 deg and 180 deg, when m = 0
                        x1_star = 0; x2_star = 0
                        y1_star = intercept; y2_star = intercept
                    
                    if wind_angle > 360:
                        print("WARNING: Angle > 360 degrees")
                    #Fill up section_lengths with Pythagoras calculation of wind path length going through each cell to reach target cell
                    if m > 0.001: #Positive gradient
                        if (x1_star < x1 and x2_star <= x1) or (x1_star >= x2 and x2_star > x2): #misses, bottom right and top right
                            section_lengths[r, c] = 0
                        elif (x1_star >= x1 and x2_star >= x2): #1A (see diagram)
                            test = math.sqrt(((x2-x1_star)**2) + ((y2_star-y1)**2))
                            if test >= min_trajectory: #check whether the path through the cell is significant enough to be counted
                                section_lengths[r, c] = test; path_counter += 1
                        elif (x2_star <= x2 and x1_star <= x1): #2A
                            test = math.sqrt(((x2_star-x1)**2) + ((y2-y1_star)**2))
                            if test >= min_trajectory:
                                section_lengths[r, c] = test; path_counter += 1
                        elif (x1_star > x1 and x2_star < x2): #3A
                            test = math.sqrt(((x1_star-x1)**2) + ((y2-y1)**2))
                            if test >= min_trajectory:
                                section_lengths[r, c] = test; path_counter += 1
                        elif (y1_star > y1 and y2_star < y2): #4A
                            test = math.sqrt(((x2-x1)**2) + ((y2_star-y1_star)**2))
                            if test >= min_trajectory:
                                section_lengths[r, c] = test; path_counter += 1
                    elif m < -0.001: #Negative gradient
                        if (x1_star <= x1 and x2_star <= x1) or (x1_star > x2 and x2_star >= x2): #misses, bottom left and top right
                            section_lengths[r, c] = 0
                        elif (x1_star <= x2 and x2_star <= x1): #1B (see diagram)
                            test = math.sqrt(((x1_star-x1)**2) + ((y1_star-y1)**2))
                            if test >= min_trajectory: #check whether the path through the cell is significant enough to be counted
                                section_lengths[r, c] = test; path_counter += 1
                        elif (x1_star >= x2 and x2_star >= x1): #2B
                            test = math.sqrt(((x2-x2_star)**2) + ((y2-y2_star)**2))
                            if test >= min_trajectory:
                                section_lengths[r, c] = test; path_counter += 1
                        elif (x1_star < x2 and x2_star > x1): #3B
                            test = math.sqrt(((x1_star-x2_star)**2) + ((y2-y1)**2))
                            if test >= min_trajectory:
                                section_lengths[r, c] = test; path_counter += 1
                        elif (y1_star < y2 and y2_star > y1): #4B
                            test = math.sqrt(((y1_star-y2_star)**2) + ((x2-x1)**2))
                            if test >= min_trajectory:
                                section_lengths[r, c] = test; path_counter += 1
                    else: #For when m is very low, i.e. 0deg and 180deg
                        if r == target_row: #If the rows are aligned, then count as path
                            section_lengths[r, c] = 1                            
                            path_counter += 1
                        else:
                            section_lengths[r, c] = 0
                    #Fill up 'path' with the lengths of each section of wind path leading to the target cell (if minimum path resolution is exceeded)   
                    if section_lengths[r, c] > 0:
                        path[((target_row)*Nc) + target_col, path_counter] = section_lengths[r, c]
                        path_cell_location[((target_row)*Nc) + target_col, path_counter] = (r*Nc) + c #Give each path cell its locator along the path                        

    #--------------------------- Windspeed calculation ------------------------   
    for r in range(Nr):
        for c in range(Nc):
            current_target_cell_index = (r*Nc) + c #Assign index of the target cell
            number_path_cells = np.size(np.where(path[current_target_cell_index,:] >= min_trajectory)) #Number of cells in the path to target cell           
            exp_wall = np.zeros(3*(max(Nr, Nc))) #To fill with exposed wall heights of each cell in path, at each iteration             
            if number_path_cells > 0:
                for k in range (number_path_cells):
                    cell_number = path_cell_location[current_target_cell_index, k] #Number of cell along the path
                    L[k] = path[current_target_cell_index, k] #Path length through each cell
                    g[k] = G[cell_number] #Height of sand in each cell in path
                    p[k] = P[cell_number] #Porosity of veg in each cell in path
                    tr[k] = TR[cell_number] - g[k] #Height of exposed trunk in each cell in path
                    root[k] = ROOT[cell_number] #Rooting heights in each cell in path                 
                    if H[cell_number] > 0: #If there's actually veg in the first place, do calculation
                        h[k] = (H[cell_number] + root[k]) - g[k] #Height of exposed vegetation obstacle in each cell in path
                        if h[k] < 0: #Exposed vegetation can't be negative
                            h[k] = 0
                    else:
                        h[k] = 0 #If no plant, exposed height is zero
         
                    if WALL[cell_number] > 0: #If there is a wall present...
                        exp_wall[k] = WALL[cell_number] - g[k] #Height of exposed wall 
                        if exp_wall[k] < 0:                        
                            exp_wall[k] = 0 #If wall is buried under sand (i.e. exposed height is negative), then 
                        g[k] += exp_wall[k] #Apparent surface height is the sand height plus any exposed wall
                        
                #---------------- Windspeed over pathway ----------------------                    
                cum_veg_counter = 0. #Resets the path length of cells of veg the path encounters
                cum_veg_absence_counter = 0.   #Resets the path length of cells since path encountered veg
                cum_veg_porosity = 100 #Resets the cumulated apparent porosity along path 
                nearest_veg_height = 0. #Resets the value of nearest veg height                
                lee_minimum_windspeed = 0 #Reset minimum wind speed in the lee of a plant
                               
                for k in range (number_path_cells):  
                    
                    #-------------- Airflow compression from slope ------------
                    if k == 0: #No slope to consider for first cell
                        ground_slope = 0
                    else:
                        ground_slope = ((g[k] - g[k-1])) #Difference in height compared to upwind cell                    
                    #Work out potential compression change depending on slope direction
                    if ground_slope > 0: #If ground is sloping upwards
                        kg = (ground_slope*(max_air_compression_change/max_slope))/100
                    elif ground_slope < 0: #If ground is sloping downwards
                        kg = (ground_slope*(min_air_compression_change/min_slope))/100
                    else: #If there is no slope
                        kg = 0      
                    #Alter to limit the change due to compression
                    if kg > max_air_compression_change:
                        kg = max_air_compression_change
                    elif kg < min_air_compression_change:
                        kg = min_air_compression_change
                    
                    #----------------- Effect of vegetation -------------------
                    if k == 0: #First cell in path
                        if windspeed_stochasticity == 'on':
                            surface_windspeed[0] = unobstructed_windspeed + normal_dist_multiplier[current_target_cell_index]*unobstructed_windspeed #Set first cell as unobstructed windspeed
                        else:
                            surface_windspeed[0] = unobstructed_windspeed #Set first cell as unobstructed windspeed
                    else: #Rest of path                 
                        if h[k] > min_veg_height_for_wind_effect or exp_wall[k] > 0: #If there IS VEG OR A WALL in polled cell... (treat walls like very dense plants)
                            if cum_veg_counter == 0: #If there isn't a plant in the cell directly upwind, cum_veg_porosity resets
                                cum_veg_porosity = 100
                            cum_veg_counter += 1
                            cum_veg_absence_counter = 0 #Reset the counter of absence of veg
                            if h[k] > min_veg_height_for_wind_effect: #This is a PLANT
                                nearest_veg_height = copy.copy(h[k]) #Store the veg height as the nearest veg height in case there is no more veg
                                nearest_trunk_height = copy.copy(tr[k])                             
                                veg_porosity = p[k]
                                exposed_trunk = tr[k]
                                exposed_wall = 0 #Becase this isn't a wall
                            else: #This is a WALL (not a plant)
                                nearest_veg_height = copy.copy(exp_wall[k]) #Store the wall height as the nearest wall height in case there is no more wall
                                nearest_trunk_height = 0                             
                                veg_porosity = 0 #Wall has zero porosity
                                exposed_trunk = 0
                                exposed_wall = exp_wall[k]
                            if ground_slope != 0: #Ground slopes upwards or downwards
                                temp_surface_windspeed = (surface_windspeed[k-1])*kg
                            if exposed_trunk > trunk_limit: #Consider it a tree (because exposed trunk is large enough)
                                if cum_veg_counter == 1: #Only increase the windspeed as soon as wind hits tree, then keep the same under the tree canopy
                                    tree_speedup_factor = 0.4*math.exp(-(((exposed_trunk-1.5)**2)/1.13))*(1-(1/(1+math.exp(-2*(-4.2))))) + 1 
                                    temp_surface_windspeed = tree_speedup_factor*surface_windspeed[k-1] #Windspeed underneath tree (fixed value)               
                                else:
                                    temp_surface_windspeed = surface_windspeed[k-1]
                            elif exposed_wall > 0: #Consider it a wall
                                plant_slowdown_factor = 0 #Windspeed just stops
                                temp_surface_windspeed = plant_slowdown_factor*surface_windspeed[k-1]
                            else: #Consider it a shrub/grass
                                cum_veg_porosity = ((cum_veg_porosity/100)*veg_porosity)*1.5 
                                if cum_veg_porosity > veg_porosity:
                                    cum_veg_porosity = veg_porosity
                                plant_slowdown_factor = ((0.0146*cum_veg_porosity)-0.4076)
                                if plant_slowdown_factor > 1: #Prevent it from going >1 for v high porosities
                                    plant_slowdown_factor = 1
                                if plant_slowdown_factor < 0:
                                    plant_slowdown_factor = 0
                                temp_surface_windspeed = plant_slowdown_factor*surface_windspeed[k-1] #Windspeed THROUGH plant (fixed value, not linear decline)                               
                            
                            if windspeed_stochasticity == 'on':  
                                temp_surface_windspeed += normal_dist_multiplier[current_target_cell_index]*temp_surface_windspeed
                            lee_minimum_windspeed = temp_surface_windspeed #Keep updating minimum wind speed in lee until there is no more veg
                        
                        else: #If there is NO veg in polled cell
                            cum_veg_absence_counter += L[k]
                            cum_veg_counter = 0 #Reset the counter of veg
                            
                            if nearest_veg_height == 0: #No veg upwind
                                temp_surface_windspeed = unobstructed_windspeed
                                if ground_slope != 0: #Ground slopes upwards or downwards
                                    temp_surface_windspeed += (surface_windspeed[k-1])*kg
                                    
                            else: #Veg present upwind
                                hd = (cum_veg_absence_counter*cell_width)/nearest_veg_height #To convert into downwind heights   
                                if hd < max_downwind_height: #If hd is less than specified maximum hd for wind to be considered unobstructed                               
                                    if nearest_trunk_height > trunk_limit: #Consider it a tree 
                                        temp_surface_windspeed = unobstructed_windspeed*(0.4*math.exp(-(((nearest_trunk_height-1.5)**2)/1.13))*(1-(1/(1+math.exp(-2*(hd-4.2))))) + 1)                                    
                                    elif exposed_wall > 0: #Consider it a wall
                                        temp_surface_windspeed = 0 + (unobstructed_windspeed-0)*(1-math.exp(-(0.0105*0 + 0.1627)*hd)) #Normalised wind speed in recovery curve
                                    else: #Consider it a grass/shrub                                   
                                        temp_surface_windspeed = lee_minimum_windspeed + (unobstructed_windspeed-lee_minimum_windspeed)*(1-math.exp(-(0.0105*cum_veg_porosity + 0.1627)*hd)) #Normalised wind speed in recovery curve
                                else: #hd is past length where plant still has an aerodynamic effect
                                    temp_surface_windspeed = unobstructed_windspeed
                                if ground_slope != 0: #Ground slopes upwards or downwards
                                    temp_surface_windspeed += (surface_windspeed[k-1])*kg
                            
                            if windspeed_stochasticity == 'on': 
                                temp_surface_windspeed += normal_dist_multiplier[current_target_cell_index]*temp_surface_windspeed #Add stochasticity
                        
                        if temp_surface_windspeed > unobstructed_windspeed*2: #Limit wind speed
                            temp_surface_windspeed = unobstructed_windspeed*2
                        if temp_surface_windspeed <= 0:
                            temp_surface_windspeed = 0.000001 #To avoid division errors                 
                        
                        surface_windspeed[k] = temp_surface_windspeed 
                        
                windspeed_grid[r, c] = surface_windspeed[number_path_cells-1] #Record windspeed for final cell along the path (i.e. the target cell)      
                
            else:
                if windspeed_stochasticity == 'on': 
                    windspeed_grid[r, c] = unobstructed_windspeed + normal_dist_multiplier[current_target_cell_index]*unobstructed_windspeed
                else:
                    windspeed_grid[r, c] = unobstructed_windspeed
               
    return(windspeed_grid, path_cell_location, path)
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-