#-----------------------------------------------------
#BOIDS GUI CODE
#Last Updated: Jan. 19, 2024
#See README.md file for information
#-----------------------------------------------------


#-----------------------------------------------------
#IMPORT MODULES
import pygame
import pygame_widgets
from pygame_widgets.toggle import Toggle
from pygame_widgets.slider import Slider
import numpy as np
import os
import argparse
#-----------------------------------------------------


#----------------------------------------------
#USE PARSER FOR COMMAND LINE ARGUMENTS
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
        description='Python simulation of flock behaviors, i.e. Boids, with focus on graphical user interface and visuals using pygame.',
        epilog = 'For more information on this software, contact Jordan Ducatel at jfducatel@gmail.com.')
#----------------------------------------------


#----------------------------------------------
#set ranodm seed:
np.random.seed(1234)
#----------------------------------------------


#----------------------------------------------
#DEFINE FUNCTION:
#Define Triangle drawing function:
def draw_triangle(t_x0_rel, t_y0_rel, t_angle, t_color):
    '''
    Description:
        Draws a pygame polygon object that is a triangle with a given orientation.
    Input:
        t_x0, t_y0: relative center coordinates of triangle. Range between zero and one. dtype: int or float
        t_angle: Orientation angle for triangle. Units: rad. dtype: int/float
        t_color: RGB color of triangle. format: (R, G, B). dtype: tuple
    '''
    t_x0 = t_x0_rel*sim_var['width']
    t_y0 = t_y0_rel*sim_var['height']
    t_width = sim_var['t_size']*sim_var['width']
    t_height = 1.5 * t_width
    t_x4 = t_x0 - 0.3 * t_height * np.cos(t_angle)
    t_y4 = t_y0 - 0.3 * t_height * np.sin(t_angle)
    t_x1 = t_x4 - 0.5 * t_width * np.sin(t_angle)
    t_y1 = t_y4 + 0.5 * t_width * np.cos(t_angle)
    t_x2 = t_x4 + 0.5 * t_width * np.sin(t_angle)
    t_y2 = t_y4 - 0.5 * t_width * np.cos(t_angle)
    t_x3 = t_x0 + 0.7 * t_height * np.cos(t_angle)
    t_y3 = t_y0 + 0.7 * t_height * np.sin(t_angle)

    pygame.draw.polygon(screen, t_color, [[t_x1, t_y1], [t_x2, t_y2], [t_x3, t_y3]])
    return


#Define loop around edges function:
def loop_around_edges(t_x0, t_y0):
    '''
    Description:
        Update the input t_x0 and t_y0 variables to loop around the edges of the bouding box of the similation.
    Input:
        t_x0, t_y0: relative center coordinates of triangle. dtype: int or float
    Output:
        t_x0, t_y0: relative center coordinates of triangle. dtype: int or float        
    '''
    if t_x0 < (sim_var['bounding box scale'] - sim_var['loop around buffer']):
        t_x0 = 1 - t_x0 - sim_var['loop around correction']
    if t_x0 > 1 - (sim_var['bounding box scale'] - sim_var['loop around buffer']):
        t_x0 = 1 - t_x0 + sim_var['loop around correction']
    if t_y0 < (sim_var['bounding box scale'] - sim_var['loop around buffer']):
        t_y0 = 1 - t_y0 - sim_var['loop around correction']
    if t_y0 > 1 - (sim_var['bounding box scale'] - sim_var['loop around buffer']):
        t_y0 = 1 - t_y0 + sim_var['loop around correction']
    return t_x0, t_y0


#Define boids initialization
def initialize_boids(N):
    '''
    Description:
        Initialize an array of boids location, velocities and colors.
    Input:
        N: number of boids to initialize. dtype: int.
    Output:
        t_x0_i: Initial x position. unit: pix. dtype: 1D array
        t_y0_i: Initial y position. unit: pix. dtype: 1D array
        t_angle_i: Initial angle position. unit: rad. dtype: 1D array
        t_vx0_i: Initial x velocity. unit: pix / frame. dtype: 1D array
        t_vy0_i: Initial y velocity. unit: pix / frame. dtype: 1D array
        t_colors: Boids individual colors. element format: (R, G, B). dtype: 1D array
        case_study_original_color: Test study Boid color. format: (R, G, B). dtype: tupple
    '''
    t_x0_i = np.random.uniform(low=sim_var['bounding box scale'], high=1-sim_var['bounding box scale'], size=N)
    t_y0_i = np.random.uniform(low=sim_var['bounding box scale'], high=1-sim_var['bounding box scale'], size=N)
    t_angle_i = np.random.uniform(low=-np.pi, high=np.pi, size=N)
    t_vx0_i = sim_var['t_speed'] * np.cos(t_angle_i)
    t_vy0_i = sim_var['t_speed'] * np.sin(t_angle_i)
    t_colors = np.random.choice(['lightblue', 'blue', 'darkblue'], size=N)
    case_study_original_color = t_colors[0]
    return t_x0_i, t_y0_i, t_angle_i, t_vx0_i, t_vy0_i, t_colors, case_study_original_color
#----------------------------------------------



#----------------------------------------------
#DEFINE VARIABLES
#define colors dictionary:
color = {'red':(204, 0, 0),
         'darkred':(153, 0, 0),
         'darkdarkred':(102, 0, 0),
         'darkgray':(32, 32, 32),
         'white':(255, 255, 255),
         'lightgray':(128, 128, 128),
         'lightblue':(153, 204, 255),
         'blue':(51, 153, 255),
         'darkblue':(0, 102, 204),
         'darkgreen':(0, 204, 102)}



#Define Global variable for the simulation:
sim_var = {'width': 640,
           'height': 480,
           'fps': 30,
           'bounding box scale': 0.1,
           't_size': 0.01,
           't_speed': 0.2,
           'loop around buffer': 0.05,
           'collision strength': 0.1,
           'alignment strength': 0.04,
           'cohesion scale': 0.05,
           'number of triangles': 50,
           'controls box scale': 0.6,
           'toggle scale': 0.03,
           'loop around correction': 0.005}
#----------------------------------------------


#----------------------------------------------
#Setup pygame environment and windows and other variables
os.environ["SDL_VIDEO_CENTERED"]='1'

#Initialize pygame GUI:
pygame.init()
pygame.display.set_caption("Boid Simulation")

screen = pygame.display.set_mode((sim_var['width'] * (1+sim_var['controls box scale']), sim_var['height']))
clock = pygame.time.Clock()
#----------------------------------------------


#----------------------------------------------
#Initialize randomly triangles position and velocity:
N = sim_var['number of triangles']
t_x0_i, t_y0_i, t_angle_i, t_vx0_i, t_vy0_i, t_colors, case_study_original_color = initialize_boids(sim_var['number of triangles'])
#----------------------------------------------


#----------------------------------------------
#DEFINE TEXT AND CONTROLS
#Define text parameters for controls:
font_1 = pygame.font.Font('freesansbold.ttf', 32)
font_2 = pygame.font.Font('freesansbold.ttf', 20)

text_1 = font_1.render('Controls', True, color['white'], color['darkgray'])
text_2 = font_2.render('Boids #', True, color['white'], color['darkgray'])
text_3 = font_2.render('Play / Pause', True, color['white'], color['darkgray'])
text_4 = font_2.render('Case Study', True, color['white'], color['darkgray'])
text_5 = font_2.render('Nearest', True, color['white'], color['darkgray'])
text_6 = font_2.render('Edges', True, color['white'], color['darkgray'])
text_7 = font_2.render('Separation', True, color['white'], color['darkgray'])
text_8 = font_2.render('Alignment', True, color['white'], color['darkgray'])
text_9 = font_2.render('Cohesion', True, color['white'], color['darkgray'])

textRect_1 = text_1.get_rect()
textRect_2 = text_2.get_rect()
textRect_3 = text_3.get_rect()
textRect_4 = text_4.get_rect()
textRect_5 = text_5.get_rect()
textRect_6 = text_6.get_rect()
textRect_7 = text_7.get_rect()
textRect_8 = text_8.get_rect()
textRect_9 = text_9.get_rect()

textRect_1.center = (sim_var['width'] * (1 + (sim_var['controls box scale'] - sim_var['loop around buffer'])/2), 
                     sim_var['height'] * (2*sim_var['loop around buffer']))
textRect_2.center = (sim_var['width'] * (1 + (sim_var['controls box scale'] - sim_var['loop around buffer'])*1/4), 
                     sim_var['height'] * ((2+3*1)*sim_var['loop around buffer']))
textRect_3.center = (sim_var['width'] * (1 + (sim_var['controls box scale'] - sim_var['loop around buffer'])*3/4), 
                     sim_var['height'] * ((2+3*1)*sim_var['loop around buffer']))
textRect_4.center = (sim_var['width'] * (1 + (sim_var['controls box scale'] - sim_var['loop around buffer'])*1/4), 
                     sim_var['height'] * ((2+3*2)*sim_var['loop around buffer']))
textRect_5.center = (sim_var['width'] * (1 + (sim_var['controls box scale'] - sim_var['loop around buffer'])*3/4), 
                     sim_var['height'] * ((2+3*2)*sim_var['loop around buffer']))
textRect_6.center = (sim_var['width'] * (1 + (sim_var['controls box scale'] - sim_var['loop around buffer'])*1/4), 
                     sim_var['height'] * ((2+3*3)*sim_var['loop around buffer']))
textRect_7.center = (sim_var['width'] * (1 + (sim_var['controls box scale'] - sim_var['loop around buffer'])*3/4), 
                     sim_var['height'] * ((2+3*3)*sim_var['loop around buffer']))
textRect_8.center = (sim_var['width'] * (1 + (sim_var['controls box scale'] - sim_var['loop around buffer'])*1/4), 
                     sim_var['height'] * ((2+3*4)*sim_var['loop around buffer']))
textRect_9.center = (sim_var['width'] * (1 + (sim_var['controls box scale'] - sim_var['loop around buffer'])*3/4), 
                     sim_var['height'] * ((2+3*4)*sim_var['loop around buffer']))


#Add toggle:
toggle_3 = Toggle(screen, 
                  int(sim_var['width'] * (1 + (sim_var['controls box scale'] - sim_var['loop around buffer'])*3/4) - (sim_var['toggle scale'] * 2 * sim_var['width'])/2), 
                  int(sim_var['height'] * ((2+3*1+1)*sim_var['loop around buffer'])), 
                  int(sim_var['toggle scale'] * 2 * sim_var['width']),
                  int(sim_var['toggle scale'] * sim_var['height']),
                  startOn=True)

toggle_4 = Toggle(screen, 
                  int(sim_var['width'] * (1 + (sim_var['controls box scale'] - sim_var['loop around buffer'])*1/4) - (sim_var['toggle scale'] * 2 * sim_var['width'])/2), 
                  int(sim_var['height'] * ((2+3*2+1)*sim_var['loop around buffer'])), 
                  int(sim_var['toggle scale'] * 2 * sim_var['width']),
                  int(sim_var['toggle scale'] * sim_var['height']))
toggle_5 = Toggle(screen, 
                  int(sim_var['width'] * (1 + (sim_var['controls box scale'] - sim_var['loop around buffer'])*3/4) - (sim_var['toggle scale'] * 2 * sim_var['width'])/2), 
                  int(sim_var['height'] * ((2+3*2+1)*sim_var['loop around buffer'])), 
                  int(sim_var['toggle scale'] * 2 * sim_var['width']),
                  int(sim_var['toggle scale'] * sim_var['height']))


toggle_6 = Toggle(screen, 
                  int(sim_var['width'] * (1 + (sim_var['controls box scale'] - sim_var['loop around buffer'])*1/4) - (sim_var['toggle scale'] * 2 * sim_var['width'])/2), 
                  int(sim_var['height'] * ((2+3*3+1)*sim_var['loop around buffer'])), 
                  int(sim_var['toggle scale'] * 2 * sim_var['width']),
                  int(sim_var['toggle scale'] * sim_var['height']),
                  startOn=True)
toggle_7 = Toggle(screen, 
                  int(sim_var['width'] * (1 + (sim_var['controls box scale'] - sim_var['loop around buffer'])*3/4) - (sim_var['toggle scale'] * 2 * sim_var['width'])/2), 
                  int(sim_var['height'] * ((2+3*3+1)*sim_var['loop around buffer'])), 
                  int(sim_var['toggle scale'] * 2 * sim_var['width']),
                  int(sim_var['toggle scale'] * sim_var['height']),
                  startOn=True)

toggle_8 = Toggle(screen, 
                  int(sim_var['width'] * (1 + (sim_var['controls box scale'] - sim_var['loop around buffer'])*1/4) - (sim_var['toggle scale'] * 2 * sim_var['width'])/2), 
                  int(sim_var['height'] * ((2+3*4+1)*sim_var['loop around buffer'])), 
                  int(sim_var['toggle scale'] * 2 * sim_var['width']),
                  int(sim_var['toggle scale'] * sim_var['height']),
                  startOn=True)
toggle_9 = Toggle(screen, 
                  int(sim_var['width'] * (1 + (sim_var['controls box scale'] - sim_var['loop around buffer'])*3/4) - (sim_var['toggle scale'] * 2 * sim_var['width'])/2), 
                  int(sim_var['height'] * ((2+3*4+1)*sim_var['loop around buffer'])), 
                  int(sim_var['toggle scale'] * 2 * sim_var['width']),
                  int(sim_var['toggle scale'] * sim_var['height']),
                  startOn=True)

#Add slider for boid number:
slider_2 = Slider(screen,
                  int(sim_var['width'] * (1 + (sim_var['controls box scale'] - sim_var['loop around buffer'])*1/4) - (sim_var['toggle scale'] * 4 * sim_var['width'])/2), 
                  int(sim_var['height'] * ((2+3*1+1)*sim_var['loop around buffer'])), 
                  int(sim_var['toggle scale'] * 4 * sim_var['width']),
                  int(sim_var['toggle scale'] * sim_var['height']),
                  min=1, max=100, step=1, initial=50,
                  colour=(141, 185, 244),
                  handleColour=(26, 115, 232))

slider_2_value_old = slider_2.getValue()
#----------------------------------------------


#----------------------------------------------
#RUN PYGAME GUI:
run = True
while run:
    clock.tick(sim_var['fps']) #Define clock
    screen.fill(color['darkgray']) #Define background color
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            run = False

    #Reinitialize simulation with more / less triangles if slider value changes:
    if slider_2.getValue() != slider_2_value_old:
        N = slider_2.getValue()
        t_x0_i, t_y0_i, t_angle_i, t_vx0_i, t_vy0_i, t_colors, case_study_original_color = initialize_boids(N)
        slider_2_value_old = slider_2.getValue()
    
    
    #Draw box enclosure for simulation:
    pygame.draw.rect(screen, color['lightgray'], 
                     (sim_var['bounding box scale']*sim_var['width'], 
                      sim_var['bounding box scale']*sim_var['height'], 
                      (1-2*sim_var['bounding box scale'])*sim_var['width'], 
                      (1-2*sim_var['bounding box scale'])*sim_var['height']), 
                     width=3)
                     
    #Draw buffer box enclosure for loop around:
    pygame.draw.rect(screen, color['white'], 
                     ((sim_var['bounding box scale'] - sim_var['loop around buffer'])*sim_var['width'], 
                      (sim_var['bounding box scale'] - sim_var['loop around buffer'])*sim_var['height'], 
                      (1-2*(sim_var['bounding box scale'] - sim_var['loop around buffer']))*sim_var['width'], 
                      (1-2*(sim_var['bounding box scale'] - sim_var['loop around buffer']))*sim_var['height']), 
                     width=3)

    #Case study Toggle:
    if toggle_4.getValue() == False:
        t_colors[0] = case_study_original_color
        #If case study not enabled, disable nearest boid toggle:
        toggle_5.startOn = False
        toggle_5.disable()

    else:
        t_colors[0] = 'red' #case study selected
        #Togle is enabled for nearest:
        toggle_5.enable()


    
    #Find Closest triangle to case study:
    distance = np.sqrt((t_x0_i[0]*sim_var['width'] - t_x0_i*sim_var['width'])**2 + 
                       (t_y0_i[0]*sim_var['height'] - t_y0_i*sim_var['height'])**2)

    if N > 1:
        closest_index = np.argsort(distance)[1]
    else:
        closest_index = 0

    old_closest_color = t_colors[closest_index]
    if toggle_5.getValue() == True and toggle_4.getValue() == True:
        t_colors[closest_index] = 'darkgreen' #change triangle color
    
    #Loop over all triangles drawn for each frame:
    for ii in range(slider_2.getValue()):
        #Draw Triangle:
        draw_triangle(t_x0_rel=t_x0_i[ii], t_y0_rel=t_y0_i[ii], t_angle=t_angle_i[ii], t_color=color[t_colors[ii]])
    
        #Loop around edges:
        if toggle_6.getValue() == True:
            t_x0_i[ii], t_y0_i[ii] = loop_around_edges(t_x0=t_x0_i[ii], t_y0=t_y0_i[ii])

    #restore triangle color after being drawn:
    t_colors[closest_index] = old_closest_color


    #Take a step for triangles:
    #3 options: no change, counter clockwise step, clockwise step
    for ii in range(N):
        #Find closest triangle index:
        distance = np.sqrt((t_x0_i[ii]*sim_var['width'] - t_x0_i*sim_var['width'])**2 + 
                           (t_y0_i[ii]*sim_var['height'] - t_y0_i*sim_var['height'])**2)
        if N > 1:
            closest_index = np.argsort(distance)[1]
        else:
            closest_index = 0
        
        #Update angle to steer away or towards closest triangle:
        d_angle = sim_var['collision strength']
        
        t_x0_i_test_0 = t_x0_i[ii] + 1/sim_var['fps'] * t_vx0_i[ii]
        t_y0_i_test_0 = t_y0_i[ii] + 1/sim_var['fps'] * t_vy0_i[ii]
        distance_test_0 = np.sqrt((t_x0_i_test_0 - t_x0_i[closest_index])**2 + 
                                  (t_y0_i_test_0 - t_y0_i[closest_index])**2)
        t_x0_i_test_1 = t_x0_i[ii] + 1/sim_var['fps'] * (sim_var['t_speed'] * np.cos(t_angle_i[ii] + d_angle))
        t_y0_i_test_1 = t_y0_i[ii] + 1/sim_var['fps'] * (sim_var['t_speed'] * np.sin(t_angle_i[ii] + d_angle))
        distance_test_1 = np.sqrt((t_x0_i_test_1 - t_x0_i[closest_index])**2 + 
                                  (t_y0_i_test_1 - t_y0_i[closest_index])**2)
        t_x0_i_test_2 = t_x0_i[ii] + 1/sim_var['fps'] * (sim_var['t_speed'] * np.cos(t_angle_i[ii] - d_angle))
        t_y0_i_test_2 = t_y0_i[ii] + 1/sim_var['fps'] * (sim_var['t_speed'] * np.sin(t_angle_i[ii] - d_angle))
        distance_test_2 = np.sqrt((t_x0_i_test_2 - t_x0_i[closest_index])**2 + 
                                  (t_y0_i_test_2 - t_y0_i[closest_index])**2)
        distance_test = np.array([distance_test_0, distance_test_1, distance_test_2])
        angle_test = np.array([t_angle_i[ii], t_angle_i[ii] + d_angle, t_angle_i[ii] - d_angle])


        #Separation:
        if toggle_7.getValue() == True and toggle_9.getValue() == True:        
            #If below coherence scale: repel each other, if above it then atrack each other:
            if distance_test_0 <= sim_var['cohesion scale']:
                new_angle = angle_test[np.argmax(distance_test)]
            elif distance_test_0 > sim_var['cohesion scale']:
                new_angle = angle_test[np.argmin(distance_test)]
        
        elif toggle_7.getValue() == True and toggle_9.getValue() == False:        
            new_angle = angle_test[np.argmax(distance_test)]
        
        elif toggle_7.getValue() == False and toggle_9.getValue() == True:        
            if distance_test_0 > sim_var['cohesion scale']:
                new_angle = angle_test[np.argmin(distance_test)]
        
        elif toggle_7.getValue() == False and toggle_9.getValue() == False:
            new_angle = angle_test[0] #No change
            
        
        #Play / Pause Toggle:
        if toggle_3.getValue() == False:
            t_angle_i[ii] = new_angle


        
        #Update and steer in nearest triangle direction:
        d_angle_align = sim_var['alignment strength']
        angle_closest = t_angle_i[closest_index]
        diff_angle = angle_closest - t_angle_i[ii]
        
        #Play / Pause Toggle:
        if toggle_3.getValue() == False:
            #alignmnet:
            if toggle_8.getValue() == True:
                t_angle_i[ii] += d_angle_align * np.sign(diff_angle)
                
    
    #Play / Pause Toggle:
    if toggle_3.getValue() == False:
        #Update Triangle position (Can do array operations):
        t_x0_i += 1/sim_var['fps'] * t_vx0_i
        t_y0_i += 1/sim_var['fps'] * t_vy0_i

        #update triangle speed direction (Can do array operations):
        #t_angle_i += np.random.uniform(low=-0.1, high=0.1, size=N)
        t_vx0_i = sim_var['t_speed'] * np.cos(t_angle_i)
        t_vy0_i = sim_var['t_speed'] * np.sin(t_angle_i)


    #Define simulation controls:
    #Draw control box
    pygame.draw.rect(screen, color['white'], 
                     ((1)*sim_var['width'], 
                     (sim_var['loop around buffer'])*sim_var['height'], 
                     (sim_var['controls box scale'] - sim_var['loop around buffer'])*sim_var['width'], 
                     (1 - 2*sim_var['loop around buffer'])*sim_var['height']), 
                     width=3)

    #Add text in control box:
    screen.blit(text_1, textRect_1)
    screen.blit(text_2, textRect_2)
    screen.blit(text_3, textRect_3)
    screen.blit(text_4, textRect_4)
    screen.blit(text_5, textRect_5)
    screen.blit(text_6, textRect_6)
    screen.blit(text_7, textRect_7)
    screen.blit(text_8, textRect_8)
    screen.blit(text_9, textRect_9)


    pygame_widgets.update(events)
    pygame.display.update()

pygame.quit()
#----------------------------------------------




