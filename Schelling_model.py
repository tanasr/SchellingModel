"""
COURSE: Particle Methods
PROF.: Igor Pivkin

ASSIGNMENT 1 - Schelling Agent-Based-Model
@author: Radenko Tanasic

Schelling model: instructions
    Create a grid given a width and height, and occupy randomly with agents
    Here:
        100x100 grid, 2 distinct agent types, distribution of agents: 50/50
            10% empty cells
            45% red agents
            45% blue agents
        threshold for neighbours to be happy: 4
"""

import numpy as np
from matplotlib import pyplot as plt
import itertools
import time
import sys


class SchellingABM:
    def __init__(self, grid_width, grid_height, empty_ratio, nr_races, H, iterations):
        self.width = int(grid_width)
        self.height = int(grid_height)
        self.empty_ratio = empty_ratio
        self.nr_races = int(nr_races)
        self.neighbours = int(H)
        self.iterations = int(iterations)


    # assign vectors for each individ. race, append, shuffle and reshape to matrix
    def set_up_agents(self):
        # Note: agents[x][y] --> x are rows, y are columns (not xy-coord!)
        self.amount_empty = int(self.width*self.height*self.empty_ratio)
        self.amount_agents = int(self.width*self.height*(1-self.empty_ratio))
        # empty_cells = np.zeros(self.amount_empty,dtype=int)
        
        # append agents to the empty list to generate self.agents
        self.agents = np.zeros(self.amount_empty,dtype=int)
        for i in range(1,self.nr_races+1):
            temp = i*np.ones(int(self.amount_agents/self.nr_races), dtype=int)
            self.agents = np.append(self.agents,temp)
        np.random.shuffle(self.agents)
        self.agents = np.reshape(self.agents,(self.height,self.width))
        # create copy for final plot
        self.agents_init = np.copy(self.agents)
    
    
    def count_neighbours(self, row, col) -> int:
        '''
        Description
        ----------
        Iterates through each cell and checks first, if this cell is empty. If so,
        it returns 0. If the cell is not empty, then for each agent of the same
        type in the neighbourhood, a counter called 'same' will increase by 1.
        If the counter is < the threshold for happiness, it will return the 
        value -1 for being unhappy, and 1 for being happy otherwise.
        For periodic boundary conditions the modulus is used to implement a wrap-
        around. Python's indexing [-1] retrieves the last element of a vector, 
        thus already a warp-around on left and top boundaries (where x=0 and y=0)
                
        Parameters
        ----------
        row : int
            rows of the matrix, up-and-down movement.
        col : int
            columns of the matrix, lef-and-right movement.

        Returns
        -------
        int
            0,-1 or 1.

        ''' 
        
        centre = self.agents[row,col]
        same = int(0)
        if centre != 0:
            if centre == self.agents[row,col-1]: #left boundary
                same += 1
            if centre == self.agents[(row+1)%self.height,col-1]: #bottom left
                same += 1
            if centre == self.agents[(row+1)%self.height,col]: #bottom
                same += 1
            if centre == self.agents[(row+1)%self.height,(col+1)%self.width]: #bottom right
                same += 1
            if centre == self.agents[row,(col+1)%self.width]: #right
                same += 1
            if centre == self.agents[row-1,(col+1)%self.width]: #up right
                same += 1
            if centre == self.agents[row-1,col]: #up
                same += 1
            if centre == self.agents[row-1,col-1]: #up left
                same += 1
            
            if same < self.neighbours:
                return int(-1) #unhappy
            else: return int(1) #happy
        else: return 0 #empty cell
    
    
    #determine the happiness of every agent
    def count_happiness(self):
        '''
        Description
        -------
        Calculates the happiness of each agent in the matrix.
        
        Returns
        -------
        float: happiness_factor. Number of happy agent/total number of agents
        
        vec: count_unhappy, vector containing the nr. of unhappy agents after each iteraiton
        '''
        
        self.happiness_mat = np.zeros((self.height,self.width),dtype=int)
        count_unhappy = []
        for row in range(0,self.height):
            for col in range(0,self.width):
                self.happiness_mat[row,col] = self.count_neighbours(row,col)
        
        #append the nr. of unhappy agents after each iteration to this list
        count_unhappy.append(np.count_nonzero(self.happiness_mat == -1))
        count_unhappy = np.array(count_unhappy)
        happiness_factor = np.count_nonzero(self.happiness_mat == 1)/self.amount_agents
        return happiness_factor, count_unhappy
    
    
    #find the location of unhappy agents and update those
    def find_unhappy_agent(self, update_order='random'):
        '''
        Description
        -------
        According to the flag, choose an unhappy agent which is then being
        moved to an empty cell. It will return a list of row and column 
        coordinates of the unhappy agent.
        
        Parameters
        -------
        update_order: 'row', 'col', 'random'\n
            
        Returns
        -------
        list: row and column coordinates of unhappy agent
        '''
        
        #go through the list of all agents by row
        if update_order=='row':
            #retrieve index of unhappy agents from happiness mat
            idx_row, idx_col = np.where((self.happiness_mat == -1) == True) 
            
        #go through the list of all agents by column
        elif update_order == 'col':
            transp_mat = self.happiness_mat.T
            #retrieve index of unhappy agents from transposed happiness mat
            idx_col, idx_row = np.where((transp_mat == -1) == True)
            
        #go through the list of all agents randomly
        elif update_order == 'random':
            #retrieve index of unhappy agents from happiness mat
            idx_row, idx_col = np.where((self.happiness_mat == -1) == True)
            
            #create an empty matrix which will be shuffled
            indices = np.empty((len(idx_col),2),dtype=int)
            #fill the matrix with the index of the unhappy agents, 
            indices[:,0] = idx_row
            indices[:,1] = idx_col
            #shuffle the matrix and assign the new index array for rows and cols
            np.random.shuffle(indices) 
            #extract again to have rows and cols
            idx_row = indices[:,0]
            idx_col = indices[:,1]
        else: 
            sys.exit('Please specify correct update_order')
        
        return idx_row, idx_col
    
    
    #find a place where to move the unhappy agent
    def find_empty_cell(self, idx_agent, flag='random'):
        '''
        Description
        -------
        Finds the coordinates of an empty cell according to the flag
        
        Parameters
        -------
        idx_agent: a list with row and col coordinates of the unhappy agent
        subject to move
        flag: string, either 'top_down', 'bottom_up', 'random' or 'nearest'
            top_down: takes the first available empty cell from the top\n
            bottom_up: takes the last available empty cell at the bottom\n
            random: chooses a random empty cell\n
            nearest: finds the nearest empty cell around the centre
        
        Returns
        -------
        list: row and column coordinate of the empty cell 
        '''
        
        if flag == 'random':
            idx0_row, idx0_col = np.where((self.agents == 0) == True)
            empty_choice = np.random.choice(range(0,self.amount_empty))
            idx0_row = idx0_row[empty_choice]
            idx0_col = idx0_col[empty_choice]
    
        if flag == 'nearest':
        #create a circular, (explosion) region around a centre point in
        #order to find the nearest empty cell around that particular point
            empty_cell = []
            radius = 0
            centre = [idx_agent[0],idx_agent[1]] #the unhappy agent to move
            
            #increase radius after if none found, until there is one empty cell
            while len(empty_cell) == 0:
                radius += 1
                combinations = []
                a_list = []
                for i in range(1,radius+1):
                    list1 = list(range(-i,i+1)) #for radius = 1: [-1,0,1]
                    a_list = [list(each_permutation) 
                              for each_permutation in itertools.permutations(list1, 2)]
            
                    # append the corners manually, for example with radius = 1:
                    # [1, 1] & [-1, -1]
                    if i == radius:
                        a_list.append([list1[0], list1[0]])
                        a_list.append([list1[-1], list1[-1]])
                    combinations.append( a_list )
            
                # for each radius, concatenate the list into one large list
                concat_list = []
                for lst in combinations:
                    concat_list = concat_list + lst
            
                # get only the unique [x,y] values in concatenated list
                unique_list = []
                for combo in concat_list:
                    tmp_list = concat_list.copy()
                    tmp_list.remove(combo)
                    if combo in tmp_list:
                        pass
                    else: unique_list.append(combo)
                        
                #check around the centre if in the region around the centre created
                #above there exists an empty cell
                #TODO: does not work with non-square grids yet
                for cell in unique_list: 
                    i = (centre[0] + cell[0])%self.width #coordinates of explosion edge
                    j = (centre[1] + cell[1])%self.height
                    if self.agents[i,j] == 0:
                        empty_cell.append([i,j])
    
                if len(empty_cell) > 0:
                #assuming all neighbours are of same distance (radius), 
                #chose any empty cell in that region
                    idx = np.random.choice(range(0,len(empty_cell)))
                    idx0_row = empty_cell[idx][0]
                    idx0_col = empty_cell[idx][1]
    
        return idx0_row, idx0_col
    
    

    #move the unhappy agent and switch the values of between the cells
    def move_agents(self, update_order='row', moving_order='random'):
        '''
        Description
        -------
        The unhappy agent is moved to an empty cell either randomly or to the
        nearest empty cell around it, or filled up from the top or bottom. 
        
        Parameters
        -------
        update_order: 'row', 'col', 'random'\n
        moving_order: 'random', 'nearest'
            
        Returns
        -------
        int: count for nr. of moved agents
        '''
        
        if update_order == 'row':
            idx_row, idx_col = self.find_unhappy_agent(update_order)
        elif update_order == 'col':
            idx_row, idx_col = self.find_unhappy_agent(update_order)
        elif update_order == 'random':
            idx_row, idx_col = self.find_unhappy_agent(update_order)
        else: 
            sys.exit('Please specify correct moving_order')
            
        
        #iterate over every unhappy agent found and move it to an empty cell
        for i in range(0,len(idx_row)):
            
            #create a backup of the agent
            old_state = self.agents[idx_row[i], idx_col[i]] 

            if moving_order == 'random':
                idx0_row, idx0_col = self.find_empty_cell([idx_row, idx_col], 
                                                      flag = moving_order)
            elif moving_order == 'nearest':
                idx0_row, idx0_col = self.find_empty_cell([idx_row[i],idx_col[i]],
                                                      flag = moving_order)
            else: 
                sys.exit('Please specify correct input')
        
            #replace empty cell with the state of the re-allocated agent 
            self.agents[idx0_row, idx0_col] = old_state
            #replace the unhappy agent with 0, b/c it was moved
            self.agents[idx_row[i], idx_col[i]] = 0
        
        #return the number of moved agents
        return len(idx_row)
    
    
    
    def run_simulation(self, update_order, moving_order='nearest', strat_switch='no'):
        '''
        Parameters
        ----------
        update_order : 'row','col','random'
            strategy of update for unhappy agents
        strat_switch : 'yes' or 'no', optional
            change strategy mid simulation if derivative is close to zero. \n
            The default is 'no'.

        Returns
        -------
        happiness_vec: vector containing the happiness ratios after each iteration
        '''
        print(
        '---- Simulation for H = %i and %i agents on a (%i x %i) grid. Strategy: %s ----' 
        % (self.neighbours,self.nr_races,self.width,self.height,update_order))
        
        #record the evolution of the happiness over each iteration
        happiness_vec = []
        for i in range(self.iterations):
            if i == 0:
                #count initial happiness and agents to be moved, and make first entry in vector
                ratio, nr_unhappy = self.count_happiness() 
                happiness_vec.append(ratio)
                print(
                'Initial state: Nr. of unhappy agents: %i, overall happiness: %.4f' % 
                  (nr_unhappy[-1],ratio))
            
            #make first update of agents, count again, append to vector
            self.move_agents(update_order,moving_order) #returns nr. of moved agents
            ratio, nr_unhappy = self.count_happiness()
            happiness_vec.append(ratio)
            print('Iter.: %i, unhappy left: %i, happiness: %.4f ' % 
                  (i+1, nr_unhappy, ratio,))
            
            #include stopping criteria
            if nr_unhappy[-1] == 0:
                print('- Total happiness achieved! No unhappy agents left')
                break
            elif i > 5 and np.count_nonzero(nr_unhappy[-5:] < 2) > 5:
                print(
                'Sufficient overall happiness achieved! %i unhappy agents left' 
                % nr_unhappy[-1])
                break
            # stop if derivative is approaching 0 AND 70% of the iterations done
            elif i > (self.iterations*0.7) and np.abs(np.mean(
                    happiness_vec[-15:])-ratio) < 1e-9:
                print('- Converged to sufficient happiness. No more improvement.')
                break
            

            # change strategy mid run
            if strat_switch == 'yes':
                if np.abs(np.mean(happiness_vec[-10:])-ratio) < 1e-3:
                    update_order = 'row'
                    print('Strategy changed to %s' % update_order)
                if i == 150:
                    update_order = 'row'
                    print('Strategy changed to %s' % update_order)
                if i > 150 and np.abs(np.mean(happiness_vec[-10:])-ratio) < 1e-3:
                # if i == [150,200,250] and np.abs(np.mean(happiness_vec[-10:])-ratio) < 1e-3:  
                # if i == 300:
                    if update_order == 'col':
                        update_order = np.random.choice(['random','row','nearest'])
                    elif update_order == 'row':
                        update_order = np.random.choice(['col','random','nearest'])
                    elif update_order == 'random':
                        update_order = np.random.choice(['col','row','nearest'])
                    else: update_order = np.random.choice(['col','row','random'])
                    print('Strategy changed to %s' % update_order)
                
                #strate change for nearest and then switching to random
                if i == np.round(0.05*self.iterations): 
                    moving_order = 'random'
                    print('Strategy changed to %s' % moving_order)

            if i == self.iterations-1:
                print('- Limit of iterations achieved!')
        return happiness_vec
    
    
    # plot the initial and final states of agents
    def generate_state_plot(self, state, hv, 
                            update_order='not defined', 
                            moving_order='not defined',
                            plottype=1):
        '''
        state 'initial' = plot initial state \n
        state 'result' = plot the terminated state \n
        update_order = 'row', 'col' or 'random \n
        plottype 0 = plt.imshow(interpolate='sinc') \n
        plottype 1 = plt.imshow() \n
        plottype 2 = plt.spy()  \n
        '''

        # Titles for each plot
        if state=='initial':
            title = str(
            'Initial state of agents on a (%i x %i) grid \n nr. of races: %i, empty ratio: %.2f' 
                          % (self.width,self.height,self.nr_races,self.empty_ratio))
        elif state=='final':
            title = str(
            'Updating: %s, movement: %s \n neighbours: %i, races: %i, final happiness: %.3f' 
            % (update_order,moving_order,self.neighbours,self.nr_races, hv[-1]))
        
        # Generate figure and customize
        plt.rcParams['font.size'] = '14'
        plt.figure(figsize=(7,6))
        plt.title(title)
        plt.tick_params(
            axis='x', which='both',
            top=False,bottom=False,left=False,right=False,
            labeltop=False,labelbottom=False) # labels along the bottom edge are off
        plt.tick_params(
            axis='y', which='both',      # both major and minor ticks are affected
            top=False, bottom=False, right=False, left=False,
            labelright=False, labelleft=False)
        plt.tight_layout()
        
        #Plot agents (terminal state)
        if plottype == 0:
            from matplotlib import colors
            # https://matplotlib.org/stable/gallery/color/named_colors.html
            # colormap = colors.ListedColormap(["blue","lightgrey","red"]) 
            #order from low to high number
            
            fullcolormap = ["lightgrey","blue","red","yellow","green","brown","black"]
            colormap = colors.ListedColormap(fullcolormap[0:self.nr_races+1])
            plt.imshow(self.agents, cmap=colormap, interpolation='sinc')
            # plt.show()
            
        if plottype == 1:
            from matplotlib import colors
            # https://matplotlib.org/stable/gallery/color/named_colors.html
            # colormap = colors.ListedColormap(["blue","lightgrey","red"]) 
            #order from low to high number
            
            fullcolormap = ["lightgrey","blue","red","yellow","green","brown","black"]
            colormap = colors.ListedColormap(fullcolormap[0:self.nr_races+1])
            plt.imshow(self.agents, cmap=colormap)
            # plt.show()
            
        if plottype == 2: 
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.spy.html
            c = ['C0','C1','C2','C3','C4','C5','C6','C7','C8']
            for race in range(1,self.nr_races+1):
                plt.spy(self.agents == race, markersize=(2), c=c[race-1])
            # plt.show()  
            
    # Plot the happiness matrix, i.e., the location of all unhappy agents if any
    def generate_hmat_plot(self, hv):
        
        if np.count_nonzero(self.happiness_mat == -1) > 0:
            title = str('Happiness matrix - Location of unhappy agents \n' + 
                        'Nr. of unhappy agents: %i, happiness ratio: %.4f' % 
                        (np.count_nonzero(self.happiness_mat == -1), hv[-1]))
            plt.figure(figsize=(5,5))
            plt.title(title, fontsize = 9)
            plt.tick_params(
                axis='x',
                which='both',
                top=False,
                bottom=False,
                labeltop=False,labelbottom=False)
            plt.tick_params(
                axis='y',
                which='both',
                top=False,
                left=False,
                labelleft=False)
            plt.spy(self.happiness_mat == -1) #markersize=2
            # plt.show()
        else: 
            print(
        '-----\n No unhappy agents left to generate an unhappyness-matrix plot! \n -----'
                )
    
    
    # Plot the evolution of the happiness of agents over iterations
    def generate_hdev_plot(self, hv, 
                           update_order='not defined', 
                           moving_order='not_defined'):
        
        plt.figure(figsize=(7,5))
        plt.title('Update: %s, movement: %s, races: %i, H = %i \n Final happiness ratio: %.4f' 
            % (update_order,moving_order,self.nr_races,self.neighbours,hv[-1]))
        plt.plot(hv)
        plt.grid()
        # plt.show()
    
        
def main():
    # SchellingABM(grid_width, grid_height, empty_ratio, nr_races, neighbours, iterations):
    
    #set parameters
    N = 100
    empty_ratio = 0.1
    H = 4
    nr_races = 2
    iterations = 200
    
    update_order = 'random'
    moving_order = 'random'
    strat_switch = 'no'
    
    
    # ---------- Run simulation ----------
    tic = time.time()
    ''' Initialisation of model '''
    schelling = SchellingABM(N, N, 
                             empty_ratio, 
                             nr_races, 
                             H, 
                             iterations)
    schelling.set_up_agents()
    schelling.generate_state_plot(state = 'initial', 
                                  hv = 0,
                                  plottype = 1)
    happiness_vec = schelling.run_simulation(update_order, 
                                             moving_order, 
                                             strat_switch)
    toc = time.time()
    print('- Time in seconds elapsed: %.2fsec' % (toc-tic))
    
    
    # ---------- Generate plots ----------
    schelling.generate_state_plot(state = 'final', 
                                  hv = happiness_vec, 
                                  update_order = update_order,
                                  moving_order = moving_order,
                                  plottype = 1)
    plt.show()
    schelling.generate_hmat_plot(hv = happiness_vec)
    plt.show()
    schelling.generate_hdev_plot(hv = happiness_vec, 
                                  update_order=update_order, 
                                  moving_order=moving_order)
    plt.show()
                  
       
if __name__ == "__main__":
    main()