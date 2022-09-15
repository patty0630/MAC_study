import pyomo.environ as pyo
import numpy as np

class System:
    def __init__(self, N_agents, N_var, index_agents, global_ind):
        """
        :N_agents: Number of subsystems
        :index_agents: Dictionary where index_agents[1] gives the list of 
                      agent 1 variable indices as e.g. [1, 3]
        :global_ind:   list of indices for global variables e.g. [3]
        """
        
        self.N = N_agents
        
        if len(index_agents) != N_agents:
            raise ValueError('index_agents list should have as many items as N_agents')
        
        self.idx_agents = index_agents
        self.global_ind = global_ind
        self.N_var = N_var
    
    
    def initialize_ADMM(self, rho, N_it, fi_list, z, u = 0): 
        self.rho = rho
        self.N_it = N_it
        self.f_list = fi_list
        self.prim_r = []
        self.dual_r = []
        
        self.systems = {}
        self.u_list = {}
        self.obj = {}
        
        if len(z) != len(self.global_ind):
            raise ValueError('z should have as many elements as global_ind')
        
        for i in range(self.N):
            self.systems[i+1] = {} ; self.u_list[i+1] = {}
            self.obj[i+1] = []
            for j in range(self.N_var):
                if j+1 in self.idx_agents[i+1]:
                    self.systems[i+1][j+1] = []
                if j+1 in self.global_ind:
                    self.u_list[i+1][j+1] = [u]
        
        self.z_list = {} ; self.z_temp = {}
        for i in self.global_ind:
            self.z_list[i] = [z[i]] ; self.z_temp[i] = [] 
        
    
    def compute_residuals(self):
        for idx in self.global_ind:
            self.z_temp[idx] = [self.systems[i+1][idx][-1] for i in range(self.N)]
        self.prim_r += [ np.linalg.norm([np.linalg.norm([self.systems[i+1][idx][-1] - \
                         np.mean(self.z_temp[idx]) for i in range(self.N)]) for idx in self.global_ind])]
        self.dual_r += [ self.rho*np.linalg.norm([np.linalg.norm(np.mean(self.z_temp[idx]) - \
                         self.z_list[idx][-1]) for idx in self.global_ind])]
    
    
    def update_lists(self):
        for idx in self.global_ind:
            self.z_list[idx] += [np.mean(self.z_temp[idx])]
            for i in range(self.N):
                self.u_list[i+1][idx] += [self.u_list[i+1][idx][-1] + self.z_temp[idx][i] - \
                                          np.mean(self.z_temp[idx])]
    
    def solve_subproblems(self):
        for i in range(self.N):
            ver = int(np.ceil((i+1)/2))
            #print(ver)
            #print(z_list, u_list, rho)
            instance = self.f_list[i](self.z_list, self.rho, self.global_ind, 
                                      self.idx_agents[i+1], ver, u_list = self.u_list[i+1])
            #print(pyo.value(instance.obj))
            self.obj[i+1] += [pyo.value(instance.obj)]
            for j in instance.x:
                self.systems[i+1][j] += [pyo.value(instance.x[j])]
    
    def solve_ADMM(self):
        for k in range(self.N_it):
            self.solve_subproblems()    
            self.compute_residuals()
            self.update_lists()
