import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
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
    
    
    def initialize_Decomp(self, rho, N_it, fi_list, z, rho_inc = 1): 
        self.rho = rho
        self.N_it = N_it
        self.f_list = fi_list
        self.prim_r = []
        self.dual_r = []
        
        self.rho_inc = rho_inc
        
        self.systems = {}
        #self.u_list = {}
        
        if len(z) != len(self.global_ind):
            raise ValueError('z should have as many elements as global_ind')
        
        for i in range(self.N):
            self.systems[i+1] = {} #; self.u_list[i+1] = {}
            for j in range(self.N_var):
                if j+1 in self.idx_agents[i+1]:
                    self.systems[i+1][j+1] = []
                # if j+1 in self.global_ind:
                #     self.u_list[i+1][j+1] = [u]
        
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
        
    
    def solve_subproblems(self):
        self.obj = 0
        self.conv = 0
        for i in range(self.N):
            #print(z_list, u_list, rho)
            try: 
                instance, s = self.f_list[i](self.z_list, self.rho, 
                                          self.global_ind, self.idx_agents[i+1], solver = True)
            
                if (s.solver.status != SolverStatus.ok) or (s.solver.termination_condition != TerminationCondition.optimal):
                    self.conv = 1
            except:
                instance = self.f_list[i](self.z_list, self.rho, 
                                          self.global_ind, self.idx_agents[i+1], solver = True)
            
            for j in instance.x:
                self.systems[i+1][j] += [pyo.value(instance.x[j])]

            self.obj += pyo.value(instance.obj)
    
    def f_surr(self, x):
        
        for i in range(len(self.global_ind)):
            self.z_list[self.global_ind[i]] += [x[i]]
            
        self.solve_subproblems()    
        self.compute_residuals()
        self.rho *= self.rho_inc
        
        return self.obj, [self.conv]
        
    def solve(self, solver, x0, bounds, init_trust, budget = 100, 
              N_min_s = 6, beta_red = 0.9, rnd_seed = 0, 
              method = 'local', constr = 'Discrimination',
              ineq_known = None, eq_known = None):
        self.obj_list = []
        
        dict_out = solver(self.f_surr, x0, init_trust, bounds = bounds, max_f_eval = budget, \
                           N_min_samples = N_min_s, beta_red = beta_red, \
                           rnd = rnd_seed, method = method, \
                           constr_handling = constr, 
                           eq_known = eq_known,
                           ineq_known = ineq_known)
        
        return dict_out
