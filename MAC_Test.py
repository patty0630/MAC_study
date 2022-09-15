import pyomo.environ as pyo
import numpy as np
# from Algorithms.ADMM_Scaled_Consensus_acc import System as ADMM_Scaled_acc
from Algorithms.ADMM_Scaled_Consensus import System as ADMM_Scaled

from Algorithms.Coordinator_Augmented import System as Coordinator_ADMM
from Algorithms.CUATRO import CUATRO

from Algorithms.PyBobyqa_wrapped.Wrapper_for_pybobyqa import PyBobyqaWrapper
from Algorithms.DIRECT_wrapped.Wrapper_for_Direct import DIRECTWrapper

from utilities import postprocess_ADMM


from Problems.MAC import f1 as f1Raw
from Problems.MAC import f2 as f2Raw
from Problems.MAC import f1Lin as f1LinRaw
from Problems.MAC import f2Lin as f2LinRaw

import pickle

rho = 1e4
N_list = [3,4,6,8]

for i in N_list:
    
    pen=int(np.log10(rho))

    N_it = 500
    N_runs = 1
    
    N = i

    
    N_var = 10
    
    save_data_list = []

    
    def f1(z_list, rho, global_ind, index, ver, u_list = None, solver = False):
        return f1Raw(z_list, rho, global_ind, index, ver, u_list = None, solver = False, dim=N_var)
    def f2(z_list, rho, global_ind, index, ver, u_list = None, solver = False):
        return f2Raw(z_list, rho, global_ind, index, ver, u_list = None, solver = False, dim=N_var)
    
    
    list_fi = []
    for i in range(N):
        if not i%2:
            list_fi += [f1]
        else:
            list_fi += [f2]
    # list_fi = [f1, f2]
    
    # global_ind = [3]
    # index_agents = {1: [1, 3], 2: [2, 3]}
    global_ind = [i+1 for i in range(N_var)]
    
    index_agents = {}
    for i in range(N):
        index_agents[i+1] = global_ind
    z = {n: 1/N_var for n in global_ind}
    
    actual_f = 0
    # actual_x = 0.398
    
    ADMM_Scaled_system10nonconv = ADMM_Scaled(N, N_var, index_agents, global_ind)
    ADMM_Scaled_system10nonconv.initialize_ADMM(rho, N_it, list_fi, z)
    ADMM_Scaled_system10nonconv.solve_ADMM()
    print('ADMM done')
    save_data_list += [postprocess_ADMM(ADMM_Scaled_system10nonconv)]
    
    
    bounds = np.array([[0, 1]]*N_var)
    x0 = np.zeros(N_var)+1/N_var
    init_trust = 0.5 # 0.25
    N_s = 40
    beta = 0.95
    
    ADMM_CUATRO_list10nonconv = []
    s = 'CUATRO'
    for i in range(1):
        Coordinator_ADMM_system10nonconv = Coordinator_ADMM(N, N_var, index_agents, global_ind)
        Coordinator_ADMM_system10nonconv.initialize_Decomp(rho, N_it, list_fi, z)
        output_Coord1_10nonconv = Coordinator_ADMM_system10nonconv.solve(CUATRO, x0, bounds, init_trust, 
                                budget = N_it, beta_red = beta, N_min_s = N_s)  
        ADMM_CUATRO_list10nonconv += [output_Coord1_10nonconv]
        print(s + ' run ' + str(i+1) + ': Done')
    print('Coord1 done')
    
    save_data_list += [ADMM_CUATRO_list10nonconv]
    
    
    def f_pbqa(x):
        z_list = {i: [x[i-1]] for i in global_ind}
        return np.sum([pyo.value(f(z_list, rho, global_ind, global_ind, int(np.ceil((i+1)/2))).obj) for i, f in enumerate(list_fi)]), [0]
    f_DIR = lambda x, grad: f_pbqa(x)
    
    def f_BO(x):
        if x.ndim > 1:
            x_temp = x[-1] 
        else:
            x_temp = x
        # temp_dict = {i+1: x[:,i] for i in range(len(x))}
        z_list = {i: [x_temp[i-1]] for i in global_ind}
        return np.sum([pyo.value(f(z_list, rho, global_ind, global_ind, int(np.ceil((i+1)/2))).obj) for i, f in enumerate(list_fi)])
    
    domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (0., 1.)} for i in range(N_var)]
    y0 = np.array([f_BO(x0)])
    
    pybobyqa10nonconv = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
                                          maxfun=N_it, constraints=1, \
                                          seek_global_minimum= True, \
                                          objfun_has_noise=False)
    
    save_data_list += [pybobyqa10nonconv]
    
    s = 'DIRECT'
    DIRECT_List10nonconv = []
    for i in range(N_runs): 
        DIRECT10nonconv =  DIRECTWrapper().solve(f_DIR, x0, bounds, \
                                        maxfun = N_it, constraints=1)
        for j in range(len(DIRECT10nonconv['f_best_so_far'])):
            if DIRECT10nonconv['f_best_so_far'][j] > float(y0):
                DIRECT10nonconv['f_best_so_far'][j] = float(y0)
        DIRECT_List10nonconv += [DIRECT10nonconv]
        print(s + ' run ' + str(i+1) + ': Done')
    print('DIRECT done')
    
    save_data_list += [DIRECT_List10nonconv]
    
    s_list = ['ADMM', 'CUATRO', 'Py-BOBYQA', 'DIRECT-L']
    
    dim = len(x0)
    problem = 'MAC_nonconv_'
    
    for k in range(len(s_list)):
        with open('./Data/' + problem + str(N_var) + 'dim_' + str(N) + 'agents_1E' + str(pen) + '_' + s_list[k] + '.pickle', 'wb') as handle:
            pickle.dump(save_data_list[k], handle, protocol=pickle.HIGHEST_PROTOCOL) 
    
    

    

    def f1Lin(z_list, rho, global_ind, index, ver, u_list = None, solver = False):
        return f1LinRaw(z_list, rho, global_ind, index, ver, u_list = None, solver = False, dim=N_var)
    def f2Lin(z_list, rho, global_ind, index, ver, u_list = None, solver = False):
        return f2LinRaw(z_list, rho, global_ind, index, ver, u_list = None, solver = False, dim=N_var)
    
    list_fi = []
    for i in range(N):
        if not i%2:
            list_fi += [f1Lin]
        else:
            list_fi += [f2Lin]
    #list_fi = [f1Lin, f2Lin]
    
    # global_ind = [3]
    # index_agents = {1: [1, 3], 2: [2, 3]}
    global_ind = [i+1 for i in range(N_var)]
    index_agents = {}
    for i in range(N):
        index_agents[i+1] = global_ind
    z = {n: 1/N_var for n in global_ind}
    
    save_data_list = []
    
    ADMM_Scaled_system10conv = ADMM_Scaled(N, N_var, index_agents, global_ind)
    ADMM_Scaled_system10conv.initialize_ADMM(rho/10, N_it, list_fi, z)
    ADMM_Scaled_system10conv.solve_ADMM()
    print('ADMM done')
    
    save_data_list += [postprocess_ADMM(ADMM_Scaled_system10conv)]
    
    bounds = np.array([[0, 1]]*N_var)
    x0 = np.zeros(N_var)+1/N_var
    init_trust = 0.5 # 0.25
    N_s = 40
    beta = 0.95
    
    
    ADMM_CUATRO_list10conv = []
    s = 'CUATRO'
    for i in range(1):
        Coordinator_ADMM_system10conv = Coordinator_ADMM(N, N_var, index_agents, global_ind)
        Coordinator_ADMM_system10conv.initialize_Decomp(rho, N_it, list_fi, z)
        output_Coord1_10conv = Coordinator_ADMM_system10conv.solve(CUATRO, x0, bounds, init_trust, 
                                budget = N_it, beta_red = beta, N_min_s = N_s)  
        ADMM_CUATRO_list10conv += [output_Coord1_10conv]
        print(s + ' run ' + str(i+1) + ': Done')
    print('Coord1 done')
    
    save_data_list += [ADMM_CUATRO_list10conv]
    
    
    def f_pbqa(x):
        z_list = {i: [x[i-1]] for i in global_ind}
        return np.sum([pyo.value(f(z_list, rho, global_ind, global_ind, int(np.ceil((i+1)/2))).obj) for i, f in enumerate(list_fi)]), [0]
    f_DIR = lambda x, grad: f_pbqa(x)
    def f_BO(x):
        if x.ndim > 1:
           x_temp = x[-1] 
        else:
           x_temp = x
        # temp_dict = {i+1: x[:,i] for i in range(len(x))}
        z_list = {i: [x_temp[i-1]] for i in global_ind}
        return np.sum([pyo.value(f(z_list, rho, global_ind, global_ind, int(np.ceil((i+1)/2))).obj) for i, f in enumerate(list_fi)])
    
    domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (0., 1.)} for i in range(N_var)]
    y0 = np.array([f_BO(x0)])
    
    pybobyqa10conv = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
                                          maxfun=N_it, constraints=1, \
                                          seek_global_minimum= True, \
                                          objfun_has_noise=False)
    
    save_data_list += [pybobyqa10conv]
        
    s = 'DIRECT'
    DIRECT_List10conv = []
    for i in range(N_runs): 
        DIRECT10conv =  DIRECTWrapper().solve(f_DIR, x0, bounds, \
                                        maxfun = N_it, constraints=1)
        for j in range(len(DIRECT10conv['f_best_so_far'])):
            if DIRECT10conv['f_best_so_far'][j] > float(y0):
                DIRECT10conv['f_best_so_far'][j] = float(y0)
        DIRECT_List10conv += [DIRECT10conv]
        print(s + ' run ' + str(i+1) + ': Done')
    print('DIRECT done')
    
    save_data_list += [DIRECT_List10conv]
    
    dim = len(x0)
    problem = 'MAC_conv_'
    
    for k in range(len(s_list)):
        with open('./Data/' + problem + str(N_var) +'dim_'+ str(N) + 'agents_1E' + str(pen) + '_' + s_list[k] + '.pickle', 'wb') as handle:
            pickle.dump(save_data_list[k], handle, protocol=pickle.HIGHEST_PROTOCOL) 
    
