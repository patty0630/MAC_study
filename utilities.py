import numpy as np

def post_ALADIN(array, init):
    N = len(array)
    if init is not None:
        best = float(init)
    else:
        best = array[0]
    for i in range(N):
        if best < array[i]:
            array[i] = best
        else:
            best = array[i]
    return array

def postprocessing(ax1, ax2, string, result, actual_f, coord_input = False, ALADIN = False, BO = False, samecoord = False, init = None, N=2, c='black'):
    if BO:
        obj_global = result
        conv_arr = (obj_global - actual_f)**2 
        n_eval = np.arange(len(obj_global))+1
        ax1.step(n_eval, conv_arr, '--', where = 'post', label = string, c=c)
        ax2.step(n_eval, obj_global, '--', where = 'post', label = string, c=c)
    elif ALADIN:
        if samecoord:
            obj_arr = np.array(result.best_obj)
            n_eval = np.array(result.n_eval)
            # obj_global = np.array(result.obj_global)
            # z_arr = np.array(result.center_list)
            # ax2.scatter(z_arr[:,0], z_arr[:,1], label = string, s = 10)
            conv_arr = (obj_arr - actual_f)**2 
            ax1.step(n_eval, conv_arr, '--', where = 'post', label = string, c=c)
            ax2.step(n_eval, obj_arr, '--', where = 'post', label = string, c=c)
        else:
            obj_global = np.sum(np.array([result.obj[i+1] for i in range(N)]), axis = 0)
            obj_global = post_ALADIN(obj_global, init)
            # z_arr1 = np.mean([result.z_list[idx+1][global_ind[0]] for idx in range(N)], axis = 0)
            # z_arr2 = np.mean([result.z_list[idx+1][global_ind[1]] for idx in range(N)], axis = 0)
            # ax2.scatter(z_arr1, z_arr2, label = string, s = 10)
            conv_arr = (obj_global - actual_f)**2 
            n_eval = np.arange(len(obj_global))+1
            ax1.step(n_eval, conv_arr, '--', where = 'post', label = string, c=c)
            ax2.step(n_eval, obj_global, '--', where = 'post', label = string, c=c)
    elif not coord_input:
        obj_arr = np.sum(np.array([result.obj[i+1] for i in range(N)]), axis = 0)
        # z_arr1 = np.array(result.z_list[global_ind[0]])
        # z_arr2 = np.array(result.z_list[global_ind[1]])
        conv_arr = (obj_arr - actual_f)**2 
        ax1.step(np.arange(len(obj_arr))+1, conv_arr, '--', where = 'post', label = string, c=c)
        # ax2.scatter(z_arr1, z_arr2, label = string, s = 10)   
        ax2.step(np.arange(len(obj_arr))+1, obj_arr, '--', where = 'post', label = string, c=c)
    else:
        # f = np.array(result['f_store'])
        # x_list = np.array(result['x_store'])
        # x_best = np.array(result['x_best_so_far'])
        f_best = np.array(result['f_best_so_far'])
        if init is not None:
            f_best = preprocess_BO(f_best, init)
        ind_best = np.array(result['samples_at_iteration'])       
        ax1.step(ind_best, (f_best - actual_f)**2, '--', where = 'post', label = string, c=c)
        # ax2.plot(x_best[:,0], x_best[:,1], '--', c = 'k', linewidth = 1)
        # ax2.scatter(x_list[:,0], x_list[:,1], label = string, s = 10)
        ax2.step(ind_best, f_best, '--', where = 'post', label = string, c=c)
        
    return ax1, ax2

def construct_A(index_dict, global_list, N_ag, only_global = False):
    N_g = len(global_list)
    if not only_global:
        A_list = {i+1: np.zeros(((N_ag-1)*N_g, len(index_dict[i+1]))) for i in range(N_ag)}
        for i in range(len(global_list)):
            for j in range(N_ag-1):
                idx = np.where(np.array(index_dict[j+1]) == np.array(global_list[i]))
                A_list[j+1][i*(N_ag-1)+j, idx] = 1
                idx = np.where(np.array(index_dict[j+2]) == np.array(global_list[i]))
                A_list[j+2][i*(N_ag-1)+j, idx] = -1
    else:
        A_list = {i+1: np.zeros(((N_ag-1)*N_g, N_g)) for i in range(N_ag)}
        for i in range(len(global_list)):
            for j in range(N_ag-1):
                A_list[j+1][i*(N_ag-1)+j, i] = 1
                A_list[j+2][i*(N_ag-1)+j, i] = -1
    return A_list

def preprocess_BO(array, init, N_eval = None):
    f_best = array.copy()
    if N_eval is None:
        N_eval = len(f_best)
    else:
        N_eval = N_eval
    f_arr = np.zeros(N_eval)
    f_arr[0] = float(init)
    best = float(init)
    for j in range(1, N_eval):
        if (j < len(f_best)):
            if (best > f_best[j]):
                best = f_best[j]  
        f_arr[j] = best
    return f_arr

def average_from_list(array_total):
    f_median = np.median(array_total, axis = 0)
    f_min = np.min(array_total, axis = 0)
    f_max = np.max(array_total, axis = 0)
    return f_median, f_min, f_max

def postprocessing_List(ax1, ax2, string, List, actual_f, coord_input = False, ALADIN = False, BO = False, samecoord = False, init = None, N=2, N_it=50, c='black'):
    N_runs = len(List)
    eval_array = np.zeros((N_runs, N_it))
    conv_array = np.zeros((N_runs, N_it))
    if BO:
        for i in range(N_runs):
            obj_global = List[i]
            conv = (obj_global - actual_f)**2 
            eval_array[i] = obj_global[:N_it]
            conv_array[i] = conv[:N_it]
            # n_eval = np.arange(len(obj_global))+1
        # ax1.step(n_eval, conv_arr, '--', where = 'post', label = string)
        # ax2.step(n_eval, obj_global, '--', where = 'post', label = string)
    elif ALADIN:
        if samecoord:
            for i in range(N_runs):
                obj = np.array(List[i].best_obj)
                n_eval = np.array(List[i].n_eval)
                conv = (obj - actual_f)**2
                eval_array[i,0] = obj[0]
                conv_array[i,0] = conv[0]
                for j in range(1,N_it):
                    ind = np.where(n_eval <= j+1)
                    if len(ind[0]) == 0:
                        eval_array[i,j] = obj[0]
                        conv_array[i,j] = conv[0]
                    else:
                        eval_array[i,j] = obj[ind][-1]
                        conv_array[i,j] = conv[ind][-1]
        else:
            for i in range(N_runs):
                obj_global = np.sum(np.array([List[i].obj[k+1] for k in range(N)]), axis = 0)
                obj_global = post_ALADIN(obj_global, init)
                conv = (obj_global - actual_f)**2 
                eval_array[i] = obj_global[:N_it]
                conv_array[i] = conv[:N_it]
    elif not coord_input:
        for i in range(N_runs):
            obj_global = np.sum(np.array([List[i].obj[k+1] for k in range(N)]), axis = 0)
            obj_global = post_ALADIN(obj_global, init)
            conv = (obj_global - actual_f)**2 
            eval_array[i] = obj_global[:N_it]
            conv_array[i] = conv[:N_it]
    else:
        for i in range(N_runs):
            obj = np.array(List[i]['f_best_so_far'])
            if init is not None:
                obj = preprocess_BO(obj, init)
            conv = (obj - actual_f)**2
            n_eval = np.array(List[i]['samples_at_iteration'])  
            eval_array[i,0] = obj[0]
            conv_array[i,0] = conv[0]
            for j in range(1,N_it):
                    ind = np.where(n_eval <= j+1)
                    if len(ind[0]) == 0:
                        eval_array[i,j] = obj[0]
                        conv_array[i,j] = conv[0]
                    else:
                        eval_array[i,j] = obj[ind][-1]
                        conv_array[i,j] = conv[ind][-1]
    
    x_axis = np.arange(1, 1+N_it)
    eval_median, eval_min, eval_max = average_from_list(eval_array)
    conv_median, conv_min, conv_max = average_from_list(conv_array)
    ax1.step(x_axis, conv_median, where = 'post', label = string, c=c, linestyle='--')
    ax1.fill_between(x_axis, conv_min, conv_max, alpha = .5, step = 'post', color=c)
    ax2.step(x_axis, eval_median, where = 'post', label = string, c=c, linestyle='--')
    ax2.fill_between(x_axis, eval_min, eval_max, alpha = .5, step = 'post', color=c)
            
    return ax1, ax2

class dummy_dict:
        def __init__(self, input_dict):
            self.obj = input_dict.obj

def postprocess_ADMM(input_dict):
    return dummy_dict(input_dict)
