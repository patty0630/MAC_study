import cvxpy as cp
import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt

def quadratic_LA(X, Y, P, q, r):
    N = len(X)
    Z = np.zeros(X.shape)
    for i in range(N):
        for j in range(N):
            X_ = np.array([X[i,j], Y[i,j]]).reshape(-1,1)
            Z[i,j] = float(X_.T @ P @ X_ + q.T @ X_ + r)
    return Z

def make_PSD(P):
    eig_val, eig_vec = LA.eigh(P)
    # print(eig_val)
    eig_val = np.array([max(val, 1e-8) for val in eig_val])
    # print(eig_val)
    P = np.dot(eig_vec, eig_val[:, np.newaxis]*eig_vec.T)
    return P

def LHS(bounds, N, rnd_seed = 1):
    np.random.seed(rnd_seed)
    matrix = np.zeros((len(bounds), N))
    for i in range(len(bounds)):
        l, u = bounds[i]
        rnd_ind = np.arange(N)
        np.random.shuffle(rnd_ind)
        # print(rnd_ind)
        rnd_array = l + (np.random.rand(N)+ rnd_ind)*(u-l)/N
        matrix[i] = rnd_array
    return matrix

def sample_LHS(sim, bounds, N, rnd_seed = 1):
    data_points = LHS(bounds, N, rnd_seed = rnd_seed).T
    func_eval, g_eval, feas = sample_simulation(data_points.tolist(), sim)
    
    return data_points, func_eval, g_eval, feas

def sample_points(center, radius, f, bounds, N = 10):
    
    if bounds is None:
        data_points = np.array(center*N).reshape(N, len(center)) + \
                      np.random.uniform(-radius, radius, (N, len(center)))
    else:
        uniform_sampling = np.zeros((N, len(center)))
        for i in range(len(center)):
            lower_bound = - radius ; upper_bound = radius
            if center[i] - radius < bounds[i,0]:
                lower_bound = bounds[i,0] - center[i]
            if center[i] + radius > bounds[i,1]:
                upper_bound = bounds[i,1] - center[i]
            uniform_sampling[:,i] = np.random.uniform(lower_bound, upper_bound, N)
            
        data_points = np.array(center*N).reshape(N, len(center)) + \
                        uniform_sampling
    func_eval, g_eval, feas = sample_simulation(data_points.tolist(), f)
    
    return data_points, func_eval, g_eval, feas


def update_best_lists(X_list, f_list, g_list, X_best, f_best, g_best):
    g_feas = constr_creation(X_list, g_list)
    f = np.array(f_list)
    ind = np.where(f == np.min(f[g_feas == 1]))
    X_best += np.array(X_list)[ind].tolist()[:1]
    f_best += f[ind].tolist()[:1]
    g_best += np.array(g_list)[ind].tolist()[:1]
    
    return X_best, f_best, g_best

def samples_in_trust(center, radius, \
                     X_samples_list, y_samples_list, g_list):
    X = np.array(X_samples_list) 
    y = np.array(y_samples_list) 
    g = np.array(g_list)
    ind = np.where(np.linalg.norm(X - np.array(center), axis = 1,\
                                  keepdims = True, ord = np.inf) < radius)[0]
    X_in_trust = X[ind] ; y_in_trust = y[ind] ; g_in_trust = g[ind]
    feas_in_trust = constr_creation(X_in_trust, g_in_trust.tolist())
    
    return X_in_trust, y_in_trust, g_in_trust, feas_in_trust

def quadratic_fitting(X_mat, y_mat, discr = False):
    N, M = X_mat.shape[0], X_mat.shape[1]
    P = cp.Variable((M, M), PSD = True)
    q = cp.Variable((M, 1))
    r = cp.Variable()
    X = cp.Parameter(X_mat.shape)
    y = cp.Parameter(y_mat.shape)
    X.value = X_mat
    y.value = y_mat
    quadratic = cp.bmat([cp.quad_form(X.value[i].reshape(-1,1), P) + \
                        q.T @ X.value[i].reshape(-1,1) + r - y.value[i] for i in range(N)])
    # quadratic = cp.quad_form(X, P) + q.T @ X 
    # quadratic = cp.quad_form(X, P) + q.T @ X + r
    obj = cp.Minimize(cp.norm(quadratic))
    if not discr:
        prob = cp.Problem(obj)
    else:
        const_P = [P >> np.eye(M)*1e-9]
        prob = cp.Problem(obj, constraints = const_P)
    if not prob.is_dcp():
        print("Problem is not disciplined convex. No global certificate")
    prob.solve()
    if prob.status not in ['unbounded', 'infeasible']:
        return P.value, q.value, r.value
    else:
        print(prob.status, ' CVX objective fitting call at: ')
        print('X matrix', X_mat)
        print('y array', y_mat)
        raise ValueError

def quadratic_discrimination(x_inside, y_outside):
    N, M, D = x_inside.shape[0], y_outside.shape[0], x_inside.shape[1]
    u = cp.Variable(N, pos = True)
    v = cp.Variable(M, pos = True)
    P = cp.Variable((D,D), PSD = True)
    q = cp.Variable((D, 1))
    r = cp.Variable()
    X = cp.Parameter(x_inside.shape, value = x_inside)
    Y = cp.Parameter(y_outside.shape)
    X.value = x_inside ; Y.value = y_outside
    const_u = [cp.quad_form(X.value[i].reshape(-1,1), P) + \
                        q.T @ X.value[i].reshape(-1,1) + r <= -(1 - u[i]) for i in range(N)]
    const_v = [cp.quad_form(Y.value[i].reshape(-1,1), P) + \
                        q.T @ Y.value[i].reshape(-1,1) + r >= (1 - v[i]) for i in range(M)]
    const_P = [P >> np.eye(D)*1e-9]
    # const_P = [P >> np.eye(D)*1]
    # const_P = [P >> 0]
    prob = cp.Problem(cp.Minimize(cp.sum(u) + cp.sum(v)), \
                      constraints = const_u + const_v + const_P)
    if not prob.is_dcp():
        print("Problem is not disciplined convex. No global certificate")
    prob.solve()
    if prob.status not in ['unbounded', 'infeasible']:
        return P.value, q.value, r.value
    else:
        print(prob.status, ' CVX ineq. classification call at: ')
        print('x_inside', x_inside)
        print('x_outside', y_outside)
        raise ValueError


def quadratic_min(P_, q_, r_, center, radius, bounds, 
                  ineq = None, ineq_known = None, eq_known = None):
    X = cp.Variable((len(center), 1))
    # P = cp.Parameter(P_.shape, value = P_, PSD = True)
    try:
        P = cp.Parameter(P_.shape, value = P_, PSD = True)
    except:
        P_ = make_PSD(P_)
        if (P_ == 0).all():
            P_ = np.eye(len(P_))*1e-8
            P = cp.Parameter(P_.shape, value = P_, PSD = True)
        try:
            P = cp.Parameter(P_.shape, value = P_, PSD = True)
        except:
            P_ = np.eye(len(P))*1e-8
            P = cp.Parameter(P_.shape, value = P_, PSD = True)
    q = cp.Parameter(q_.shape, value = q_)
    r = cp.Parameter(r_.shape, value = r_)
    objective = cp.Minimize(cp.quad_form(X, P) + q.T @ X + r)
    trust_center = np.array(center).reshape((P_.shape[0], 1))
    constraints = []
    if ineq != None:
        for coeff in ineq:
            P_ineq, q_ineq, r_ineq = coeff
            if not ((P_ineq is None) or (q_ineq is None) or (r_ineq is None)):
                P_iq = cp.Parameter(P_ineq.shape, value = P_ineq, PSD = True)
                q_iq = cp.Parameter(q_ineq.shape, value = q_ineq)
                r_iq = cp.Parameter(r_ineq.shape, value = r_ineq)
                constraints += [cp.norm(X - trust_center) <= radius,
                           cp.quad_form(X, P_iq) + q_iq.T @ X + r_iq <= 0]

    else:
        constraints = [cp.norm(X - trust_center) <= radius]
    if bounds is not None:
        constraints += [bounds[i,0] <=  X[i] for i in range(P_.shape[0])]
        constraints += [X[i] <= bounds[i,1] for i in range(P_.shape[0])]
    if eq_known is not None:
        for coeff in eq_known:
            q_equal, r_equal = coeff
            q_eq = cp.Parameter(q_equal.shape, value = q_equal)
            r_eq = cp.Parameter(r_equal.shape, value = r_equal)
            constraints += [q_eq.T @ X + r_eq == 0]
    if ineq_known is not None:
        for coeff in eq_known:
            P_ineq, q_ineq, r_ineq = coeff
            q_iq = cp.Parameter(q_ineq.shape, value = q_ineq)
            r_iq = cp.Parameter(r_ineq.shape, value = r_ineq)
            if P_ineq is not None:
                P_iq = cp.Parameter(P_ineq.shape, value = P_ineq, PSD = True)
                constraints += [cp.quad_form(X, P_iq) + q_iq.T @ X + r_iq <= 0]
            else:
                constraints += [q_iq.T @ X + r_iq <= 0]
    prob = cp.Problem(objective, constraints)
    if not prob.is_dcp():
        print("Problem is not disciplined convex. No global certificate")
    prob.solve()
    if prob.status not in ['unbounded', 'infeasible']:
        return X.value.reshape(P_.shape[0])
    else:
        print(prob.status, ' CVX min. call at: ')
        print('Center', center)
        print('Radius', radius)
        print('P_', P_)
        print('q_', q_)
        print('r_', r_)
        print('Ineq', ineq)
        raise ValueError
        

def minimise(X_samples, feas_X, infeas_X, g_array, P, q, r, bounds, center, radius, 
             method, ineq_known = None, eq_known = None):
    if (P is None) or (q is None) or (r is None):
        print("P is of type None. Jump step..")
    elif len(feas_X) != len(X_samples) :
        
        all_feas = True
        if method == 'Discrimination':
            try:
                P_ineq, q_ineq, r_ineq = quadratic_discrimination(feas_X, infeas_X)
            except:
                P_ineq, q_ineq, r_ineq = None, None, None
                all_feas = False
            ineq_list = [(P_ineq, q_ineq, r_ineq)]
            # print('Discrimination constants: ', P_ineq, q_ineq, r_ineq)
        
        else:
            ineq_list = []
            n_ineq = g_array.shape[1]
            # print(g_array)
            for i in range(n_ineq):
                g_pred = g_array[:,i]
                try:
                    fitting_out = quadratic_fitting(X_samples, g_pred, discr = True)
                # print(i, fitting_out)
                # print(g_pred)
                    ineq_list += [fitting_out]
                    # print('Yes')
                except:
                    ineq_list += [(None, None, None)]
                    
        if all_feas:
            try:
                center_ = list(quadratic_min(P, q, r, center, radius, bounds, \
                                             ineq = ineq_list, ineq_known = ineq_known, eq_known = eq_known))
            except:
                P = make_PSD(P)
                # print(P)
                try:
                    center_ = list(quadratic_min(P, q, r, center, radius, bounds, \
                               ineq = ineq_list, ineq_known = ineq_known, eq_known = eq_known))
                except:
                    center_ = center
        else:
            try:
                center_ = list(quadratic_min(P, q, r, center, radius, bounds,
                                             ineq_known = ineq_known, eq_known = eq_known))
            except:
                P = make_PSD(P)
                # print(P)
                try:
                    center_ = list(quadratic_min(P, q, r, center, radius, bounds,
                                                 ineq_known = ineq_known, eq_known = eq_known))
                except:
                    center_ = center
    else:
        try:
            center_ = list(quadratic_min(P, q, r, center, radius, bounds,
                                         ineq_known = ineq_known, eq_known = eq_known))
        except:
            P = make_PSD(P)
            # print(P)
            try:
                center_ = list(quadratic_min(P, q, r, center, radius, bounds,
                                             ineq_known = ineq_known, eq_known = eq_known))
            except:
                center_ = center
    return center_

def constr_creation(x, g):
    if g is None:
        if any(isinstance(item, float) for item in x) or any(isinstance(item, int) for item in x):
            feas = 1
        else:
            feas = np.ones(len(np.array(x)))
    elif any(isinstance(item, float) for item in x) or any(isinstance(item, int) for item in x):
        feas = np.product((np.array(g) <= 0).astype(int))
    else:
        feas = np.product( (np.array(g) <= 0).astype(int), axis = 1)
    return feas

def sample_oracle(x, f, ineq = []):
    if any(isinstance(item, float) for item in x) or any(isinstance(item, int) for item in x):
        y = [f(x)]
        if ineq == []:
            g_list = None
        else:
            g_list = [[g_(x) for g_ in ineq]]
    else:
        y = []
        g_list = []
        for x_ in x:
            y += [f(x_)]
            if ineq != []:
                g_list += [[g_(x_) for g_ in ineq]]
    if g_list == []:
        g_list = None
    
    feas = constr_creation(x, g_list)
    
    return y, g_list, feas


def sample_simulation(x, sim):
    f_list = [] ; g_list = []
    if any(isinstance(item, float) for item in x) or any(isinstance(item, int) for item in x):
        obj, constr_vec = sim(x)
        f_list += [obj]
        if constr_vec is not None:
            g_list = [constr_vec]
        # print('Yes')
        
    else:
        for x_ in x:
            obj, constr_vec = sim(x_)
            f_list += [obj]
            if constr_vec is not None:
                g_list += [constr_vec]
    if constr_vec is None:
        g_list = None
    
    feas = constr_creation(x, g_list)
    
    return f_list, g_list, feas


def CUATRO(sim, x0, init_radius, constraints = [], bounds = None, \
           ineq_known = None, eq_known = None, \
           max_f_eval = 100, max_iter = 1000, tolerance = 1e-12, beta_inc = 1.2, \
           beta_red = 0.8, eta1 = 0.2, eta2 = 0.8, method = 'local', \
           N_min_samples = 6, rnd = 1, print_status = False, \
           constr_handling = 'Discrimination'):
    
    X_samples_list = [] ; f_eval_list = [] ; g_eval_list = []
    best_x = [] ; best_f = [] ; best_g = []
    radius_list = [] ; nbr_samples_list = []
    
    np.random.seed(rnd)
    
    center_ = list(x0) ; radius = init_radius
    center = [float(c) for c in center_]
    
    f_eval, g_eval, feas = sample_simulation(center, sim)
    new_f = f_eval[0]
    
    if feas == 0:
        raise ValueError("Please enter feasible starting point")
    
    X_samples_list += [center]
    f_eval_list += [new_f]
    g_eval_list += g_eval
    
    best_x = X_samples_list.copy()
    best_f = f_eval_list.copy()
    best_g = g_eval_list.copy()
        
    radius_list += [init_radius]
    nbr_samples_list += [len(f_eval_list)]
    
    
    
    if method == 'local':
        X_samples, y_samples, g_eval, feas =  sample_points(center, radius, sim, \
                                                            bounds, N = N_min_samples)
    elif method == 'global':
        X_samples, y_samples, g_eval, feas = sample_LHS(sim, bounds, \
                                                        N_min_samples, rnd_seed = rnd)
    else:
        raise ValueError('Invalid input for method')
    
    # feas_samples = oracle.sample_constr(X_samples, g_list = g_eval)
    X_samples_list += X_samples.tolist()
    f_eval_list += y_samples
    g_eval_list += g_eval
    
    old_trust = center
    old_f = best_f[0]

    P, q, r = quadratic_fitting(X_samples, np.array(y_samples))
    feas_X = X_samples.copy()[feas == 1]
    infeas_X = X_samples.copy()[feas != 1]

    if not ((P is None) or (q is None) or (r is None)):
        center_ = minimise(X_samples, feas_X, infeas_X, np.array(g_eval), P, q, \
                           r, bounds, center, radius, constr_handling,
                           ineq_known = ineq_known, eq_known = eq_known)
    else:
        print('P is None in first iteration')
        center_ = list(x0)
    
    center = [float(c) for c in center_]
    
    f_eval, g_eval, new_feas = sample_simulation(center, sim)
    
    new_f = f_eval[0]
    # g_eval = oracle.sample_g(center)
    # print(center)
    # print(g_eval)
    # new_feas = oracle.sample_constr(center, g_list = g_eval) 
    # print(new_feas)
    X_samples_list += [center]
    f_eval_list += [new_f]
    g_eval_list += g_eval
    
    best_x, best_f, best_g = update_best_lists(X_samples_list, \
                                               f_eval_list, g_eval_list,  \
                                               best_x, best_f, best_g)
    
    X = np.array(center).reshape(-1,1)
    new_pred_f = X.T @ P @ X + q.T @ X + r
    X_old = np.array(old_trust).reshape(-1,1)
    old_pred_f = X_old.T @ P @ X_old + q.T @ X_old + r
    
    pred_dec = old_pred_f - new_pred_f ; dec = old_f - new_f
    
    N = 1
    
    while (len(f_eval_list) < max_f_eval - 1) and (N <= max_iter) and (radius > tolerance):
        
        rnd += 1
        np.random.seed(rnd)
        
        if method == 'local':
            if (new_feas == 0) or (new_f - old_f > 0):
                radius *= beta_red
                center = old_trust
            else:
                if (dec >= eta2*pred_dec) and (abs(np.linalg.norm(np.array(old_trust) - np.array(center)) - radius) < 1e-8).any():
                    radius *= beta_inc
                    old_trust = center
                    old_f = new_f

                elif dec <= eta1*pred_dec:
                    radius *= beta_red
                    center = old_trust
                else:
                    old_trust = center
                    old_f = new_f
        else:
            radius *= beta_red
            if (new_feas == 0) or (new_f - old_f > 0):
                center = old_trust
            else:
                old_trust = center
                old_f = new_f
        
        
        radius_list += [radius]
        nbr_samples_list += [len(f_eval_list)]
        
        if P is not None:
            X = np.array(old_trust).reshape(-1,1)
            old_pred_f = X.T @ P @ X + q.T @ X + r
        
        X_in_trust, y_in_trust, g_in_trust, feas_in_trust = samples_in_trust(center, radius, \
                                                                X_samples_list, f_eval_list, g_eval_list)
        N_samples, N_x = X_in_trust.shape
        if N_samples >= N_min_samples:
            N_s = 1
        else:
            N_s = N_min_samples - N_samples
        if (len(f_eval_list) + N_s) > max_f_eval - 1:
            N_s = max(max_f_eval - 1 - len(f_eval_list), 1)
        
        X_samples, y_samples, g_eval, feas_samples =  sample_points(center, radius, sim, \
                                                                    bounds, N = N_s)
        
        X_samples_list += X_samples.tolist()
        f_eval_list += y_samples
        g_eval_list += g_eval
        
        X_samples = np.array(X_in_trust.tolist() + X_samples.tolist())
        y_samples = np.array(y_in_trust.tolist() + y_samples)
        g_samples = np.array(g_in_trust.tolist() + g_eval)
        feas_samples = np.array(feas_in_trust.tolist() + feas_samples.tolist())
        
        try:
            P, q, r = quadratic_fitting(X_samples, y_samples)
        except:
            print('Mosek failed to find convex quadratic fit')
            
        feas_X = X_samples.copy()[feas_samples == 1]
        infeas_X = X_samples.copy()[feas_samples != 1]
    
        # while len(infeas_X) == len(X_samples):
        #     print("No feasible points sampled. Reducing radius and resampling..")
        #     print('Current center: ', center)
        #     print('Iteration: ', N)
        #     X_samples, y_samples, g_eval, feas_samples =  sample_points(center, radius, \
        #                                                   bounds, N = 9)
        #     X_samples_list += X_samples.tolist()
        #     f_eval_list += y_samples
        #     g_eval_list += g_eval

            
        
        if not ((P is None) or (q is None) or (r is None)):
            
            center_ = minimise(X_samples, feas_X, infeas_X, g_samples, P, q, r, bounds, \
                               center, radius, constr_handling,
                               ineq_known = ineq_known, eq_known = eq_known)
            
            center = [float(c) for c in center_]
        
            f_eval, g_eval, new_feas = sample_simulation(center, sim)
            new_f = f_eval[0]
        
            X_samples_list += [center]
            f_eval_list += [new_f]
            g_eval_list += g_eval
            X = np.array(center).reshape(-1,1)
            new_pred_f = X.T @ P @ X + q.T @ X + r
    
            pred_dec = old_pred_f - new_pred_f ; dec = old_f - new_f
        
        best_x, best_f, best_g = update_best_lists(X_samples_list, \
                                               f_eval_list, g_eval_list,  \
                                               best_x, best_f, best_g)
            
        N += 1
    
    N_evals = len(f_eval_list)
    radius_list += [radius] 
    nbr_samples_list += [len(f_eval_list)]
   
    if N > max_iter:
        status = "Max # of iterations reached"
    elif radius < tolerance:
        status = "Radius below threshold"
    else:
        status = "Max # of function evaluations"
    
    if print_status:
        print('Minimisation terminated: ', status)
    
    output = {'x_best_so_far': best_x, 'f_best_so_far': best_f, \
              'g_best_so_far': best_g, 'x_store': X_samples_list, \
              'f_store': f_eval_list, 'g_store': g_eval_list, \
              'N_eval': N_evals, 'N_iter': N, 'TR': radius_list, \
              'samples_at_iteration': nbr_samples_list}
    
    return output
