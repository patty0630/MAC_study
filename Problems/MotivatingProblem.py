import pyomo.environ as pyo


def centralised(data):
    model = pyo.AbstractModel()
    model.i = pyo.RangeSet(1, 2)
    model.x_init = pyo.Param(model.i)
    model.x3 = pyo.Param()
    model.x = pyo.Var(model.i, within = pyo.Reals, bounds = (-10, 10))

    
    def o(m):
        return (m.x[1] - 7)**2 + (m.x[1]*m.x3-3)**2 + (m.x[2]+2)**2 + \
                (m.x[2]*m.x3-2)**2
    model.obj = pyo.Objective(rule = o)
    
    def h(m):
        return m.x[1] + m.x3 == 5
    model.eq1 = pyo.Constraint(rule = h)
    
    ins = model.create_instance(data)
    
    for j in ins.i:
        ins.x[j] = ins.x_init[j]
    
    #solver = pyo.SolverFactory('cbc')
    solver = pyo.SolverFactory('ipopt')
    solver.solve(ins)
    #print(pyo.value(ins.obj))
    #exit()
    
    return ins


def SP1(data, return_solver = False):
    model = pyo.AbstractModel()
    #model.x_init = pyo.Param()
    model.z_I = pyo.Set()
    model.z = pyo.Param(model.z_I)
    model.u = pyo.Param(model.z_I)
    model.rho = pyo.Param()
    model.I = pyo.Set()
    model.x = pyo.Var(model.I, within = pyo.Reals, bounds = (-10, 10))

    if return_solver:
        def o(m):
            return (m.x[1] - 7)**2 + (m.x[1]*m.z[3]-3)**2 + \
                m.rho/2*sum((m.x[j] - m.z[j] + m.u[j])**2 for j in m.z_I)
        model.obj = pyo.Objective(rule = o)
    else:
        def o(m):
            return (m.x[1] - 7)**2 + (m.x[1]*m.x[3]-3)**2 + \
                m.rho/2*sum((m.x[j] - m.z[j] + m.u[j])**2 for j in m.z_I)
        model.obj = pyo.Objective(rule = o)
    
    def g(m):
        return -m.x[1] <= 0
    model.ineq1 = pyo.Constraint(rule = g)

    if return_solver:
        def h(m):
            return m.x[1] + m.z[3] == 5
        model.eq1 = pyo.Constraint(rule = h)
    else:
        def h(m):
            return m.x[1] + m.x[3] == 5
        model.eq1 = pyo.Constraint(rule = h)

    ins = model.create_instance(data)
    
    #ins.x = ins.x_init
    
    #solver = pyo.SolverFactory('cbc')
    solver = pyo.SolverFactory('ipopt')
    res = solver.solve(ins)
    
    if return_solver:
        return ins, res
    else:
        return ins

def SP2(data, return_solver = False):
    model = pyo.AbstractModel()
    #model.x_init = pyo.Param()
    model.z_I = pyo.Set()
    model.z = pyo.Param(model.z_I)
    model.u = pyo.Param(model.z_I)
    model.rho = pyo.Param()
    model.I = pyo.Set()
    model.x = pyo.Var(model.I, within = pyo.Reals, bounds = (-10, 10))

    if return_solver:
        def o(m):
            return (m.x[2] + 2)**2 + (m.x[2]*m.z[3]-2)**2 + \
                m.rho/2*sum((m.x[3] - m.z[j] + m.u[j])**2 for j in m.z_I)
        model.obj = pyo.Objective(rule = o)
    else:
        def o(m):
            return (m.x[2] + 2)**2 + (m.x[2]*m.x[3]-2)**2 + \
                m.rho/2*sum((m.x[3] - m.z[j] + m.u[j])**2 for j in m.z_I)
        model.obj = pyo.Objective(rule = o)

    ins = model.create_instance(data)
    
    #ins.x = ins.x_init
       
    #solver = pyo.SolverFactory('cbc')
    solver = pyo.SolverFactory('ipopt')
    res = solver.solve(ins)
    
    if return_solver:
        return ins, res
    else:
        return ins


def f1(z_list, rho, global_ind, index, u_list = None, solver = False):
    
    data = {None: {
                  'z': {}, 'u': {}, 'z_I': {None: global_ind}, 
                  'rho': {None: rho}, 'I': {None: index}
                }
          }
    
    for idx in global_ind:
        data[None]['z'][idx] = z_list[idx][-1]
        if u_list is not None:
            data[None]['u'][idx] = u_list[idx][-1]
        else:
            data[None]['u'][idx] = 0
    
    return SP1(data, return_solver = solver)

def f2(z_list, rho, global_ind, index, u_list = None, solver = False):

    data = {None: {
                  'z': {}, 'u': {}, 'z_I': {None: global_ind}, 
                  'rho': {None: rho}, 'I': {None: index}
                }
          }
    
    for idx in global_ind:
        data[None]['z'][idx] = z_list[idx][-1]
        if u_list is not None:
            data[None]['u'][idx] = u_list[idx][-1]
        else:
            data[None]['u'][idx] = 0

    return SP2(data, return_solver = solver)
