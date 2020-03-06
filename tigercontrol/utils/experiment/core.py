def get_trajectory(environment, controller, T = 100):

    (environment_id, environment_params) = environment
    (controller_id, controller_params) = controller

    if(controller_params is None):
        controller_params = {}
    
    environment = tigercontrol.environment(environment_id)
    x = environment.reset(**environment_params)
    
    controller_params['A'], controller_params['B'] = environment.A, environment.B
   
    controller = tigercontrol.controller(controller_id)
    controller.initialize(**controller_params)
    
    trajectory1, trajectory2 = [], []
    norms = []
    avg_regret = []
    cur_avg = 0

    for i in range(T):
        u = controller.get_action(x)
        x = environment.step(u)
        trajectory1.append(x)
        norms.append(np.linalg.norm(x))
        cur_avg = (i / (i + 1)) * cur_avg + (np.linalg.norm(x) + np.linalg.norm(u)) / (i + 1)
        avg_regret.append(cur_avg)

    return trajectory1, norms, avg_regret

def to_dataframe(alg, loss):
    inst_loss, avg_loss = loss
    return pd.DataFrame(data = {'Algorithm': alg, 'Time': np.arange(T, dtype=np.float32),
                                'Instantaneous Cost': inst_loss, 'Average Cost': avg_loss})

def benchmark(A, B, Wgen, cost_fn = quad):

    loss_lqr = evaluate(LQR(A, B), A, B, Wgen, cost_fn)
    loss_gpc = evaluate(GPC(A, B, cost_fn=cost_fn), A, B, Wgen, cost_fn)
    loss_bpc = evaluate(BPC(A, B), A, B, Wgen, cost_fn)

    return loss_lqr, loss_gpc_1, loss_gpc_2, loss_bpc

def repeat_benchmark(A, B, Wgen, rep = 3, cost_fn = quad):
    all_data = []
    for r in range(rep):
        loss = benchmark(A, B, Wgen, cost_fn)
        data = pd.concat(list(map(lambda x: to_dataframe(*x), list(zip(alg_name, loss)))))
        all_data.append(data)
    all_data = pd.concat(all_data)
    return all_data[all_data['Instantaneous Cost'].notnull()]

def plot(title, data, scale = 'linear'):
    fig, axs = plt.subplots(ncols=2, figsize=(15,4))
    axs[0].set_yscale(scale)
    sns.lineplot(x = 'Time', y = 'Instantaneous Cost', hue = 'Algorithm', 
                 data = data, ax = axs[0], ci = 'sd', palette = color_code).set_title(title)
    axs[1].set_yscale(scale)
    sns.lineplot(x = 'Time', y = 'Average Cost', hue = 'Algorithm', 
                 data = data, ax = axs[1], ci = 'sd', palette = color_code).set_title(title)


class Wgen:
    def __init__(self, n, m):
        global T
        self.t = -1
        self.w = (np.sin(np.arange(T*m)/(32*np.pi)).reshape(T,m) @ np.ones((m, n))).reshape(T, n, 1)

    def next(self):
        self.t += 1
        return self.w[self.t]

class Wgen:
    def __init__(self, n, m):
        global T
        self.t = -1
        W = random.normal(size = (T, n, 1), scale = 1/T**(0.5))
        for i in range(1, T):
            W[i] = W[i] + W[i-1]
        self.w = W

    def next(self):
        self.t += 1
        return self.w[self.t]


class Wgen:
    def __init__(self, n, m):
        global T
        self.t = -1
        W = random.normal(size = (T, n, 1), scale = 1/T**(0.5))
        for i in range(1, T):
            W[i] = W[i] + W[i-1]
        self.w = W

    def next(self):
        self.t += 1
        return self.w[self.t]