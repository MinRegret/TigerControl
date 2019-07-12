# ctsb.help() method

from ctsb import problem_registry


def help():
    s_prob, s_mod = "\n", "\n"
    for problem_id in problem_registry.list_ids():
        s_prob += "\t" + problem_id + "\n"
    for model_id in model_registry.list_ids():
        s_mod += "\t" + model_id + "\n"
    print(global_help_string.format(s_prob, s_mod))



global_help_string = """

Welcome to CTSB - the Control and Time-Series Benchmarks framework!

If this is your first time using CTSB, you might want to read more about it in 
detail at https://github.com/johnhallman/ctsb.

If you want to get going immediately, a good place to start is with our Problem 
class, which provide implementations of standard control and time-series benchmarks.
In order to return a Problem instance, simply call ctsb.problem(*problem name*). This
is the list of currently available Problems:
{}
For example, you can retrieve an linear dynamical system instance via:

    problem = ctsb.problem("LDS-v0")

Before you can start using your problem to benchmark your algorithm, you must
initialize it by providing relevant dimensionality and other parameters. For our
LDS instance, this corresponds to the input, output, and hidden state dimension 
respectively:

    d_in, d_out, d_hid = 3, 1, 5
    observation = problem.initialize(d_in, d_out, d_hid)

Notice that initializing the problem statement causes it to return the first value
of the resulting time-series! Now, the parameters for 'initialize' vary from 
problem to problem. To learn about the specific requirements, or the dynamics, of 
your selected problem instance, call its 'help' method:

    problem.help()

Once you have initialized your problem, you can move the system forward one
time-step by calling the 'step' method along with the appropriate inputs:

    action = np.zeros(d_in)
    next_observation = problem.step(action)

In order to select actions, consider using our preimplemented model, which you
may call in the same manner as with the problems:

    d_hid = 3
    model = ctsb.model("AutoRegressor")
    model.initialize(d_hid)

This is the list of currently available Models:
{}
Models come with three core methods — initialize, predict, and update. Initialize
is called first in order to initialize and store internal parameters of the model,
predict takes an input observation and returns a prediction for the dependent
variables, and update adjusts the internal parameters of the model.

Below is an example of a model applied to an ARMA time-series:

    T = steps 
    p, q = 3, 0
    problem = ctsb.problem("ARMA-v0")
    cur_x = problem.initialize(p, q)
    model = ctsb.model("AutoRegressor")
    model.initialize(p)
    loss = lambda y_true, y_pred: (y_true - y_pred)**2
 
    results = []
    for i in range(T):
        cur_y_pred = model.predict(cur_x)
        cur_y_true = problem.step()
        cur_loss = loss(cur_y_true, cur_y_pred)
        results.append(cur_loss)
        model.update(cur_loss)
        cur_x = cur_y_true

Good luck exploring CTSB!

"""


