# ctsb.help() method

from ctsb import error, logger, problem_registry


def help():
    s = "\n"
    for problem_id in problem_registry.keys():
        s += "\t" + problem_id + "\n"
    print(global_help_string.format(s))



global_help_string = """

Welcome to CTSB - the Control and Time-Series Benchmarks framework!

If this is your first time using CTSB, you might want to read more about it in 
detail at https://github.com/johnhallman/ctsb.

If you want to get going immediately, a good place to start is with our Problem 
class, which provide implementations of standard control and time-series benchmarks.
In order to return a Problem instance, simply call ctsb.make(*problem name*). This
is the list of currently available Problems:
{}
For example, you can retrieve an linear dynamical system instance via:

    'problem = ctsb.problem("LDS-v0")'

Before you can start using your problem to benchmark your algorithm, you must
initialize it by providing relevant dimensionality and other parameters. For our
LDS instance, this corresponds to the input, output, and hidden state dimension 
respectively:

    'd_in, d_out, d_hid = 3, 1, 5'
    'observation = problem.initialize(d_in, d_out, d_hid)'

Notice that initializing the problem statement causes it to return the first value
of the resulting time-series! Now, the parameters for 'initialize' vary from 
problem to problem. To learn about the specific requirements, or the dynamics, of 
your selected problem instance, call its 'help' method:

    'problem.help()'

Once you have initialized your problem, you can move the system forward one
time-step by calling the 'step' method along with the appropriate inputs:

    'action = np.zeros(d_in)'
    'next_observation = problem.step(action)'

Continue using 'step' to move the dynamics forward for as long as you need to
test your model!

"""


