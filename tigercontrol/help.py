# tigercontrol.help() method

from tigercontrol import problem_registry, model_registry


def help():
    s_prob, s_mod = "\n", "\n"
    for problem_id in problem_registry.list_ids():
        s_prob += "\t" + problem_id + "\n"
    for model_id in model_registry.list_ids():
        s_mod += "\t" + model_id + "\n"
    print(global_help_string.format(s_prob, s_mod))



global_help_string = """

Welcome to TigerControl - the Control and Time-Series Benchmarks framework!

If this is your first time using TigerControl, you might want to read more about it in 
detail at github.com/johnhallman/tigercontrol, or documentation at tigercontrol.readthedocs.io.

If you're looking for a specific Problem or Model, you can call it via the 
tigercontrol.problem and tigercontrol.model methods respectively, such as:

    problem = tigercontrol.problem("nameOfModel")

Below is the list of all currently avaliable problems and models:

    Problems
    ---------
    {}

    Models
    ---------
    {}

Good luck exploring TigerControl!

"""


