# ctsb.help() method

from ctsb import problem_registry, model_registry


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
detail at github.com/johnhallman/ctsb, or documentation at ctsb.readthedocs.io.

If you're looking for a specific Problem or Model, you can call it via the 
ctsb.problem and ctsb.model methods respectively, such as:

    problem = ctsb.problem("nameOfModel")

Below is the list of all currently avaliable problems and models:

    Problems
    ---------
    {}

    Models
    ---------
    {}

Good luck exploring CTSB!

"""


