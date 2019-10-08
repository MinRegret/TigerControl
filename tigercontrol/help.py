# tigercontrol.help() method

from tigercontrol import problem_registry, method_registry


def help():
    s_prob, s_mod = "\n", "\n"
    for problem_id in problem_registry.list_ids():
        s_prob += "\t" + problem_id + "\n"
    for method_id in method_registry.list_ids():
        s_mod += "\t" + method_id + "\n"
    print(global_help_string.format(s_prob, s_mod))



global_help_string = """

Welcome to TigerControl!

If this is your first time using TigerControl, you might want to read more about it in 
detail at github.com/johnhallman/tigercontrol, or documentation at tigercontrol.readthedocs.io.

If you're looking for a specific Problem or Method, you can call it via the 
tigercontrol.problem and tigercontrol.method methods respectively, such as:

    problem = tigercontrol.problem("nameOfMethod")

Below is the list of all currently avaliable problems and methods:

    Problems
    ---------
    {}

    Methods
    ---------
    {}

Good luck exploring TigerControl!

"""


