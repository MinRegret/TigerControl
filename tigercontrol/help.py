# tigercontrol.help() controller

from tigercontrol import environment_registry, controller_registry


def help():
    s_prob, s_mod = "\n", "\n"
    for environment_id in environment_registry.list_ids():
        s_prob += "\t" + environment_id + "\n"
    for controller_id in controller_registry.list_ids():
        s_mod += "\t" + controller_id + "\n"
    print(global_help_string.format(s_prob, s_mod))



global_help_string = """

Welcome to TigerControl!

If this is your first time using TigerControl, you might want to read more about it in 
detail at github.com/johnhallman/tigercontrol, or documentation at tigercontrol.readthedocs.io.

If you're looking for a specific Environment or Controller, you can call it via the 
tigercontrol.environment and tigercontrol.controllers controllers respectively, such as:

    environment = tigercontrol.environment("nameOfController")

Below is the list of all currently avaliable environments and controllers:

    Environments
    ---------
    {}

    Controllers
    ---------
    {}

Good luck exploring TigerControl!

"""


