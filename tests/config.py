import config_local

n_goals = 1000
n_goals_test = 100
n_motor_commands = 100
test_set_size = 100


local_variables = ["n_goals", "n_goals_test", "n_motor_commands", "test_set_size"]
# Overwrite variables with the ones provided in config_local
variables = locals()
for variable in local_variables:
    variables[variable] = getattr(config_local, variable, variables[variable])


n_goals_total = n_goals + n_goals_test
