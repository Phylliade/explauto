import matplotlib.pyplot as plt
from explauto import Environment
from explauto import SensorimotorModel
from explauto.interest_model.random import RandomInterest
import numpy as np
from explauto.sensorimotor_model.inverse.cma import fmin as cma_fmin
import pickle


height = True
test_set_size = 100

environment = Environment.from_configuration('gym', 'MCC_height')
m = environment.random_motors(1)
s = environment.compute_sensori_effect(m[0])

if False:
    model = SensorimotorModel.from_configuration(environment.conf, 'LWLR-BFGS', 'default')
else:
    model = SensorimotorModel.from_configuration(
        environment.conf,
        'nearest_neighbor',
        'default'
    )
model.mode = "exploit"
# We don't want the sensorimotor model to add exploration noise

motor_commands = []

# Bootstrap
for m in environment.random_motors(n=10):
    s = environment.compute_sensori_effect(m)
    # environment.plot_arm(ax, m, alpha=0.2)
    print("Achievement: {}".format(s))
    model.update(m, s)

im_model = RandomInterest(environment.conf, environment.conf.s_dims)

n_goals = 40
# Number of goals
cma_maxfevals = 50
# Maximum error function evaluations by CMAES (actually CMAES will slightly overshoot it)
cma_sigma0 = 0.2
# Standard deviation in initial covariance matrix

errors_valid = []
errors = []
errors_full = []

if height:
    goals = []
    achievements = []

else:
    goals_x = []
    goals_y = []

for i in range(n_goals):
    s_g = im_model.sample()
    # Sample a random goal
    m0 = model.inverse_prediction(s_g)

    # Find the nearest neighbor of s_g and output the corresponding m

    def error_f(m_):
        # Error function corresponding to the new goal s_g.
        s_ = environment.compute_sensori_effect(m_)
        # Execute a motor command
        model.update(m_, s_)
        # Update the surrogate model
        return np.linalg.norm(s_ - s_g)
        # Output the distance between the reached point s_ and the goal s_g

    # Call CMAES with the error function for the new goal and use m0 to bootstrap exploration

    m = cma_fmin(
        error_f,
        m0,
        cma_sigma0,
        options={
            'bounds': [environment.conf.m_mins, environment.conf.m_maxs],
            'verb_log': 0,
            'verb_disp': False,
            'maxfevals': cma_maxfevals
        })[0]

    s = environment.compute_sensori_effect(m)
    # Execute best motor command found by CMAES (optional)
    model.update(m, s)
    # Update the surrogate model



    error = np.linalg.norm(s_g - s)
    print("Iteration {} out of {}; Achievement: {}; Goal: {}; Reaching error: {}".format(i, n_goals, s, s_g, error))

    # Collect statistics
    errors_full.append(error)

    # Add the motor command
    motor_commands.append(m)

    if n_goals - i < test_set_size:
        errors.append(error)

        if height:
            goals.append(s_g[0])
            achievements.append(s[0])
        else:
            if s_g[0] <= s_g[1]:
                errors_valid.append(error)
            goals_x.append(s_g[0])
            goals_y.append(s_g[1])


# plt.plot(errors_valid)
# plt.savefig("evolution.png")


ax = plt.subplot()

if True:
    if height:
        er = ax.scatter(goals, errors)
        go = ax.scatter(goals, goals)
        ac = ax.scatter(goals, achievements)
        ax.legend((er, go, ac), ("Errors", "Goals", "Achievements"))
        plt.xlabel("Target Goal")
        plt.ylabel("Achieved Goal")

    else:
        sc = plt.scatter(goals_x, goals_y, c=errors)
        plt.colorbar(sc)
        plt.xlabel("min")
        plt.ylabel("max")

plt.savefig("scatter.png")

# Errors
fig_errors = plt.figure()
plt.plot(errors_full)
plt.xlabel("Time")
plt.ylabel("Errors")
plt.savefig("errors.png")

# Parameter space exploration
plt.figure()
motor_commands = np.array(motor_commands)
sc = plt.scatter(motor_commands[:, 0], motor_commands[:, 1], c=errors_full)
plt.colorbar(sc)
plt.title("Parameter space exploration")
plt.savefig("parameter_space.png")

with open("errors.p", "wb") as fd:
    pickle.dump(errors, fd)

with open("errors_full.p", "wb") as fd:
    pickle.dump(errors, fd)
