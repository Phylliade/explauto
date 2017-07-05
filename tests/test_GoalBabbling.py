import matplotlib.pyplot as plt
from explauto import Environment
from explauto import SensorimotorModel
from explauto.interest_model.random import RandomInterest
import numpy as np
from explauto.sensorimotor_model.inverse.cma import fmin as cma_fmin
import pickle
import config


environment = Environment.from_configuration('gym', 'MCC_max_pos_tanh')

if False:
    model = SensorimotorModel.from_configuration(environment.conf, 'LWLR-BFGS', 'default')
else:
    model = SensorimotorModel.from_configuration(
        environment.conf,
        'nearest_neighbor',
        'default'
    )


motor_commands_full = []

# Bootstrap
for m in environment.random_motors(n=config.n_motor_commands):
    s = environment.compute_sensori_effect(m)
    print("Achievement: {}".format(s))
    model.update(m, s)

im_model = RandomInterest(environment.conf, environment.conf.s_dims)

cma = False

# CMA-ES Parameters
# Maximum error function evaluations by CMAES (actually CMAES will slightly overshoot it)
cma_maxfevals = 50
# Standard deviation in initial covariance matrix
cma_sigma0 = 0.2

errors = []
errors_full = []
goals = []
achievements = []

# Train
try:
    for i in range(config.n_goals):
        # Sample a random goal
        s_g = im_model.sample()

        # Inverse it
        m0 = model.inverse_prediction(s_g)

        if i == 900:
            model.mode = "exploit"

        if cma:
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
        else:
            m = m0

        # Execute best motor command found
        s = environment.compute_sensori_effect(m)

        # Update the surrogate model
        model.update(m, s)

        error = np.linalg.norm(s_g - s)
        print("Iteration {} out of {}; Achievement: {}; Goal: {}; Reaching error: {}".format(i + 1, config.n_goals, s, s_g, error))

        # Collect statistics
        errors_full.append(error)
        motor_commands_full.append(m)

        if config.n_goals - i < config.test_set_size:
            errors.append(error)
            goals.append(s_g[0])
            achievements.append(s[0])


except KeyboardInterrupt:
    pass

# Test
errors = []
goals = []
achievements = []

# Remove exploration noise
model.mode = "exploit"

for i in range(config.n_goals_test):
    # Sample a random goal
    goal = im_model.sample()

    # Inverse it
    m = model.inverse_prediction(goal)

    # Compute the achievement and store it in the database
    achievement = environment.compute_sensori_effect(m)
    model.update(m, achievement)

    # Compute error
    error = np.linalg.norm(goal - achievement)

    print("Iteration {} out of {}; Achievement: {}; Goal: {}; Reaching error: {}".format(i + 1, config.n_goals_test, achievement, goal, error))

    # Collect statistics
    errors_full.append(error)
    motor_commands_full.append(m)
    errors.append(error)
    goals.append(goal[0])
    achievements.append(achievement[0])


# Save the replay buffer
environment.save_replay_buffer()

# Process the data
goals = np.array(goals)
errors = np.array(errors)
achievements = np.array(achievements)
motor_commands_full = np.array(motor_commands_full)


# Plots
# Learning distribution among the goals
ax = plt.subplot()
# Display only the first goal
if goals.ndim != 1:
    goals = goals[:, 0]
    errors = errors[:, 0]
    achievements = achievements[:, 0]
er = ax.scatter(goals, errors)
go = ax.scatter(goals, goals)
ac = ax.scatter(goals, achievements)
ax.legend((er, go, ac), ("Errors", "Goals", "Achievements"))
plt.xlabel("Target Goal")
plt.ylabel("Achieved Goal")

plt.savefig("success.png")

# Evolution of the errors
fig_errors = plt.figure()
plt.plot(errors_full)
plt.xlabel("Time")
plt.ylabel("Errors")
plt.savefig("errors.png")

# Parameter space exploration
plt.figure()
sc = plt.scatter(motor_commands_full[:, 0], motor_commands_full[:, 1], c=errors_full)
plt.colorbar(sc)
plt.title("Parameter space exploration")
plt.savefig("parameter_space.png")

with open("errors.p", "wb") as fd:
    pickle.dump(errors, fd)

with open("errors_full.p", "wb") as fd:
    pickle.dump(errors_full, fd)
