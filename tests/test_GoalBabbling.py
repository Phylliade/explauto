from explauto import Environment
from explauto import SensorimotorModel
from explauto.interest_model.random import RandomInterest
import numpy as np
from explauto.sensorimotor_model.inverse.cma import fmin as cma_fmin
import matplotlib.pyplot as plt

environment = Environment.from_configuration('gym', 'MCC')
m = environment.random_motors(1)
s = environment.compute_sensori_effect(m[0])

model = SensorimotorModel.from_configuration(environment.conf,
                                             'nearest_neighbor', 'default')
model.mode = "exploit"
# We don't want the sensorimotor model to add exploration noise

# Bootstrap
for m in environment.random_motors(n=100):
    s = environment.compute_sensori_effect(m)
    # environment.plot_arm(ax, m, alpha=0.2)
    model.update(m, s)

im_model = RandomInterest(environment.conf, environment.conf.s_dims)

n_goals = 5000
# Number of goals
cma_maxfevals = 50
# Maximum error function evaluations by CMAES (actually CMAES will slightly overshoot it)
cma_sigma0 = 0.2
# Standard deviation in initial covariance matrix

errors_valid = []
errors = []
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
    print("Goal:", s_g, "Reaching Error:", error)

    # Append errors only for valid goals
    if s_g[0] <= s_g[1]:
        errors_valid.append(error)

    goals_x.append(s_g[0])
    goals_y.append(s_g[1])
    errors.append(error)

#plt.plot(errors_valid)
#plt.savefig("evolution.png")
plt.scatter(goals_x, goals_y, c=errors)
plt.xlabel("min")
plt.ylabel("max")
plt.savefig("scatter.png")
