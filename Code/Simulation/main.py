from utility.simulation import Simulation
from utility.replay_buffer import ReplayBuffer
import numpy as np
import mediapy as media
import math
import mujoco
import time
import matplotlib.pyplot as plt
import os
def main():

    args = lambda: None
    args.model_name = "scene"
    
    simulation = Simulation(args)
    
    path = "./results/"
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    folders.sort()
    folder = folders[-2]
    print("folder: ", folder)

    rb = ReplayBuffer(simulation.s_dim, simulation.a_dim)
    rb.load(path + folder + "/SAC_training_replay_buffer.npz")

    states = rb.state
    states = states[~np.all(states == 0, axis=1)]

    actions = rb.action
    actions = actions[~np.all(actions == 0, axis=1)]



    state = simulation.unscale_observation(states[-1])
    action = simulation.unscale_action(actions[-1])

    simulation.init_action(state, action)

    desired_positions = []
    actual_positions = []
    position_error = []

    K = []
    B = []

    while simulation.data.time < action[-1] + simulation.extra_run_time:
        desired_position = simulation.planner.get_desired_position(simulation.data.time)
        desired_positions.append(desired_position)

        actual_position = simulation.data.qpos.copy()
        actual_positions.append(actual_position[:6])

        position_error.append(desired_position - actual_position[:6])

        simulation.controller()
        K.append(simulation.K[0][0])
        B.append(simulation.B[0][0])
        
        mujoco.mj_step(simulation.model, simulation.data)

    
    actual_positions_no_oiac = []
    position_error_no_oiac = []
    simulation.init_action(state, action)

    while simulation.data.time < action[-1] + simulation.extra_run_time:
        desired_position = simulation.planner.get_desired_position(simulation.data.time)

        actual_position = simulation.data.qpos.copy()
        actual_positions_no_oiac.append(actual_position[:6])

        position_error_no_oiac.append(desired_position - actual_position[:6])

        simulation.controller(use_oiac=False)
        
        mujoco.mj_step(simulation.model, simulation.data)


    np.save(path + folder + "/desired_positions.npy", desired_positions)
    np.save(path + folder + "/actual_positions.npy", actual_positions)
    np.save(path + folder + "/position_error.npy", position_error)
    np.save(path + folder + "/actual_positions_no_oiac.npy", actual_positions_no_oiac)
    np.save(path + folder + "/position_error_no_oiac.npy", position_error_no_oiac)
    np.save(path + folder + "/K.npy", K)

    position_error = position_error[0:400]
    position_error_no_oiac = position_error_no_oiac[0:400]
    K = K[0:400]
    B = B[0:400]

    plt.plot([x[0] for x in position_error], label="Position error with OIAC")
    plt.plot([x[0] for x in position_error_no_oiac], label="Position error with constant K and B")

    plt.xlabel("Time step")
    plt.ylabel("Position error (rad)")
    plt.legend()
    plt.grid()

    plt.savefig(path + folder + "/position_error.pdf")
    plt.close()

    #plot change in K
    plt.plot(K)
    plt.plot(B)
    plt.xlabel("Time step")
    plt.ylabel("Gain")
    plt.grid()
    plt.legend(["K", "B"])
    plt.savefig(path + folder + "/KB.pdf")
    plt.close()




if __name__ == '__main__':
    main()

    