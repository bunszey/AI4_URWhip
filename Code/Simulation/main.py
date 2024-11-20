from utility.simulation import Simulation
import numpy as np
import mediapy as media

import mujoco

def main():

    args = lambda: None
    args.model_name = "scene"
    
    simulation = Simulation(args)
    

    qpos = [0.5*np.pi,0,-0.25*np.pi,-0.5*np.pi,0,0]
    qvel = [0,0,0,0,0,0]

    simulation.set_init_posture(qpos,qvel)
    simulation.make_whip_downwards()

    action = simulation.random_action()
    simulation.set_init_posture(action[0:6],action[6:12])
    simulation.make_whip_downwards()
    simulation.planner.set_positions(action[0:6],action[6:12])
    simulation.planner.set_duration(action[12])

    while simulation.data.time < simulation.planner.D:
        simulation.controller()
        mujoco.mj_step(simulation.model, simulation.data)
        simulation.add_frame_if_needed()

    
    print("action: ",action)
    simulation.render_video()
    simulation.write_image("output_frame.png")
    

if __name__ == '__main__':
    main()