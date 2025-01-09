from utility.simulation import Simulation
import numpy as np
import mediapy as media
import math
import mujoco
import time
def main():

    args = lambda: None
    args.model_name = "scene"
    args.scene_folder = "../../Models/universal_robots_ur5e/"#_image/"

    simulation = Simulation(args)


    init_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.pi, np.pi]
    init_pos[:6] = [1.57079632679, -0.1521357693, 0.214522435, -1.57079632679, -1.57079632679, 0]
   

    init_vel = np.zeros(len(init_pos))
    
    
    simulation.reset_simulation()

    simulation.set_init_posture(init_pos, init_vel)

    simulation.move_geom("geom_target", [1.8, 0.0, 0.0])


    simulation.write_image(image_name="urmodel_new_pos.png",camera="fixedy")
    




if __name__ == '__main__':
    main()

    