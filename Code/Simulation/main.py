from utility.simulation import Simulation
import numpy as np
import mediapy as media

def main():
    args = lambda: None
    args.model_name = "scene"
    
    simulation = Simulation(args)
    qpos = [0.5*np.pi,0,-0.25*np.pi,-0.5*np.pi,0,0]
    qvel = [0,0,0,0,0,0]

    simulation.set_init_posture(qpos,qvel)
    simulation.make_whip_downwards()


    simulation.write_image("output_frame.png")
    


if __name__ == '__main__':
    main()