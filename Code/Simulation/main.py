from utility.simulation import Simulation
import numpy as np
import mediapy as media
import math
import mujoco

def main():

    args = lambda: None
    args.model_name = "scene"
    
    simulation = Simulation(args)


    action = simulation.random_action()
    action = [-1.39095287,  3.4619481,  -0.21930487 , 0.89649079 ,-1.92544355 ,-2.03098799 ,-2.02422557, -0.41235709, -1.37341055 , 1.77293438 ,-2.7159907 , -1.96190319, 0.75030443]

    for i in range(100):

        target_position = [1.60304854, 0.07446196, 0.01489245]

        make_video_of_action(simulation, action, target_position)



        

        #wait for user input
        input("Press Enter to continue...")


    

def make_video_of_action(simulation, action, target_position):
    simulation.reset_frames()
    mujoco.mj_resetData(simulation.model, simulation.data)
    
    simulation.move_geom("geom_target", target_position[0:3])

    q_init_vel = [0,0,0,0,0,0]
    simulation.set_init_posture(action[0:6], q_init_vel)
    simulation.make_whip_downwards()
    
    simulation.set_color_geom("geom_target", [0.0, 0.4470, 0.7410, 1.0])

    mujoco.mj_forward(simulation.model, simulation.data)


    simulation.planner.set_positions(action[0:6],action[6:12])
    simulation.planner.set_duration(action[12])
    
    simulation.extra_run_time = 2.0
    positions = []

    while simulation.data.time < simulation.planner.D + simulation.extra_run_time:
        simulation.controller()
        mujoco.mj_step(simulation.model, simulation.data)
        
        
        pos = simulation.get_pos_if_box_hit()
        if pos is not None:
            positions.append(pos.copy())


        if simulation.get_if_hit_target():
            #change color of target
            simulation.set_color_geom("geom_target", [0.0,1.0,0.0,1.0])
            #make a image from each axis camera
            simulation.write_image(image_name="output_frame_test_x.png",camera="fixedx")
            simulation.write_image(image_name="output_frame_test_y.png",camera="fixedy")
            simulation.write_image(image_name="output_frame_test_z.png",camera="fixedz")

        simulation.add_frame_if_needed(camera="fixed")

    
    print("action: ",action)
    print("positions: ",positions)
    simulation.render_video()


if __name__ == '__main__':
    main()

    