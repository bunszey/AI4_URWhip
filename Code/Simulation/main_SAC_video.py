from utility.simulation import Simulation
import numpy as np
import mediapy as media
import math
import mujoco
import time
import os
from RL.SAC import *
import random
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

def main():

    args = lambda: None
    args.model_name = "scene"

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    seed = 10
    discount = 0.99  # Discount factor
    tau = 0.005     # Target network update rate

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    simulation = Simulation(args)
    simulation.extra_run_time = 1.0
    simulation.fps = 120

    state_dim = simulation.s_dim
    action_dim = simulation.a_dim

    path = "./results/"
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    folders.sort()
    folder = folders[-2]
    print("folder: ", folder)

    subfolder = "SAC_video_data"
    

    text = "Project in AI4"
    text2 = "Deformable object manipulation"
    text3 = "SAC learning"
    text4 = "Episode 0, 99, 199, 299, 399, 499"

    offset = 300

    frontpage = np.ones((1300, 1920, 3), np.uint8) * 255
    frontpage = cv2.putText(frontpage, text, (30, 0+offset), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2, cv2.LINE_AA)
    frontpage = cv2.putText(frontpage, text2, (30, 100+offset), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2, cv2.LINE_AA)
    frontpage = cv2.putText(frontpage, text3, (30, 180+offset), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
    frontpage = cv2.putText(frontpage, text4, (30, 240+offset), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)

    
    frames_frontpage = []
    for i in range(120*2):
        frames_frontpage.append(frontpage)

    media.write_video(path + folder + "/SAC_frontpage.mp4", frames_frontpage, fps=simulation.fps )
    
    # Choose episodes that should be 
    episodes = [0, 99, 199, 299, 399, 499]

    for episode in episodes:
        
        target_positions = np.load(path + folder + "/SAC_target_positions.npy")

        target_position = target_positions[episode]
        #print("target_position: ", target_position)

        

        # Define the pattern for file names you want to process
        file_pattern = "positions_episode_"+ str(episode)+"_iteration"
        #print(os.listdir(path + folder + "/" + subfolder))
        # List all .npy files in the folder that match the pattern
        npy_files = [
            f for f in os.listdir(path + folder + "/" + subfolder) 
            if f.startswith(file_pattern) and f.endswith('.npy')
        ]
        npy_files.sort()  # Optional: Sort the files
        num_files = len(npy_files)
        print(f"Current Episode {episode} Number of files: {num_files}")


        range_iterations = []
        if num_files < 10:
            range_iterations = list(range(0, num_files))
            npy_files = npy_files[:num_files]
        elif num_files < 20:
            range_iterations = list(range(0, num_files, 2))
            #make sure last iteration is included
            if num_files % 2 == 1:
                range_iterations.append(num_files-1)
            npy_files = [npy_files[i] for i in range_iterations]
        elif num_files < 40:
            range_iterations = list(range(0, num_files, 5))
            #make sure last iteration is included
            if num_files % 5 != 0:
                range_iterations.append(num_files-1)
            npy_files = [npy_files[i] for i in range_iterations]
        else:
            range_iterations = list(range(0, num_files, 20))
            #make sure last iteration is included
            if num_files % 20 != 0:
                range_iterations.append(num_files-1)
            npy_files = [npy_files[i] for i in range_iterations]
        #make sure 0 is in range_iterations
        if 0 not in range_iterations:
            range_iterations.append(0)
            range_iterations.sort()
            npy_files.insert(0, npy_files[0])
            npy_files.sort()
        
        print("range_iterations: ", range_iterations)
        print("npy_files: ", npy_files)

        distance_to_target_list = np.load(path + folder + "/" + subfolder + "/distance_to_target_episode_" + str(episode) +".npy")
        #print("distance_to_target_list: ", distance_to_target_list)


        hit = False
        video_num_for_episode = 0
        # Iterate through each matching .npy file
        for file_name in npy_files:
            file_path = os.path.join(path + folder + "/" + subfolder, file_name)
            print("Processing file:", file_name)
            
            # Load the .npy file
            positions = np.load(file_path)
            
            frames_y = []
            
            frames_new_pos = []
            simulation.set_color_geom("geom_target", [0, 0.4470, 0.7410, 1])

            shortest_distance = 1000
            for i in range(len(positions)):
                q_desired = positions[i]
                simulation.data.qpos = q_desired

                simulation.move_geom("geom_target", target_position)

                mujoco.mj_forward(simulation.model, simulation.data)
                current_distance = simulation.distance_to_target()
                shortest_distance = min(shortest_distance, current_distance)

                if current_distance > shortest_distance and current_distance < 0.5 and distance_to_target_list[range_iterations[video_num_for_episode]] < 0.03 and not hit:
                    print("Hit!, Position: ", simulation.get_tip_position())
                    hit = True
                    

                if hit:
                    simulation.set_color_geom("geom_target", [0, 1, 0, 1])

                text = f"Episode: {episode}, Iteration {range_iterations[video_num_for_episode]}, Distance: {distance_to_target_list[range_iterations[video_num_for_episode]]:.4f}"
                frame_y = simulation.render_image(camera="fixedy")
                frame_y = cv2.putText(frame_y, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                
                frame_new_pos = simulation.render_image(camera="new_pos")
                frame_new_pos = cv2.putText(frame_new_pos, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                frames_y.append(frame_y)
                frames_new_pos.append(frame_new_pos)

                #print("target_position: ",target_position)
            #video_name_x = path + folder + "/SAC_video_x_" + file_name.split("_")[-1].split(".")[0] + ".mp4"  
            video_name_y = path + folder + f"/SAC_video_y_episode{episode}" + file_name.split("_")[-1].split(".")[0] + ".mp4"  
            #video_name_z = path + folder + "/SAC_video_z_" + file_name.split("_")[-1].split(".")[0] + ".mp4"
            video_name_new_pos = path + folder + f"/SAC_video_new_pos_episode{episode}_" + file_name.split("_")[-1].split(".")[0] + ".mp4"
            

            #media.write_video(video_name_x, frames_x, fps=simulation.fps)
            media.write_video(video_name_y, frames_y, fps=simulation.fps)
            #media.write_video(video_name_z, frames_z, fps=simulation.fps)
            media.write_video(video_name_new_pos, frames_new_pos, fps=simulation.fps)
            video_num_for_episode+=1
            if hit:
                break
            print("Video saved as:", video_name_new_pos)
        
        print("Done!")



    
            
    


    




if __name__ == '__main__':
    main()

    