from RL.SAC import *
from utility.simulation import Simulation
import numpy as np
import torch
import random
import math
from utility.replay_buffer import *
import os
import tqdm
import mujoco
import time



def main():
    np.set_printoptions(suppress=True, linewidth=100000)

    args = lambda: None
    args.model_name = "scene"
    args.scene_folder = "../../Models/universal_robots_ur5e/"

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    seed = 10
    discount = 0.99 # Discount factor
    tau = 0.005     # Target network update rate
    

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    simulation = Simulation(args)
    

    state_dim = simulation.s_dim
    action_dim = simulation.a_dim

    simple_path = "./results/"

    #timestamp as YYYY-MM-DD_HH:MM:SS
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
    path = simple_path + timestamp + "/"
    file_name = "SAC"


    number_of_random_exploration = 1000
    number_of_runs = 100000
    save_interval = 200
    save_video_interval = 100000
    simulation.extra_run_time = 1.0
    simulation.fps = 120
    simulation.use_all_cameras = True


    os.makedirs(path, exist_ok=True)

    target_position = [1.60304854, 0.07446196, 0.01489245]
    simulation.move_geom("geom_target", target_position[0:3])
    
    state = simulation.get_random_state()
    state[0:6] = [-0.5 * np.pi, 1.1 * np.pi, 0.0, -1.57079632679, 1.57079632679, 0]
    state[6:] = target_position
    

    action = simulation.random_action()
    simulation.init_action(state, action)

    simulation.write_image(image_name=simple_path + "start_pos_train_x.png",camera="fixedx")
    simulation.write_image(image_name=simple_path + "start_pos_train_y.png",camera="fixedy")
    simulation.write_image(image_name=simple_path + "start_pos_train_z.png",camera="fixedz")

    agent = SAC(state_dim=state_dim, action_dim=action_dim, discount=discount,tau=tau)
    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim)

    episode_reward = 0         # Cumulative reward obtained.
    episode_random_reward = 0  # Cumulative reward obtained during the random exploration phase.
    episode_timesteps = 0      # Number of timesteps in the current episode.
    episode_num = 0            # Episode counter    

    rewards_pr_200 = []        # Stores rewards obtained in each interval of 200 timesteps.
    rewards_pr_200_sum = 0     # Cumulative reward obtained in the current interval of 200 timesteps.

    evaluation_random = []     # Stores rewards obtained during the initial phase of training, where actions are chosen randomly.
    evaluation_train = []      # Stores rewards during the policy-driven training phase, storing the total reward obtained in each interval or episode.
    evaluation_hit = []        # Stores the number of timesteps required to hit the target during training or evaluation.

    hit_target = 0             # Counter for the number of times the target is hit.
    closest_distance = 1000    # Closest distance to the target.

   


    simulation.reset_simulation()

    # Create bunch of random target positions
    target_positions = []
    number_of_random_targets = math.ceil(number_of_runs / save_interval)
    for i in range(number_of_random_targets):
        target_position = simulation.get_random_position_within_box()
        target_position[0] = 1.8

        target_positions.append(target_position)
    
    np.save(path + file_name + "_target_positions", target_positions)

    #save the initial state
    init_state = np.copy(state)
    init_state[6:] = target_positions[0]
    np.save(path + file_name + "_init_state", init_state)

    #check if the replay buffer exists and load it
    skip_random = False
    if os.path.exists(simple_path + file_name + "_random_replay_buffer.npz"):
        replay_buffer.load(simple_path + file_name + "_random_replay_buffer.npz")
        skip_random = True
        print("Replay buffer loaded")

    
    progress = tqdm.tqdm(range(1, number_of_runs + 1))
    progress.set_description("Reward: {float(reward):.2f}")
    distance_to_target_in_episode = []

    for e in progress:
        
        if e <= number_of_random_exploration: # Random exploration phase
            if skip_random:
                continue
            
            state = init_state
            
            state_scaled = simulation.scale_observation(state)

            action = simulation.random_action()
            

            simulation.init_action(state, action)
            simulation.move_geom("geom_target", state[6:9])
            # print("Moving target to: ", state[6:9], "State is: ", state)

            distance_to_target, reward, almost_hit_position, done = simulation.run(action, save_video=(e % save_video_interval == 0), potential_target=state[6:9], camera="new_pos")
            
            #update the state with the target
            hit_state = np.copy(state)
            if len(almost_hit_position) > 0:
                hit_state[6:] = almost_hit_position
                reward = 50
            scaled_hit_state = simulation.scale_observation(hit_state)

            episode_random_reward += reward
            simulation.planner.debug = False

            if e%save_video_interval == 0:
                #save the distance to the target
                simulation.render_video(path + file_name + "_random_" + str(e) + ".mp4")

            if e % save_interval == 0:
                evaluation_random.append(episode_random_reward)
                np.save(path + file_name + "_random", evaluation_random)
                episode_random_reward = 0

            # print("---------------------------------------")
            # print("action: ", e, action)
            # print("state: ", state, "reward: ", reward, "done: ", done)
            # print("target: ", target)
            # print("---------------------------------------")
            progress.set_description(f"Reward: {float(reward):.2f}")

            next_state = simulation.scale_observation(state)
            action = simulation.scale_action(action)

            done_bool = float(done)
            replay_buffer.add(scaled_hit_state, action, next_state, reward, done_bool)
            init_state = state
            init_state[6:] = target_positions[episode_num]

            if e == number_of_random_exploration:
                print("evaluation_random:", evaluation_random)
                np.save(path + file_name + "_random", np.array(evaluation_random, dtype=object))
                replay_buffer.save(simple_path + file_name + "_random_replay_buffer")
                episode_random_reward = 0

        else: # Train the agent
            state = init_state
            state[6:] = target_positions[episode_num]

            state_scaled = simulation.scale_observation(state)

            action_scaled = agent.select_action(state_scaled)

            action = simulation.unscale_action(action_scaled)
            
            #check limits
            if np.any(action > simulation.action_space_high) or np.any(action < simulation.action_space_low) or np.any(action_scaled > 1) or np.any(action_scaled < -1):
                print("---------------------------------------")
                print("Action out of bounds: ", action)
                print("Action scaled: ", action_scaled)



            simulation.init_action(state, action)
            simulation.move_geom("geom_target", state[6:9])
            # print("Moving target to: ", state[6:9])

            #very first time
            if e == number_of_random_exploration + 1:
                simulation.write_image(image_name=path + file_name + "start_pos_train_x.png",camera="fixedx")
                simulation.write_image(image_name=path + file_name + "start_pos_train_y.png",camera="fixedy")
                simulation.write_image(image_name=path + file_name + "start_pos_train_z.png",camera="fixedz")

            save_video = True

            distance_to_target, reward, almost_hit_position, done = simulation.run(action, save_video=save_video, potential_target=state[6:9], camera="new_pos")


            if save_video:

                #print("Saving distance to target in episode " + str(episode_num) + " iteration " + str(e))
                distance_to_target_in_episode.append(distance_to_target)
                #make sure directory exists
                os.makedirs(path + file_name + "_video_data", exist_ok=True)
                np.save(path + file_name + "_video_data/distance_to_target_episode_" + str(episode_num), distance_to_target_in_episode)
                np.save(path + file_name + "_video_data/positions_episode_" + str(episode_num) + "_iteration" + str(e), simulation.positions)

                    
                    
                #print("Saving video")
                #simulation.render_video(path + file_name + "_training_" + str(episode_num) + "_" + str(e) + ".mp4")

            
            if distance_to_target < closest_distance:
                closest_distance = distance_to_target

            #update the state with the target
            hit_state = np.copy(state)
            # if len(almost_hit_position) > 0 and distance_to_target < 0.15 and episode_reward < 50*save_interval//4:
            #     hit_state[6:] = almost_hit_position
            #     reward = 50
            scaled_hit_state = simulation.scale_observation(hit_state)
                
            next_state = simulation.scale_observation(state)

            done_bool = float(done)

            replay_buffer.add(scaled_hit_state, action_scaled, next_state, reward, done_bool)
            

            next_state = simulation.unscale_observation(next_state)
            init_state = np.copy(next_state)

            episode_reward += reward
            episode_timesteps += 1

            agent.train(replay_buffer, batch_size=number_of_random_exploration)

            rewards_pr_200_sum += reward

            if episode_timesteps % 200 == 0:
                rewards_pr_200.append(rewards_pr_200_sum)
                rewards_pr_200_sum = 0
                np.save(path + file_name + "_rewards_pr_200", np.array(rewards_pr_200, dtype=object))


            # print("---------------------------------------")
            # print("SAC e:", e, "action: ", action, "Sac reward: ", reward, "done: ", done)
            # print("target: ", target)
            # print("---------------------------------------")
            progress.set_description(f"Reward: {float(reward):.2f}, Distance: {float(distance_to_target):.3f}")

            if episode_timesteps >= save_interval:
                print("Current episode:", episode_num, "Total sim#:", e, "Episode steps:", episode_timesteps, "Reward:", episode_reward)
                print("Init position:", state[:6], "Target not hit:", state[6:9], "Closest distance:", closest_distance)


                agent.save(path + file_name + "_agent_episode_" + str(episode_num))
                evaluation_hit.append(episode_timesteps)
                evaluation_train.append(episode_reward)
                np.save(path + file_name + "_episode_timesteps", np.array(evaluation_hit, dtype=object))
                np.save(path + file_name, np.array(evaluation_train, dtype=object))



                episode_num += 1
                hit_target += 1
                new_target = simulation.get_random_position_within_box()
                init_state[6:] = new_target
                episode_timesteps = 0
                closest_distance = 1000
                
                
                replay_buffer.save(path + file_name + "_training_replay_buffer")
                episode_reward = 0
                distance_to_target_in_episode = []


                




            if done:
                print("Current episode:", episode_num, "Total sim#:", e, "Episode steps:", episode_timesteps, "Reward:", episode_reward)
                print("Init position:", state[:6], "Target hit:", state[6:9], "Closest distance:", closest_distance)

                agent.save(path + file_name + "_agent_episode_" + str(episode_num))
                evaluation_hit.append(episode_timesteps)
                evaluation_train.append(episode_reward)
                np.save(path + file_name + "_episode_timesteps", np.array(evaluation_hit, dtype=object))
                np.save(path + file_name, np.array(evaluation_train, dtype=object))

                episode_num += 1
                hit_target += 1
                new_target = simulation.get_random_position_within_box()
                init_state[6:] = new_target
                episode_timesteps = 0
                closest_distance = 1000

                replay_buffer.save(path + file_name + "_training_replay_buffer")
                episode_reward = 0
                distance_to_target_in_episode = []

                
                


                

            

    print("data from the sac npy")
    print(np.load("results/SAC.npy"))
if __name__ == "__main__":
    main()

