from utility.simulation import Simulation
from utility.replay_buffer import ReplayBuffer
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

    state_dim = simulation.s_dim
    action_dim = simulation.a_dim

    path = "./results/"
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    folders.sort()
    folder = folders[-2]
    print("folder: ", folder)

    state = np.load(path + folder + "/SAC_init_state.npy")


    ############# Cumulative rewards plot #############

    rb = ReplayBuffer(state_dim, action_dim)
    rb.load(path + folder + "/SAC_training_replay_buffer.npz")
    
    rp_rewards = rb.reward[1000:rb.size]
    #remove all rewards that are 0
    rp_rewards = rp_rewards[rp_rewards != 0]

    sum_rewards = []
    indexs_rewards = []
    current_sum = 0
    for i in range(1, len(rp_rewards)):
        current_sum += rp_rewards[i]
        if i % 200 == 0:
            indexs_rewards.append(i)
            sum_rewards.append(current_sum)
            current_sum = 0

    #load timesteps pr episode
    timesteps_per_episode = np.load(path + folder + "/SAC_episode_timesteps.npy", allow_pickle=True)
    cumul_sum = []
    for i in range(1, len(timesteps_per_episode)):
        cumul_sum.append(sum(timesteps_per_episode[:i]))

    # Plot cumulative rewards
    plt.figure(figsize=(12, 6))
    plt.plot(indexs_rewards, sum_rewards)
    plt.xlim(0, max(indexs_rewards))
    plt.xlabel("Iteration")
    plt.ylabel("Cumulative rewards per 200 iterations")
    plt.ticklabel_format(style='plain', axis='x')
    plt.grid(axis='y')

    # Create a secondary x-axis at the top
    ax2 = plt.gca().twiny()
    ax2.set_xlim(plt.gca().get_xlim())  # Match x-limits with the main x-axis
    ax2.grid(True)
    top_ticks_every = 10    # Show ticks at the top every Xth tick
    label_every = 5         # Show labels only for every Xth tick
    # Placements of the ticks and labels
    top_ticks = [cumul_sum[i] for i in range(0, len(cumul_sum), top_ticks_every)]
    top_labels = [
        str(i) if idx % label_every == 0 else "" 
        for idx, i in enumerate(range(0, len(top_ticks) * top_ticks_every, top_ticks_every))
    ]

    ax2.set_xticks(top_ticks)
    ax2.set_xticklabels(top_labels, rotation=90)
    ax2.set_xlabel("Episode")

    # Save the plot
    plt.savefig(path + folder + "/rewards_per_200_episodes.pdf")
    plt.close()
    
    #plot the timesteps per episode
    plt.plot(timesteps_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Iterations")
    plt.grid()
    plt.savefig(path + folder + "/timesteps_per_episode.pdf")

    #get the last 50 timesteps per episode
    timesteps_per_episode_last = timesteps_per_episode[-50:]
    print("Mean timesteps per episode last 50: ", np.mean(timesteps_per_episode_last))

    ############ End of cumulative rewards plot #############


    ############# Distance to target plot with simulations setup #############

    agent = SAC(state_dim=state_dim, action_dim=action_dim, discount=discount, tau=tau)

    distance_check_pos = [1.8, 0.0, 0.0]
    shortest_distance = 1000
    furthest_distance = 0

    # Create bunch of random target positions
    target_positions = []
    number_of_test_runs = 100
    for i in range(number_of_test_runs):
        target_position = simulation.get_random_position_within_box()
        target_position[0] = 1.8
        
        current_distance = np.linalg.norm(target_position[0:3] - distance_check_pos)
        if current_distance < shortest_distance:
            shortest_distance = current_distance
        if current_distance > furthest_distance:
            furthest_distance = current_distance

        target_positions.append(target_position)

    print("shortest_distance: ", shortest_distance)
    print("furthest_distance: ", furthest_distance)


    np.save(path + folder + "/target_positions", target_positions)

    files = os.listdir(path + folder)
    current_episode = 0

    if 'all_number_of_hits' not in locals():
        all_number_of_hits = []
    if 'all_distance_to_target' not in locals():
        all_distance_to_target = []
    if 'all_rewards' not in locals():
        all_rewards = []
    if 'all_actions' not in locals():
        all_actions = []
    
    #check if all number of runs have been done
    if len(all_number_of_hits) == 0:
        if os.path.exists(path + folder + "/number_of_hits_per_episode.npy"):
            all_number_of_hits = list(np.load(path + folder + "/number_of_hits_per_episode.npy"))
            all_distance_to_target = list(np.load(path + folder + "/distance_to_target_per_episode.npy"))
            all_rewards = list(np.load(path + folder + "/rewards_per_episode.npy"))
            all_actions = list(np.load(path + folder + "/actions_per_episode.npy"))

           
            # Output a plot with the number of hits for each episode and save to pdf
            plt.plot( all_number_of_hits)
            plt.xlabel("Number of episodes")
            plt.ylabel("Percentage of hits (%)")
            yticks = list(range(0, 21, 5))
            plt.yticks(yticks, labels=yticks)
            plt.grid()
            plt.savefig(path + folder + "/number_of_hits.pdf")
            plt.close()

            #boxplot of the distance to target for each episode
            plt.boxplot(
                all_distance_to_target,
                boxprops=dict(color="blue"),         # Color of the box
                medianprops=dict(color="red"),       # Color of the median line
                whiskerprops=dict(color="green"),    # Color of the whiskers
                capprops=dict(color="purple"),       # Color of the caps
                flierprops=dict(markerfacecolor="orange", marker="o")  # Color and marker for outliers
            )
            
            #make a vertical line at 0.03
            plt.axhline(y=0.03, color='r', linestyle='-')
            plt.xlabel("Number of episodes")
            plt.ylabel("Distance to target (m)")
            xticks = list(range(0, len(all_distance_to_target)+1, 50))
            plt.xticks(xticks, labels=xticks)
            plt.grid()
            plt.savefig(path + folder + "/distance_to_target.pdf")
            plt.close()

            # Output a plot with the rewards for each episode and save to pdf
            plt.boxplot(
                all_rewards,
                boxprops=dict(color="blue"),         # Color of the box
                medianprops=dict(color="red"),       # Color of the median line
                whiskerprops=dict(color="green"),    # Color of the whiskers
                capprops=dict(color="purple"),       # Color of the caps
                flierprops=dict(markerfacecolor="orange", marker="o")  # Color and marker for outliers
            )
            plt.xlabel("Number of episodes")
            plt.ylabel("Rewards")
            plt.savefig(path + folder + "/rewards.pdf")
            plt.close()


            #get the mean distance for each episode
            mean_distance = [np.mean(x) for x in all_distance_to_target]
            plt.plot(mean_distance)
            plt.xlabel("Number of episodes")
            plt.ylabel("Mean distance to target (m)")
            plt.grid()
            plt.savefig(path + folder + "/mean_distance_to_target.pdf")
            plt.close()
            print("Mean distance last:", mean_distance[-1])




    ########## Run the test runs with the newly generated targets ##########

    pbar = tqdm(total=500-len(all_number_of_hits), desc=f"Episode {current_episode} CurrentHits 0 CurrentTargetIndex 0", dynamic_ncols=True)
    while current_episode < 500:
        # Check if file exists
        if "SAC_agent_episode_" + str(current_episode) + "_actor" in files and "SAC_agent_episode_" + str(current_episode) + "_critic" in files:
            actor_path = path + folder + "/SAC_agent_episode_" + str(current_episode)
            agent.load(actor_path)

            
            

            #check if this episode has already been run
            if len(all_number_of_hits) > current_episode:
                #check if all number of runs have been done
                if len(all_distance_to_target[current_episode]) == number_of_test_runs:
                    current_episode += 1
                    continue
                else:
                    #remove the runs that have been done
                    all_number_of_hits = all_number_of_hits[:current_episode]
                    all_distance_to_target = all_distance_to_target[:current_episode]
                    all_rewards = all_rewards[:current_episode]
                    all_actions = all_actions[:current_episode]

            number_of_hits_episode = 0
            distance_to_target_episode = []
            rewards_episode = []
            actions_episode = []


            
            
            for i in range(number_of_test_runs):
                pbar.set_description("Episode " + str(current_episode) + " CurrentHits " + str(number_of_hits_episode) + " CurrentTargetIndex " + str(i))
                target_position = target_positions[i]

                init_state = np.copy(state)
                init_state[6:9] = target_position

                init_state_scaled = simulation.scale_observation(init_state)

                action = agent.select_action(init_state_scaled)

                action_unscaled = simulation.unscale_action(action)

                simulation.init_action(init_state, action_unscaled)
                simulation.move_geom("geom_target", target_position[0:3])

                distance_to_target, reward, almost_hit_position, done = simulation.run(action_unscaled, save_video=False, potential_target=target_position)

                distance_to_target_episode.append(distance_to_target)
                rewards_episode.append(reward)
                actions_episode.append(action_unscaled)
                actions_episode.append(action)


                if done:
                    number_of_hits_episode += 1
                    #pbar.set_description(f"Episode {current_episode} CurrentHits {number_of_hits_episode}")

                # Update the progress bar for each test run
                

            # Close the progress bar after the episode's test runs are done
            

            all_number_of_hits.append(number_of_hits_episode)
            all_distance_to_target.append(distance_to_target_episode)
            all_rewards.append(rewards_episode)
            all_actions.append(actions_episode)

            indexs = range(1, current_episode + 2)

            #check if number_of_hits_per_episode file exists
            
                

            np.save(path + folder + "/number_of_hits_per_episode", all_number_of_hits)
            np.save(path + folder + "/distance_to_target_per_episode", all_distance_to_target)
            np.save(path + folder + "/rewards_per_episode", all_rewards)
            np.save(path + folder + "/actions_per_episode", all_actions)


            last_number_of_hits = number_of_hits_episode



            current_episode += 1
            pbar.update(1)
        # else:
        #     #delete from memory while waiting
        #     if 'all_number_of_hits' in locals():
        #         del all_number_of_hits
        #         del all_distance_to_target
        #         del all_rewards
        #         del all_actions

        #     time.sleep(10)
        #     files = os.listdir(path + folder)
            


if __name__ == '__main__':
    main()

    