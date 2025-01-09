import mujoco

import numpy as np
from utility.rotations import *
import mediapy as media
import time
from utility.trajectory import MinJerkTrajectoryPlanner

class Simulation:
    def __init__(self, args):
        # The whole argument passed to the main python file. 
        self.args = args
        self.debug = False
        self.extra_run_time = 0.0

        # Controller, objective function and the objective values' array
        self.ctrl     = None
        self.obj      = None
        self.obj_arr  = None

        # Save the model name
        self.model_name = getattr(args, 'model_name', None)

        self.scene_folder = getattr(args, 'scene_folder', "../../Models/universal_robots_ur5e/")

        # Construct the basic mujoco attributes
        self.model  = mujoco.MjModel.from_xml_path( self.scene_folder + self.model_name + ".xml" )  
        self.data   = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, 1300, 1920)
        self.planner = MinJerkTrajectoryPlanner()
        mujoco.mj_forward(self.model, self.data)

        # Renderer settings. 
        self.fps = 60  
        self.frames = []
        self.use_all_cameras = False
        self.frames_x = []
        self.frames_y = []
        self.frames_z = []
        self.positions = []

        # The basic info of the model  actuator: 2
        self.actuator_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]
        # Get the joint names
        self.joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]
        # Get the geometry names
        self.geom_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(self.model.ngeom)]
        # Get the body names
        self.body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(self.model.nbody)]
        #get link names
        self.link_names = [name for name in self.body_names if "link" in name]
        #get whip node names
        self.whip_node_names = [name for name in self.joint_names if "whip" in name]
        #get end site names
        self.site_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(self.model.nsite)]


        # The basic info of the model actuator:
        self.n_act = len( self.actuator_names )

        # The basic info of the model joint:
        self.nq = len( self.joint_names )      

        # The basic info of the model geom:
        self.n_geom = len( self.geom_names )                



        #boundaries for init actionspace q1i, q2i, q3i, q4i, q5i, q6i, q1f, q2f, q3f, q4f, q5f, q6f and D
        self.min_finish_pos = np.array([-0.75 * np.pi, -0.3 * np.pi, -0.5 * np.pi, -np.pi, -np.pi, -np.pi])
        self.max_finish_pos = np.array([-0.25 * np.pi, 0.5 * np.pi,   0.5 * np.pi,  np.pi,  np.pi, np.pi])

        self.min_duration = 0.4
        self.max_duration = 1.5

        #action space is now the boundaries for the actionspace  q1f, q2f, q3f, q4f, q5f, q6f and D       
        self.action_space_low = np.concatenate((self.min_finish_pos, [self.min_duration]))
        self.action_space_high = np.concatenate((self.max_finish_pos, [self.max_duration]))

        # observation space

        self.min_init_pos = np.array([-0.75 * np.pi, 0.5 * np.pi, -0.5 * np.pi, -np.pi, -np.pi, -np.pi])
        self.max_init_pos = np.array([-0.25 * np.pi, 1.25 * np.pi, 0.5 * np.pi,  np.pi,  np.pi,  np.pi])
        
        self.box_position = self.get_box_position()
        self.box_size = self.get_box_size()
        self.min_target_obs =  np.array([-5.0, -5.0, -5.0])
        self.max_target_obs =  np.array([5.0, 5.0, 5.0])

        #state space is now the boundaries for the state space  q1i, q2i, q3i, q4i, q5i, q6i, box_x, box_y, box_z
        self.obs_low = np.concatenate((self.min_init_pos, self.min_target_obs))
        self.obs_high = np.concatenate((self.max_init_pos, self.max_target_obs))


        # action and state dim
        self.a_dim = len(self.action_space_low)
        self.s_dim = len(self.obs_low)

    def get_random_state(self):
        random_joint_state = np.random.uniform(self.min_init_pos, self.max_init_pos)
        random_position = self.get_random_position_within_box()
        return np.concatenate((random_joint_state, random_position))


    def init_action(self, state, action, new=True):
        if new:
            state = np.round(state, 4)
            action = np.round(action, 4)
            self.reset_simulation()

            # Extract init & final joint positions and duration
            init_joint_positions = state[:6]
            init_vel = [0, 0, 0, 0, 0, 0]
            final_joint_positions = action[:6]
            duration = action[6]

            self.planner.set_positions(init_joint_positions, final_joint_positions)  # Current to final joint positions
            self.planner.set_duration(duration)

            
            self.set_init_posture(init_joint_positions, init_vel)
            self.make_whip_downwards()

            mujoco.mj_forward(self.model, self.data)
        else:
            action = np.round(action, 4)
            
            self.reset_simulation()
            self.set_color_geom("geom_target", [0.0, 0.4470, 0.7410, 1.0])

            self.planner.set_positions(action[:6], action[6:12])
            self.planner.set_duration(action[-1])

            init_vel = [0,0,0,0,0,0]
            self.set_init_posture(action[0:6], init_vel)
            self.make_whip_downwards()

            mujoco.mj_forward(self.model, self.data)

    def reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)
        self.reset_frames()

    def scale_observation(self, obs):
        """Scale observation to range [-1, 1]."""
        return 2 * (obs - self.obs_low) / (self.obs_high - self.obs_low) - 1

    def unscale_observation(self, obs):
        """Unscale observation from range [-1, 1] back to original range."""
        return (obs + 1) / 2 * (self.obs_high - self.obs_low) + self.obs_low

    def scale_action(self, action):
        """Scale action to range [-1, 1]."""
        return 2 * (action - self.action_space_low) / (self.action_space_high - self.action_space_low) - 1

    def unscale_action(self, action):
        """Unscale action from range [-1, 1] back to original range."""
        return (action + 1) / 2 * (self.action_space_high - self.action_space_low) + self.action_space_low


    def stay_still(self, position):
        desired_position = position
        desired_velocity = np.zeros(6)

        
        position_error = desired_position - self.data.qpos[:6]
        velocity_error = desired_velocity - self.data.qvel[:6]
    
        K = np.diag([1000,1000,1000,1000,1000,1000])
        B = np.diag([100,100,100,100,100,100])

        torque = K @ position_error + B @ velocity_error 

        self.data.ctrl[:6] = torque


    def controller(self):
        qpos_original = self.data.qpos.copy()
        qvel_original = self.data.qvel.copy()
        qacc_original = self.data.qacc.copy()

        # Zero the acceleration for gravity compensation
        self.data.qacc[:] = 0
        mujoco.mj_inverse(self.model, self.data)  # Calculate the required torques for zero acceleration under gravity
        gravity_compensation = self.data.qfrc_inverse[:6].copy()  # Copy the compensation torques

        # Restore the original acceleration values
        self.data.qacc[:] = qacc_original

        desired_position = self.planner.get_desired_position(self.data.time)
        desired_velocity = self.planner.get_desired_velocity(self.data.time)

        # Calculate position and velocity errors
        position_error = desired_position - self.data.qpos[:6]
        velocity_error = desired_velocity - self.data.qvel[:6]

        # Adapt stiffness and damping matrices
        # Define parameters for adaptation
        beta = 0.9
        a = 0.001
        C = 50.0

        # Compute tracking error epsilon(t)
        epsilon = position_error + beta * velocity_error

        # Adapt impedance matrices K(t) and B(t)
        gamma = a / (1 + C * np.linalg.norm(epsilon)**2)
        K = np.diag(epsilon * position_error / gamma)
        B = np.diag(epsilon * velocity_error / gamma)


        # K = np.diag([0.14278282, 3.02316467, 0.21054788, -0.92810028, 66.44700533, 1.30332117])
        # B = np.diag([16.18231034, 217.11434533, 7.65075116, 65.78663436, 2374.91205629, 147.80398695])

        # Calculate control torque with gravity compensation
        torque = K @ position_error + B @ velocity_error + gravity_compensation[:6] 

        if self.debug:
            if int(self.data.time * 1000000) <= int(0.03 * 1000000):
                print("-------------------------------------------------")
                print("start_position: ", self.planner.q_i)
                print("end_position: ", self.planner.q_f)
                print("desired time: ", self.planner.D)
                print(f"torque: {self.data.ctrl[:6]}")#\ntime: {self.data.time}\nposition_error: {position_error}\nvelocity_error: {velocity_error}\nepsilon: {epsilon}\nK: {K}\nB: {B}\ngravity_compensation: {gravity_compensation[:6]}")
                print(f"current_position: {self.data.qpos[:6]}")
                print(f"desired_position: {desired_position}")
                print(f"pos_error: {position_error}")
                print(f"current_velocity: {self.data.qvel[:6]}")
                print(f"desired_velocity: {desired_velocity}")
                print(f"vel_error: {velocity_error}")
                print(f"K:{np.diag(K)}")
                print(f"B:{np.diag(B)}")


        self.data.ctrl[:6] = torque
        self.data.ctrl[6:] = 0

    
    def run(self, action, save_video=False, camera="fixed", potential_target=None):
        self.positions = []
        almost_hit_positions = []
        miss_position = []
        distance_to_box = 1000
        has_hit_target = False

        distance_to_target = 1000
        potential_alt_hit = []

        while self.data.time < action[-1] + self.extra_run_time:
            self.controller()
            

            mujoco.mj_step(self.model, self.data)
            
            current_distance_to_target = np.linalg.norm(potential_target - self.get_tip_position())
            if current_distance_to_target < distance_to_target:
                distance_to_target = current_distance_to_target
                potential_alt_hit = self.get_tip_position().copy()

            
            if current_distance_to_target < 0.03:
                #print("Hit target")
                #change color
                has_hit_target = True
                self.set_color_geom("geom_target", [0.0,1.0,0.0,1.0])
            elif not has_hit_target:
                self.set_color_geom("geom_target", [1.0,0.0,0.0,1.0])
            
            current_position = self.get_pos_if_box_not_hit()
            if current_position is not None:
                current_distance = self.get_distance_to_box(current_position)
                if current_distance < distance_to_box:
                    distance_to_box = current_distance
                    miss_position = current_position
            else:   
                almost_hit_position = self.get_pos_if_box_hit()
                if almost_hit_position is not None:
                    almost_hit_positions.append(almost_hit_position)

            

            if save_video:
                #self.add_frame_if_needed(camera=camera)
                self.add_positions_if_needed(self.data.qpos.copy())

        # if potential_target is not None:
        #     almost_hit_position = potential_target
        # else:
        #     almost_hit_position = miss_position

        if has_hit_target:
            almost_hit_position = []
            reward = 50
            done = True
        else:
            if len(almost_hit_positions) > 0:
                almost_hit_position = potential_alt_hit
                reward = -10 * distance_to_target
                done = False
            else:
                almost_hit_position = []
                reward = -30 * distance_to_target
                done = False
            


        return distance_to_target, reward, almost_hit_position, done
    


    def random_action(self):
        action = np.random.uniform(self.action_space_low, self.action_space_high)
        return action
    
    def ramdom_final_position_action(self, initial_position):
        action = np.random.uniform(self.action_space_low, self.action_space_high)
        action[0:6] = initial_position
        return action

    def render_image(self, camera="fixed"):
        self.renderer.update_scene(self.data, camera=camera, scene_option=dict())
        pixels = self.renderer.render()
        return pixels

    def write_image(self, image_name = 'output_frame.png', camera="fixed"):
        media.write_image(image_name, self.render_image(camera=camera))

    def reset_frames(self):
        self.frames = []
        self.frames_x = []
        self.frames_y = []
        self.frames_z = []

    def add_frame_if_needed(self, camera="fixed"):
        if len(self.frames) < self.data.time * self.fps:
            self.add_frame(camera=camera)

    def add_positions_if_needed(self, new_position):
        if len(self.positions) < self.data.time * self.fps:
            self.positions.append(new_position)
        
    
    def add_frame(self, camera="fixed"):
        self.frames.append(self.render_image(camera=camera))
        if self.use_all_cameras:
            self.frames_x.append(self.render_image(camera="fixedx"))
            self.frames_y.append(self.render_image(camera="fixedy"))
            self.frames_z.append(self.render_image(camera="fixedz"))

    def render_video(self, video_name = 'output_video.mp4'):
        if len(self.frames) == 0:
            print("No frames to render.")
            return
        media.write_video(video_name, self.frames, fps=self.fps)
        if self.use_all_cameras:
            video_name_x = video_name.replace(".mp4", "_x.mp4")
            media.write_video(video_name_x, self.frames_x, fps=self.fps)
            video_name_y = video_name.replace(".mp4", "_y.mp4")
            media.write_video(video_name_y, self.frames_y, fps=self.fps)
            video_name_z = video_name.replace(".mp4", "_z.mp4")
            media.write_video(video_name_z, self.frames_z, fps=self.fps)

    def distance_to_target(self):
        # Get the Euclidean distance between the tip of the whip and the target
        distance = np.linalg.norm( self.get_tip_position() - self.get_target_position())
        if distance <= 0.03:
            self.set_color_geom("geom_target", [0.0,1.0,0.0,1.0])
        elif distance <= 0.2:
            self.set_color_geom("geom_target", [0.0,0.0,1.0,1.0])
        else:
            self.set_color_geom("geom_target", [0.0,0.4470,0.7410,1.0])
        return distance

    def set_init_posture( self, qpos: np.ndarray, qvel: np.ndarray ):

        # Get the number of generalized coordinates
        nq = self.nq

        self.init_qpos = qpos
        self.init_qvel = qvel
        for i in range( len( qpos ) ):
            self.data.qpos[ i ] = qpos[ i ]
            self.data.qvel[ i ] = qvel[ i ]

        # Forward the simulation to update the posture 
        mujoco.mj_forward(self.model, self.data)

    def make_whip_downwards(self):
        current_time = self.data.time
        current_pos = self.data.qpos[:6].copy()
        current_dampening = self.model.dof_damping[:].copy()
        current_steptime = self.model.opt.timestep

        #change dampening of whip to be higher
        self.model.dof_damping[6:] = 10

        self.model.opt.timestep = 0.005

        #simulate for 4 sec to make whip fall down
        while self.data.time < 4.0:
            self.stay_still(current_pos)
            mujoco.mj_step(self.model, self.data)

        #restore the original dampening and time
        self.model.dof_damping[:] = current_dampening
        self.data.time = current_time
        
        self.model.opt.timestep = current_steptime
          

        


    def get_target_position(self):
        index_of_target = self.geom_names.index("geom_target")
        return self.data.geom_xpos[index_of_target]
    
    def get_tip_position(self):
        index_of_tip = self.site_names.index("site_whip_tip")
        return self.data.site_xpos[index_of_tip]

    def move_geom(self, geom_name, new_position):
        # Find the index of the geom
        geom_id = self.geom_names.index(geom_name)

        # Get the parent body ID of the geom
        body_id = self.model.geom_bodyid[geom_id]

        # Get the joint address of the body to update its position
        jnt_qposadr = self.model.jnt_qposadr[self.model.body_jntadr[body_id]]

        # Update the position of the body (first three values in qpos represent [x, y, z])
        self.data.qpos[jnt_qposadr: jnt_qposadr + 3] = new_position
        
        # Forward the state to propagate changes
        mujoco.mj_forward(self.model, self.data)

    def set_color_geom(self, geom_name, color):
        # Find the index of the geom
        geom_id = self.geom_names.index(geom_name)

        # Update the color of the geom
        self.model.geom_rgba[geom_id] = color

        # Forward the state to propagate changes
        mujoco.mj_forward(self.model, self.data)

    def get_box_position(self):
        index_of_box = self.geom_names.index("geom_box")
        return self.data.geom_xpos[index_of_box]
    
    def get_box_size(self):
        index_of_box = self.geom_names.index("geom_box")
        return self.model.geom_size[index_of_box]
    
    def get_random_position_within_box(self):
        box_position = self.get_box_position()
        box_size = self.get_box_size()
        random_position = np.random.uniform(-box_size/2, box_size/2) + box_position
        return random_position

    def get_if_hit_target(self):

        if self.distance_to_target() < 0.01:
            return True
        else:
            return False
    
    def get_distance_to_box(self, pos):
        box_position = self.get_box_position()
        box_size = self.get_box_size()

        box_min = box_position - box_size / 2
        box_max = box_position + box_size / 2

        #Find the shortest distance between the box and the position
        distance = 0
        for i in range(3):
            if pos[i] < box_min[i]:
                distance += (box_min[i] - pos[i])**2
            elif pos[i] > box_max[i]:
                distance += (pos[i] - box_max[i])**2
        return np.sqrt(distance)



    def get_pos_if_box_hit(self):
        index_of_box = self.geom_names.index("geom_box")
        index_of_whip = self.site_names.index("site_whip_tip")

        box_position = self.data.geom_xpos[index_of_box]
        box_size = self.model.geom_size[index_of_box]
        box_min = box_position - box_size / 2
        box_max = box_position + box_size / 2

        whip_tip_position = self.data.site_xpos[index_of_whip]
        


        if np.all(box_min <= whip_tip_position) and np.all(whip_tip_position <= box_max):
            return np.asarray(whip_tip_position)
        else:
            return None
        
    def get_pos_if_box_not_hit(self):
        index_of_box = self.geom_names.index("geom_box")
        index_of_whip = self.site_names.index("site_whip_tip")

        box_position = self.data.geom_xpos[index_of_box]
        box_size = self.model.geom_size[index_of_box]
        box_min = box_position - box_size / 2
        box_max = box_position + box_size / 2

        whip_tip_position = self.data.site_xpos[index_of_whip]
        


        if np.any(box_min > whip_tip_position) or np.any(whip_tip_position > box_max):
            return np.asarray(whip_tip_position)
        else:
            return None