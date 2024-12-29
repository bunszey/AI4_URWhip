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

        self.extra_run_time = 0.0

        # Controller, objective function and the objective values' array
        self.ctrl     = None
        self.obj      = None
        self.obj_arr  = None

        # Save the model name
        self.model_name = args.model_name
       
        # Construct the basic mujoco attributes
        self.model  = mujoco.MjModel.from_xml_path( '../../Models/universal_robots_ur5e/' + self.model_name + ".xml" )  
        self.data   = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, 480, 640)
        self.planner = MinJerkTrajectoryPlanner()

        # Renderer settings. 
        self.fps = 240  
        self.frames = []


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

        # The current time, time-step (step_t), start time of the controller (ts) and total runtime (T) of the simulation
        self.t   = 0
        self.step_t  = self.model.opt.timestep                                                  
        self.stop_t= 0.15


        #boundaries for init actionspace q1i, q2i, q3i, q4i, q5i, q6i, q1f, q2f, q3f, q4f, q5f, q6f and D
        self.action_space_low = np.array( [   -0.75 * np.pi, 0.5 * np.pi, -0.5 * np.pi, -0.75 * np.pi, -np.pi, -np.pi,      -0.75 * np.pi, -0.3 * np.pi, -0.5 * np.pi, -0.75 * np.pi, -np.pi, -np.pi, 0.4 ] ) 
        self.action_space_high = np.array( [  -0.25 * np.pi, 1.25 * np.pi, 0.5 * np.pi,  0.75 * np.pi,  np.pi,  np.pi,      -0.25 * np.pi, 0.5 * np.pi,   0.5 * np.pi,  0.75 * np.pi,  np.pi, np.pi,  1.5 ] )
                                          
                                           
        # observation space
        self.obs_low = np.array([0.01])
        self.obs_high = np.array([10.0])

        # action and state dim
        self.a_dim = len(self.action_space_low)
        self.s_dim = len(self.obs_low)

        # set valuesets
        self.q1i_min = self.action_space_low[0]
        self.q2i_min = self.action_space_low[1]
        self.q3i_min = self.action_space_low[2]
        self.q4i_min = self.action_space_low[3]
        self.q5i_min = self.action_space_low[4]
        self.q6i_min = self.action_space_low[5]
        self.q1f_min = self.action_space_low[6]
        self.q2f_min = self.action_space_low[7]
        self.q3f_min = self.action_space_low[8]
        self.q4f_min = self.action_space_low[9]
        self.q5f_min = self.action_space_low[10]
        self.q6f_min = self.action_space_low[11]
        self.t_min = self.action_space_low[12]

        self.q1i_max = self.action_space_high[0]
        self.q2i_max = self.action_space_high[1]
        self.q3i_max = self.action_space_high[2]
        self.q4i_max = self.action_space_high[3]
        self.q5i_max = self.action_space_high[4]
        self.q6i_max = self.action_space_high[5]
        self.q1f_max = self.action_space_high[6]
        self.q2f_max = self.action_space_high[7]
        self.q3f_max = self.action_space_high[8]
        self.q4f_max = self.action_space_high[9]
        self.q5f_max = self.action_space_high[10]
        self.q6f_max = self.action_space_high[11]
        self.t_max = self.action_space_high[12]


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
        a = 0.01
        C = 20.0

        # Compute tracking error epsilon(t)
        epsilon = position_error + beta * velocity_error

        # Adapt impedance matrices K(t) and B(t)
        gamma = a / (1 + C * np.linalg.norm(epsilon)**2)
        K = np.diag(epsilon * position_error / gamma)
        B = np.diag(epsilon * velocity_error / gamma)


        # Calculate control torque with gravity compensation
        torque = K @ position_error + B @ velocity_error + gravity_compensation[:6] 
        self.data.ctrl[:6] = torque

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

    def add_frame_if_needed(self, camera="fixed"):
        if len(self.frames) < self.data.time * self.fps:
            self.add_frame(camera=camera)

    def add_frame(self, camera="fixed"):
        self.frames.append(self.render_image(camera=camera))

    def render_video(self, video_name = 'output_video.mp4'):
        media.write_video(video_name, self.frames, fps=self.fps)

    def distance_to_target(self):
        # Get the Euclidean distance between the tip of the whip and the target
        distance = np.linalg.norm( self.get_tip_position() - self.get_target_position())
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
        # Get the rotation of the end-effector
        index_of_endeffector = self.site_names.index("whip_dir")
        rotation_matrix = self.data.site_xmat[index_of_endeffector].reshape(3, 3)
        # print("Initial Rotation matrix: \n", rotation_matrix)

        # Target downward vector
        target_vector = np.array([0, 0, -1])

        # Iterative alignment
        max_iterations = 100
        tolerance = 1e-3
        best_alignment_error = np.inf
        best_theta_x = 0
        best_theta_y = 0

        for iteration in range(max_iterations):
            # Compute the residual rotation matrix
            residual_rotation = compute_residual_rotation(rotation_matrix, target_vector)

            # Decompose the residual rotation into joint-specific angles
            theta_x = np.arctan2(residual_rotation[2, 1], residual_rotation[2, 2])  # x-axis
            theta_y = np.arctan2(-residual_rotation[2, 0], np.sqrt(residual_rotation[2, 1]**2 + residual_rotation[2, 2]**2))  # y-axis

            # print(f"Iteration {iteration + 1}")
            # print("Residual Rotation matrix: \n", residual_rotation)
            # print("Theta_x (degrees): ", np.degrees(theta_x))
            # print("Theta_y (degrees): ", np.degrees(theta_y))

            # Apply the corrections to the joints
            index_of_whipx = self.joint_names.index("joint_whip_node1_X")
            index_of_whipy = self.joint_names.index("joint_whip_node1_Y")
            self.data.qpos[index_of_whipx] += theta_y
            self.data.qpos[index_of_whipy] += theta_x
            mujoco.mj_forward(self.model, self.data)

            # Check the updated rotation matrix
            rotation_matrix = self.data.site_xmat[index_of_endeffector].reshape(3, 3)
            # print("Updated Rotation matrix: \n", rotation_matrix)

            # Check alignment
            final_z_axis = rotation_matrix[:, 2]
            alignment_error = np.linalg.norm(final_z_axis - target_vector)
            # print("Final Z-axis direction: ", final_z_axis)
            # print("Alignment Error: ", alignment_error)
            
            # Update the best alignment error
            if alignment_error < best_alignment_error:
                best_alignment_error = alignment_error
                best_theta_x = self.data.qpos[index_of_whipx]
                best_theta_y = self.data.qpos[index_of_whipy]

            if alignment_error < tolerance:
                # print("Alignment achieved.")
                break
        else:
            # print("Alignment not achieved within the maximum iterations.")
            # print("Best Alignment Error: ", best_alignment_error)
            self.data.qpos[index_of_whipy] = best_theta_y
            self.data.qpos[index_of_whipx] = best_theta_x
            mujoco.mj_forward(self.model, self.data)
        


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