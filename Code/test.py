import mujoco
import sys
import os
import cv2  # Import OpenCV for displaying images

class UR5e:
  
    def __init__(self, path):
        print('UR5e init.')
        self.m = mujoco.MjModel.from_xml_path(path)
        self.d = mujoco.MjData(self.m)
        self.opt = mujoco.MjvOption()
        mujoco.mj_forward(self.m, self.d)
    
    def __del__(self):
        print('UR5e del.')

if __name__ == '__main__':
    # Check if the path is correct
    if len(sys.argv) < 2:
        print('Please provide the path to the xml file.')
        print('Usage: python test.py <path_to_scene_xml_file>')
        sys.exit(1)
    elif not os.path.exists(sys.argv[1]):
        print('The file does not exist.')
        sys.exit(1)

    ur5e = UR5e(sys.argv[1])

    try:
        renderer = mujoco.Renderer(ur5e.m, 480, 640)
        if renderer is None:
            print("Failed to create renderer.")
            sys.exit(1)
        print("Renderer object created.")
    except Exception as e:
        print(f"Error initializing renderer: {e}")
        sys.exit(1)


    # Simulate for a number of steps and collect frames
    stepsize = ur5e.m.opt.timestep
    simstart = ur5e.d.time
    stoptime = simstart + 5
    currentStep = 0
    noOfStepsPrSec = 1/stepsize
    stepsFor60Fps = int(noOfStepsPrSec/60)

    frames = []

    while ur5e.d.time < stoptime:
        print(f"Time: {ur5e.d.time}")
            
        mujoco.mj_step(ur5e.m, ur5e.d)

        if renderer and currentStep % stepsFor60Fps == 0:
            renderer.update_scene(ur5e.d)
            image = renderer.render()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            frames.append(image)
        currentStep += 1

    # After the simulation loop, create a window with a trackbar to scroll through the frames
    cv2.namedWindow('Frame')

    # Use a dictionary to store state variables
    state = {'frame_index': 0, 'playing': False}

    # Define trackbar callback function
    def on_trackbar(val):
        state['frame_index'] = val
        cv2.imshow('Frame', frames[state['frame_index']])

    cv2.createTrackbar('Position', 'Frame', 0, len(frames)-1, on_trackbar)

    # Show the first frame
    cv2.imshow('Frame', frames[0])

    # Playback loop
    while True:
        if state['playing']:
            state['frame_index'] += 1
            if state['frame_index'] >= len(frames):
                state['frame_index'] = 0  # Loop back to start
            cv2.setTrackbarPos('Position', 'Frame', state['frame_index'])
            cv2.imshow('Frame', frames[state['frame_index']])
            # Wait for 16ms for approx 60fps playback
            key = cv2.waitKey(16) & 0xFF
        else:
            key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for a key press

        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar toggles play/pause
            state['playing'] = not state['playing']
        elif key == ord('n'):  # 'n' key advances to next frame
            state['playing'] = False
            state['frame_index'] = (state['frame_index'] + 1) % len(frames)
            cv2.setTrackbarPos('Position', 'Frame', state['frame_index'])
            cv2.imshow('Frame', frames[state['frame_index']])
        elif key == ord('p'):  # 'p' key goes back to previous frame
            state['playing'] = False
            state['frame_index'] = (state['frame_index'] - 1) % len(frames)
            cv2.setTrackbarPos('Position', 'Frame', state['frame_index'])
            cv2.imshow('Frame', frames[state['frame_index']])
        # Add more controls as needed

    # Clean up
    cv2.destroyAllWindows()

    if renderer:
        renderer.close()
