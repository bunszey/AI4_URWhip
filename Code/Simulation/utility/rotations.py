import numpy as np
import math

from scipy.spatial.transform import Rotation as R

def skew_sym( w ):
    assert len( w ) == 3

    wtilde = np.zeros( ( 3, 3 ) )

    wtilde[ 0, 1 ] = -w[ 2 ]
    wtilde[ 0, 2 ] =  w[ 1 ]
    wtilde[ 2, 1 ] = -w[ 0 ]

    wtilde[ 1, 0 ] =  w[ 2 ]
    wtilde[ 2, 0 ] = -w[ 1 ]
    wtilde[ 1, 2 ] =  w[ 0 ]

    return wtilde

def quat2angx( q ):

    assert q[ 0 ] <= 1

    theta = 2 * np.arccos( q[ 0 ] )

    axis = q[ 1: ]

    # If the axis values are super small, then 
    tmp = np.sum( axis**2 )

    if tmp != 0:
        axis = axis/ np.sqrt( tmp )

    else:
        axis = np.array( [ 1., 0., 0. ] )
        theta = 0 

    return theta, axis


def rotx( q ):
    Rx = np.array( [ [ 1,            0,            0 ], 
                     [ 0,  np.cos( q ), -np.sin( q ) ],
                     [ 0,  np.sin( q ),  np.cos( q ) ]  ] )

    return Rx

def roty( q ):
    Ry = np.array( [ [  np.cos( q ),  0,  np.sin( q ) ], 
                     [            0,  1,            0 ],
                     [ -np.sin( q ),  0,  np.cos( q ) ]  ] )

    return Ry

def rotz( q ):
    Rz = np.array( [ [ np.cos( q ), -np.sin( q ), 0 ], 
                     [ np.sin( q ),  np.cos( q ), 0 ],
                     [           0,            0, 1 ]  ] )

    return Rz



def rot2quat( R: np.ndarray ):
    # [REF] https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
    # [REF] From Johannes

    assert len( R ) == 3 and len( R[ 0 ] ) == 3

    q = np.zeros( 4 )

    R00 = np.trace( R )
    tmp = np.array( [ R00, R[ 0,0 ], R[ 1,1 ], R[ 2,2 ] ] )
    k = np.argmax( tmp )

    q[ k ] = 0.5 * np.sqrt( 1 + 2 * tmp[ k ] - R00 )

    if k == 0:
        q[ 1 ] = 0.25/q[ k ] * ( R[ 2, 1 ] - R[ 1, 2 ] )
        q[ 2 ] = 0.25/q[ k ] * ( R[ 0, 2 ] - R[ 2, 0 ] )
        q[ 3 ] = 0.25/q[ k ] * ( R[ 1, 0 ] - R[ 0, 1 ] )

    elif k == 1:
        q[ 0 ] = 0.25/q[ k ] * ( R[ 2, 1 ] - R[ 1, 2 ] )
        q[ 2 ] = 0.25/q[ k ] * ( R[ 1, 0 ] + R[ 0, 1 ] )
        q[ 3 ] = 0.25/q[ k ] * ( R[ 0, 2 ] + R[ 2, 0 ] )

    elif k == 2:
        q[ 0 ] = 0.25/q[ k ] * ( R[ 0, 2 ] - R[ 2, 0 ] )
        q[ 2 ] = 0.25/q[ k ] * ( R[ 1, 0 ] + R[ 0, 1 ] )
        q[ 3 ] = 0.25/q[ k ] * ( R[ 2, 1 ] + R[ 1, 2 ] )

    elif k == 3:
        q[ 0 ] = 0.25/q[ k ] * ( R[ 1, 0 ] - R[ 0, 1 ] )
        q[ 1 ] = 0.25/q[ k ] * ( R[ 0, 2 ] + R[ 2, 0 ] )
        q[ 2 ] = 0.25/q[ k ] * ( R[ 2, 1 ] + R[ 1, 2 ] )

    if q[ 0 ] < 0 : q = -q

    return q




def quat2rot( quat: np.ndarray ):

    # [REF] https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html

    assert len( quat ) == 4

    q0, q1, q2 ,q3  = quat[:]    

    R = np.zeros( ( 3, 3 ) )

    R[ 0, 0 ] = q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2
    R[ 0, 1 ] = 2 * q1 * q2 - 2 * q0 * q3
    R[ 0, 2 ] = 2 * q1 * q3 + 2 * q0 * q2

    R[ 1, 0 ] = 2 * q1 * q2 + 2 * q0 * q3
    R[ 1, 1 ] = q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2
    R[ 1, 2 ] = 2 * q2 * q3 - 2 * q0 * q1

    R[ 2, 0 ] = 2 * q1 * q3 - 2 * q0 * q2
    R[ 2, 1 ] = 2 * q2 * q3 + 2 * q0 * q1
    R[ 2, 2 ] = q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2

    return R



def quat2euler( quat: np.ndarray ):                                         
    """
        Description
        -----------
        Converting a R4 quaternion vector (w, x, y, z) to Euler Angle (Roll, Pitch, Yaw)
        This code is directly from the following reference
        [REF] https://computergraphics.stackexchange.com/questions/8195/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr

        Arguments
        ---------
            [NAME]          [TYPE]        [DESCRIPTION]
            (1) quatVec     List          The quaternion vector, ordered in w, x, y and z

        Outputs
        --------
            [NAME]                   [TYPE]        [DESCRIPTION]
            (1) yaw, pitch, roll                   The euler angles of the given quaternion vector.
    """

    assert len( quat ) == 4

    w, x, y ,z  = quat[:]

    t0     =       + 2.0 * ( w * x + y * z )
    t1     = + 1.0 - 2.0 * ( x * x + y * y )
    roll   = math.atan2( t0, t1 )

    t2     = + 2.0 * ( w * y - z * x )
    t2     = + 1.0 if t2 > +1.0 else t2
    t2     = - 1.0 if t2 < -1.0 else t2
    pitch  = math.asin( t2 )

    t3     =       + 2.0 * ( w * z + x * y )
    t4     = + 1.0 - 2.0 * ( y * y + z * z )
    yaw    = math.atan2( t3, t4 )

    return yaw, pitch, roll


def rot2euler( mat: np.ndarray):
    """
        Description
        -----------
        Converting a 3x3 rotation matrix to Euler Angle (Roll, Pitch, Yaw)
        Arguments
        ---------
            [NAME]          [TYPE]        [DESCRIPTION]
            (1) mat         List          The rotation matrix 3x3 as flattened array

        Outputs
        --------
            [NAME]                   [TYPE]        [DESCRIPTION]
            (1) yaw, pitch, roll                   The euler angles of the given rotation matrix.
    """

    assert len( mat ) == 9

    m00, m01, m02 = mat[ 0:3 ]
    m10, m11, m12 = mat[ 3:6 ]
    m20, m21, m22 = mat[ 6:9 ]


    roll  = math.atan2( m21, m22 )
    pitch = -math.asin( m20 )
    yaw   = math.atan2( m10, m00 )

    return yaw, pitch, roll


def is_pointing_down(rotation_matrix):
    """
    Check if the local Z-axis of the rotation matrix is pointing downward.

    Arguments:
        rotation_matrix: np.ndarray
            A 3x3 rotation matrix.

    Returns:
        bool: True if the Z-axis is pointing downward, False otherwise.
    """
    # Extract the Z-axis (third column of the rotation matrix)
    z_axis = rotation_matrix[:, 2]  # Third column

    # Define the world downward direction
    downward = np.array([0, 0, -1])

    # Compute the dot product to check alignment
    dot_product = np.dot(z_axis, downward)

    # Threshold for determining if it's pointing downward
    return dot_product > 0.9  # Adjust threshold as needed


import numpy as np

def get_downward_pitch_and_roll(rotation_matrix):
    """
    Arguments:
        rotation_matrix: np.ndarray
            A 3x3 rotation matrix.

    Returns:
        float: pitch adjustment in radians
        float: roll adjustment in radians
    """

    # Current z-axis
    z_current = rotation_matrix[2, :]  # Third row of the rotation matrix (z-axis)

    # Desired z-axis
    z_desired = np.array([0, 0, -1])

    # Check if already aligned
    dot_product = np.dot(z_current, z_desired)
    if np.isclose(dot_product, -1):  # Perfectly aligned
        return 0.0, 0.0

    # Compute rotation axis and angle to align z-axis
    rotation_axis = np.cross(z_current, z_desired)

    # Normalize the rotation axis to avoid scaling issues
    if np.linalg.norm(rotation_axis) > 0:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    else:
        # If rotation_axis is zero, z_current and z_desired are parallel, no adjustment needed
        return 0.0, 0.0

    # Compute rotation angle
    rotation_angle = np.arccos(dot_product)

    # Convert to pitch and roll adjustments
    delta_pitch = rotation_axis[1] * rotation_angle  # Adjust pitch
    delta_roll = rotation_axis[0] * rotation_angle   # Adjust roll

    return delta_pitch, delta_roll

def get_downward_rotation(R):
    # Define the downward direction
    v_down = np.array([0, 0, -1])

    # Extract the current z-axis
    v_z = R[:, 2]

    # Compute the axis of rotation (cross product)
    v_axis = np.cross(v_z, v_down)
    v_axis = v_axis / np.linalg.norm(v_axis)  # Normalize the axis

    # Compute the angle of rotation (dot product)
    cos_theta = np.dot(v_z, v_down) / (np.linalg.norm(v_z) * np.linalg.norm(v_down))
    theta = np.arccos(cos_theta)

    # Rodrigues' rotation formula
    K = np.array([
        [0, -v_axis[2], v_axis[1]],
        [v_axis[2], 0, -v_axis[0]],
        [-v_axis[1], v_axis[0], 0]
    ])

    R_axis = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    # Compute the new rotation matrix
    R_new = np.dot(R_axis, R)

    return R_new, v_axis, theta

def compute_hinge_angles(rotation_axis, rotation_angle):
    # Normalize the rotation axis
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # Skew-symmetric matrix of the rotation axis
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    
    # Compute the rotation matrix using Rodrigues' formula
    R = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)
    
    # Extract hinge angles
    theta_x = np.arctan2(R[2, 1], R[2, 2])  # Rotation about x-axis
    theta_y = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))  # Rotation about y-axis
    
    return theta_x, theta_y

def compute_correction_quaternion(current_matrix, target_vector):
    """
    Compute the quaternion that aligns the z-axis of the current matrix to the target vector.
    """
    # Extract the current z-axis
    current_z = current_matrix[:, 2]

    # Compute the axis of rotation (cross product)
    axis = np.cross(current_z, target_vector)
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        # Already aligned
        return np.array([0, 0, 0, 1])  # Identity quaternion
    axis = axis / norm

    # Compute the angle of rotation (dot product)
    cos_theta = np.clip(np.dot(current_z, target_vector), -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # Create the quaternion
    quat = R.from_rotvec(axis * theta).as_quat()
    return quat


def decompose_rotation_matrix(R):
    """
    Decompose a rotation matrix into successive rotations about x and y axes.
    """
    # Extract angles
    theta_x = np.arctan2(R[2, 1], R[2, 2])  # Rotation about x-axis
    theta_y = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))  # Rotation about y-axis
    return theta_x, theta_y


def compute_residual_rotation(current_matrix, target_vector):
    """
    Compute the rotation matrix needed to align the z-axis of current_matrix to target_vector.
    """
    current_z = current_matrix[:, 2]
    v_cross = np.cross(current_z, target_vector)
    v_dot = np.dot(current_z, target_vector)
    norm = np.linalg.norm(v_cross)
    if norm < 1e-6:
        return np.eye(3)  # Already aligned
    axis = v_cross / norm
    angle = np.arccos(np.clip(v_dot, -1.0, 1.0))
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)