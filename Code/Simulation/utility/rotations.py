import numpy as np
import math




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