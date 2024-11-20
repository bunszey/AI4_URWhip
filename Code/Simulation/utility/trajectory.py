import numpy as np


def min_jerk_traj( t: float, ti: float, tf: float, pi: float, pf: float, D: float ):
    """
        Returning the 1D position and velocity data at time t of the minimum-jerk-trajectory ( current time )
        Time should start at t = 0
        Note that the minimum-jerk-trajectory remains at the initial (respectively, final) posture before (after) the movement.
        Arguments
        ---------
            [1] t : current time
            [2] ti: start of the movement
            [3] tf: end   of the movement
            [4] pi: initial ( reference ) posture
            [5] pf: final   ( reference ) posture
            [6]  D: duration
    """

    assert  t >=  0 and ti >= 0 and tf >= 0 and D >= 0
    assert tf >= ti

    if   t <= ti:
        pos = pi
        vel = 0

    elif ti < t <= tf:
        tau = ( t - ti ) / D                                                # Normalized time
        pos =     pi + ( pf - pi ) * ( 10. * tau ** 3 - 15. * tau ** 4 +  6. * tau ** 5 )
        vel = 1. / D * ( pf - pi ) * ( 30. * tau ** 2 - 60. * tau ** 3 + 30. * tau ** 4 )

    else:
        pos = pf
        vel = 0

    return pos, vel