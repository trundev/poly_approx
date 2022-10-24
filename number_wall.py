"""Number wall implementation

Inspired by Mathologer video https://www.youtube.com/watch?v=NO1_-qptr6c
"""
import numpy as np

WALL_DEPTH = 8

EPSILON = 1e-14

TRANGE_START = 1
TRANGE_STEP = 1
TRANGE_NUM = 10

# Simple integer numbers
EXP_BASES = np.asarray([2, 3, 4])

# Non-attenuating conjugated oscillations (result is real-only)
#EXP_BASES = np.exp([1j*np.pi/32, -1j*np.pi/32])
#EXP_BASES *= .9     # Make it attenuating

# Non-attenuating oscillations
#EXP_BASES = np.asarray([1j*np.pi/2, 1j*np.pi/3])

# Extra simplification
#EXP_BASES = EXP_BASES[:2]

#
# Number wall crosses:
#   up ->        [0, 1]
# left -> [1, 0] [1, 1] [1, 2] <- right
# down ->        [2, 1]
def wall_cross_down(nwall: np.array) -> None:
    """Calculate down-cross number-wall values (inplace)"""
    assert nwall.shape[0] > 2, 'Need at-least 3 rows'
    # down = (middle**2 - left*right) / top
    nwall[2, 1:-1] = (nwall[1, 1:-1] ** 2 - nwall[1, :-2] * nwall[1, 2:]) / nwall[0, 1:-1]

def wall_cross_up(nwall: np.array) -> None:
    """Calculate up-cross number-wall values (inplace)"""
    # "Down" on mirrored first 3 columns
    wall_cross_down(nwall[2::-1])

def wall_cross_right(nwall: np.array) -> None:
    """Calculate right-cross number-wall valuess (inplace)"""
    # right = (middle**2 - top*down) / left
    nwall[1:-1, 2] = (nwall[1:-1, 1] ** 2 - nwall[:-2, 1] * nwall[2:, 1]) / nwall[1:-1, 0]

def wall_cross_left(nwall: np.array) -> None:
    """Calculate left-cross number-wall values (inplace)"""
    # "Right" on mirrored first 3 rows
    wall_cross_right(nwall[:, 2::-1])

def generate_wall(vals: np.array):
    nwall = np.full((WALL_DEPTH,) + vals.shape, np.nan, dtype=complex if np.iscomplexobj(vals) else float)
    nwall[0] = 1
    nwall[1] = vals
    for row in range(nwall.shape[0] - 2):
        with np.errstate(divide='raise', over='raise', under='raise', invalid='raise'):
            try:
                wall_cross_down(nwall[row:])
            except (FloatingPointError, OverflowError) as ex:
                print(f'{row}:', ex)
    return nwall

def do_test(exp_bases: np.array):
    trange = np.arange(TRANGE_NUM) * TRANGE_STEP + TRANGE_START
    vals = np.power.outer(exp_bases, trange)
    vals = vals.sum(0)              #... or mean(0)
    vals = np.real_if_close(vals)   #CHECKME:
    print(f'exp_bases {np.round(exp_bases, 2)}, angles {np.round(np.angle(exp_bases, deg=True), 1)}')

    nwall = generate_wall(vals)

    np.seterr(divide='ignore', invalid='ignore')
    last_nozero = np.nanmax(np.abs(nwall), axis=-1) > EPSILON
    last_nozero = np.nonzero(last_nozero)[0][-1]
    print(f'Last non-zero row index {last_nozero}, actual depth {last_nozero+2}:')
    print(' ', ', '.join(f'{v:.3f}' for v in nwall[last_nozero]))
    step = np.nanmean(nwall[last_nozero, 1:] / nwall[last_nozero, :-1])
    print(f'Multiply step: {step:.2f}, angle {np.round(np.angle(step, deg=True), 1)}')

    # Cut after the first all-zero row (allow validation)
    nwall = nwall[:last_nozero+2]

    # Validate right-cross
    nwall_view = nwall[..., :-nwall.shape[0]+2]
    test_wall = nwall_view.copy()
    test_wall[1:-1, -1] = np.nan    # Clean last (right) column
    wall_cross_right(test_wall[:, -3:])
    assert ((test_wall == nwall_view) | np.isnan(nwall_view)).all(), 'cross-right failed'

    # Validate left-cross
    nwall_view = nwall[..., nwall.shape[0]-2:]
    test_wall = nwall_view.copy()
    test_wall[1:-1, 0] = np.nan    # Clean first (left) column
    wall_cross_left(test_wall)
    assert ((test_wall == nwall_view) | np.isnan(nwall_view)).all(), 'cross-left failed'


if __name__ == '__main__':
    do_test(EXP_BASES)
