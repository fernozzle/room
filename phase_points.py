from __future__ import print_function
import math

# For comparison only
def get_semitones(hertz):
    octaves = math.log(hertz) / math.log(2)
    return octaves * 12

def lerp(start, end, f):
    return start + f * (end - start)

def delerp(start, end, x):
    return float(x - start) / (end - start)

def rdp_approx(points, tolerance_st):
    first_point = points[ 0]
    last_point  = points[-1]
    
    max_st_dist = 0
    max_st_dist_index = 0
    for index in xrange(1, len(points) - 1):
        current_point = points[index]
        
        progress    = delerp(first_point[0], last_point[0], current_point[0])
        beeline_value = lerp(first_point[1], last_point[1], progress)

        current_st_dist = abs(get_semitones(current_point[1]) - get_semitones(beeline_value))
        if current_st_dist > max_st_dist:
            max_st_dist_index = index
            max_st_dist = current_st_dist

    if max_st_dist > tolerance_st:
        left_result  = rdp_approx(points[:max_st_dist_index + 1], tolerance_st)
        right_result = rdp_approx(points[max_st_dist_index:], tolerance_st)
        result = left_result[:-1] + right_result
    else:
        result = [points[0], points[-1]]

    return result

# ==================================================

# Load list of pitches
pitches_fh = open('pitches.txt', 'r')
pitches = [float(x) for x in pitches_fh.readlines()]

# Convert to list of (time, pitch) points
timestep = .04
pitch_points = []
for i in xrange(0, len(pitches)):
    time  = i * timestep
    pitch = max(pitches[i], 1)
    pitch_points.append([time, pitch])

# Ramer-Douglas-Peucker optimization
print('Original pitch count: %i' % len(pitch_points))
pitch_points = rdp_approx(pitch_points, 0.3)
print('Optimized pitch count: %i' % len(pitch_points))

# Integrate to (time, phase) points
phase_points = []
current_time = 0
current_phase = 0
for i in xrange(0, len(pitch_points)):
    current_point = pitch_points[i]
    
    previous_time = current_time
    current_time = current_point[0]
    
    current_phase += current_point[1] * (current_time - previous_time)
    phase_points.append([current_time, current_phase])

# Write to file
every = 1
count = 0
print('Phase step: ' + str(timestep * every))
phase_points_fh = open('phase_points.txt', 'w')
for phase_point in phase_points:
    if count == 0:
        print('PP(%5.2f, %7.2f)' % (phase_point[0], phase_point[1]), file=phase_points_fh)
        count = every
    count -= 1
phase_points_fh.close()

print('Done')
