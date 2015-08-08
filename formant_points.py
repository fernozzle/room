from __future__ import print_function
import math, csv

# For comparison only
def get_semitones(hertz):
    octaves = math.log(hertz) / math.log(2)
    return octaves * 12

def st_diff(freq1, freq2):
    return abs(get_semitones(freq1) - get_semitones(freq2))

def lerp(start, end, f):
    return start + f * (end - start)

def delerp(start, end, x):
    return float(x - start) / (end - start)

def rdp_formant_approx(formant_points, tolerance_st):
    first_point = formant_points[ 0]
    last_point  = formant_points[-1]
    
    max_st_dist = 0
    max_st_dist_index = 0
    for index in xrange(1, len(formant_points) - 1):
        current_point = formant_points[index]
        
        progress    = delerp(first_point[0], last_point[0], current_point[0])
        beeline_value = [lerp(start, end, progress) for start, end in zip(first_point[1], last_point[1])]

        current_st_dist = max([st_diff(smp_freq, bl_freq) for smp_freq, bl_freq in zip(current_point[1][1:], beeline_value[1:])])
        
        if current_st_dist > max_st_dist:
            max_st_dist_index = index
            max_st_dist = current_st_dist

    if max_st_dist > tolerance_st:
        left_result  = rdp_formant_approx(formant_points[:max_st_dist_index + 1], tolerance_st)
        right_result = rdp_formant_approx(formant_points[max_st_dist_index:], tolerance_st)
        result = left_result[:-1] + right_result
    else:
        result = [formant_points[0], formant_points[-1]]

    return result

# ==================================================

def halve(iterator):
    for i, line in enumerate(iterator):
        if not i % 2:
            yield line

def double(iterator):
    for line in iterator:
        yield line
        yield line

timestep = .01
'''
with open('formants classic.tsv', 'r') as formants_file, open('formants classic.tsv', 'r') as intensity_file, open('formant_points.txt', 'w') as formant_points_file:
    formants_file_rows  = csv.reader(formants_file,  delimiter ='\t')
    intensity_file_rows = csv.reader(intensity_file, delimiter ='\t')

    # Read list of frames (intensity, F1, F2, F3)
    formants_file_rows.next()
    intensity_file_rows.next()
    formant_data = [map(float, irow[0:1] + frow[1:4]) for irow, frow in zip(intensity_file_rows, formants_file_rows)]
    '''
'''
with open('formants.frm', 'r') as formants_file, open('room defric 2.pwr', 'r') as intensity_file, open('formant_points.txt', 'w') as formant_points_file:
    formants_file_rows  = csv.reader(formants_file, delimiter=' ')

    formant_data = [map(float, [intensity] + frow[0:3]) for intensity, frow in zip(intensity_file, formants_file_rows)]
'''
with open('formants.frm', 'r') as formants_file, open('formants classic.tsv', 'r') as intensity_file, open('formant_points.txt', 'w') as formant_points_file:
    formants_file_rows  = csv.reader(formants_file, delimiter=' ')
    intensity_file_rows = csv.reader(intensity_file, delimiter ='\t')

    intensity_file_rows.next()
    formant_data = [map(float, irow[0:1] + frow[0:3]) for irow, frow in zip(double(intensity_file_rows), formants_file_rows)]

    # Convert to list of [time, frame]
    formant_points = [[index * timestep, datum] for index, datum in enumerate(formant_data)]
    
    print('Original formant point count: %i' % len(formant_points))
    formant_points = rdp_formant_approx(formant_points, 4)
    print('Optimized formant point count: %i' % len(formant_points))

    # Write to file
    print('Phase step: ' + str(timestep))
    for formant_point in formant_points:
        print('FP(%.2f, %f, %.f., %.f., %.f.)' % (formant_point[0], formant_point[1][0], formant_point[1][1], formant_point[1][2], formant_point[1][3],), file=formant_points_file)


print('Done')
