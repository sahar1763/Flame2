import numpy as np

def extract_subframes(image, subframe_res):
    subframe_height, subframe_width = subframe_res
    num_subframes_y = image.shape[0] // subframe_height
    num_subframes_x = image.shape[1] // subframe_width
    subframes = image.reshape(num_subframes_y, subframe_height,
                            num_subframes_x, subframe_width, 3)
    subframes = subframes.transpose(0, 2, 1, 3, 4)
    return subframes

def phase0(frame_generator, agg_frames, init_values):
    for i_frame, frame in enumerate(frame_generator):
        init_values = agg_frames(init_values, frame, i_frame)
        yield init_values

def phase1(frame_generator, init_values, map_to_cells):
    for frame in frame_generator:
        pass

def phase2(clf, frame_generator, subframe_res = (120,120)):
    for frame in frame_generator:
        subframes = extract_subframes(frame, subframe_res)
        subframes_cls = clf.predict(subframes)
        for subframe_i, subframe_cls in subframes_cls:
            if subframe_cls:
                print("!", subframe_i)
    