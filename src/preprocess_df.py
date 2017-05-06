# This file is required when training new model
# Use with Udacity provided dataset
import pandas as pd
import numpy as np

def remove_bias(df):
    """
    Remove dominantly zero steering input in sample data
    """
    minor_steering_df = df[df.steering.abs() <= 0.02].sample(frac=0.9)
    return df.drop(minor_steering_df.index)

def aug_l_c_r_df(df, correction = 0.25):
    """
    shift: column is used for image affine transformation
    A correction factor is used to compensate perspective difference among
    left, center and right cameras
    """
    new_data = []
    for it, row in df.iterrows():
        steering = row.steering
        new_data.append({
            'image': row.left,
            'steering': np.clip(steering + correction, -1, 1),
            'is_flipped': False
        })

        new_data.append({
            'image': row.center,
            'steering': steering,
            'is_flipped': False
        })

        new_data.append({
            'image': row.right,
            'steering': np.clip(steering - correction, -1, 1),
            'is_flipped': False
        })

    return pd.DataFrame(new_data, columns=('image', 'steering', 'is_flipped'))

def aug_flip(df):
    flip_df = df[df.steering.abs() > 0.07].sample(frac=0.5)
#     flip_df = df[df.steering != 0].sample(frac=0.4)
    flip_df.loc[:, 'is_flipped'] = True
    flip_df.loc[:, 'steering'] = -flip_df.loc[:, 'steering']
#     return flip_df
    return pd.concat([df, flip_df])

def get_random():
#     (b - a) * np.random.random_sample() + a
    return (-1 - 1) * np.random.random_sample() + 1

def shift_to_steer_input(shift, max_steer):
    """
    steer -1, 1
    """
    return shift * max_steer

MAX_SHIFT = 0.3

def shift_image(df):
    df.loc[:, 'tx'] = 0
    new_df = df[df.steering != 0].copy()
    new_df.loc[:, 'tx'] = 0.0


    def update_row_steering(row):
        row.tx = get_random()
        row.steering = row.steering + row.tx * shift_to_steer_input(row.tx, MAX_SHIFT)
        row.steering = np.clip(row.steering, -1, 1)
        return row

    new_df = new_df.apply(update_row_steering, axis=1)
    return pd.concat([df, new_df])

def process_driver_log(df):
    _df = remove_bias(df)
    _df = aug_l_c_r_df(_df)
    _df = aug_flip(_df)
    _df = shift_image(_df)
    return _df

drive_log_df = pd.read_csv('../data/driving_log.csv')
processed_driver_log = process_driver_log(drive_log_df)
