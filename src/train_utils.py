import matplotlib.image as mpimg
from image_utils import *
from config import W, H

def load_image_by_row(row):
    img = mpimg.imread("../data/{0}".format(row.image.strip()))
    if row.is_flipped:
        img = flip_image(img)
    if row.tx != 0:
        img = translate_image(img, row.tx)
    return crop_image(color_select(img))

def load_features_and_labels(df):
    """
    Input should be pandas dataFrame
    with colums: ['image', 'steering', 'is_flipped', 'tx']
    Okay, everything sits in memory now
    """
    images = [load_image_by_row(row) for _, row in df.iterrows()]
    return np.array(images).reshape((len(images), H, W, 1)), df.steering

import json
def save_model(mdl, model_name, base_path='./', verbose=True):
    """
    e.g.: ./model.json or ./model.h5
    """
    model_json = mdl.to_json()
    model_path = base_path + model_name 
    with open(str(model_path) + '.json', 'w') as outfile:
        json.dump(model_json, outfile)

    mdl.save(str(model_path) + '.h5')
    if verbose:
        print(str(model_path) + " saved" )


