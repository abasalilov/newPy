
import innvestigate.utils as iutils
import innvestigate.applications.imagenet
import innvestigate
import urllib.request
import keras.models
import keras.backend
import keras
import matplotlib.pyplot as plt
import os
import numpy as np
import imp
import warnings
# Importing Image module from PIL package
from PIL import Image
import PIL
import imp
import threading
from functools import wraps
import urllib

# Use utility libraries to focus on relevant iNNvestigate routines.
eutils = imp.load_source(
    "utils", "/Users/aleks/Desktop/qv/newPy/utils/utils.py")
imgnetutils = imp.load_source(
    "utils_imagenet", "/Users/aleks/Desktop/qv/newPy/utils/utils_imagenet.py")

# Load the model definition.
tmp = getattr(innvestigate.applications.imagenet,
              os.environ.get("NETWORKNAME", "vgg16"))
net = tmp(load_weights=True, load_patterns="relu")
model = keras.models.load_model("./models/compareMethods.h5")
patterns = net["patterns"]
input_range = net["input_range"]
noise_scale = (input_range[1]-input_range[0]) * 0.1

# Methods we use and some properties.
methods = [
    # NAME                    OPT.PARAMS                POSTPROC FXN                TITLE
    # Show input.
    ("input",                 {},
     imgnetutils.image,         "Input"),

    # Function
    ("gradient",              {"postprocess": "abs"},
     imgnetutils.graymap,       "Gradient"),
    ("smoothgrad",            {"augment_by_n": 64,
                               "noise_scale": noise_scale,
                               "postprocess": "square"}, imgnetutils.graymap,       "SmoothGrad"),

    # # Signal
    # ("deconvnet",             {},
    #  imgnetutils.bk_proj,       "Deconvnet"),
    # ("guided_backprop",       {},
    #  imgnetutils.bk_proj,       "Guided Backprop",),
    # ("pattern.net",           {"patterns": patterns},
    #  imgnetutils.bk_proj,       "PatternNet"),

    # # Interaction
    # ("pattern.attribution",   {"patterns": patterns},
    #  imgnetutils.heatmap,       "PatternAttribution"),
    # ("deep_taylor.bounded",   {"low": input_range[0],
    #                            "high": input_range[1]}, imgnetutils.heatmap,       "DeepTaylor"),
    # ("input_t_gradient",      {},
    #  imgnetutils.heatmap,       "Input * Gradient"),
    # ("integrated_gradients",  {
    #  "reference_inputs": input_range[0], "steps": 64}, imgnetutils.heatmap,       "Integrated Gradients"),
    # ("lrp.z",                 {},
    #  imgnetutils.heatmap,       "LRP-Z"),
    # ("lrp.epsilon",           {"epsilon": 1},
    #  imgnetutils.heatmap,       "LRP-Epsilon"),
    # ("lrp.sequential_preset_a_flat", {"epsilon": 1},
    #  imgnetutils.heatmap,       "LRP-PresetAFlat"),
    # ("lrp.sequential_preset_b_flat", {"epsilon": 1},
    #  imgnetutils.heatmap,       "LRP-PresetBFlat"),
]


def runUsingModel():
    images, label_to_class_name = eutils.get_imagenet_data(
        net["image_shape"][0])

    model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)
    # Create analyzers.
    analyzers = []
    for method in methods:
        try:
            analyzer = innvestigate.create_analyzer(method[0],        # analysis method identifier
                                                    model_wo_softmax,  # model without softmax output
                                                    **method[1])      # optional analysis parameters
        except innvestigate.NotAnalyzeableModelException:
            # Not all methods work with all models.
            analyzer = None
        analyzers.append(analyzer)

    analysis = np.zeros([len(images), len(analyzers)]+net["image_shape"]+[3])
    text = []

    for i, (x, y) in enumerate(images):
        x = x[None, :, :, :]
        x_pp = imgnetutils.preprocess(x, net)
        # Predict final activations, probabilites, and label.
        presm = model_wo_softmax.predict_on_batch(x_pp)[0]
        prob = model.predict_on_batch(x_pp)[0]
        y_hat = prob.argmax()
        # Save prediction info:
        text.append((
            "%.2f" % presm.max(),             # pre-softmax logits
            "%.2f" % prob.max(),              # probabilistic softmax output
            "%s" % label_to_class_name[y_hat]  # predicted label
        ))

    return text


def getImage(url):
    urllib.request.urlretrieve(
        url, "utils/images/n02799071_986.jpg")
    return True


getImage("https://www.armytimes.com/resizer/fHaORr_V3wCVnEGj36XGrOfomUY=/1200x0/filters:quality(100)/arc-anglerfish-arc2-prod-mco.s3.amazonaws.com/public/2FFY4HEYTZBDBKU7LW2EN6T774.jpg")
answer = runUsingModel()
print(answer)
