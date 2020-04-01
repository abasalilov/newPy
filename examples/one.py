from __future__ import\
    absolute_import, print_function, division, unicode_literals
import matplotlib
import tensorflow as tf
import innvestigate.utils as iutils
import innvestigate.applications.imagenet
import innvestigate
import keras.models
import keras.backend
import keras
import os
import numpy as np
import imp
import warnings
# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL.Image
import shutil

warnings.simplefilter('ignore')
base_dir = os.path.dirname(__file__)
print('basedir', base_dir)

# Use utility libraries to focus on relevant iNNvestigate routines.
eutils = imp.load_source("utils", os.path.join(base_dir, "utils.py"))
imgnetutils = imp.load_source(
    "utils_imagenet", os.path.join(base_dir, "./utils_imagenet.py"))

# Load the model definition.
tmp = getattr(innvestigate.applications.imagenet,
              os.environ.get("NETWORKNAME", "vgg16"))
net = tmp(load_weights=True, load_patterns="relu")

# Build the model.
model = keras.models.Model(inputs=net["in"], outputs=net["sm_out"])
model.compile(optimizer="adam", loss="categorical_crossentropy")

# Handle input depending on model and backend.
channels_first = keras.backend.image_data_format() == "channels_first"
color_conversion = "BGRtoRGB" if net["color_coding"] == "BGR" else None

images, label_to_class_name = eutils.get_imagenet_data(net["image_shape"][0])

if not len(images):
    raise Exception("Please download the example images using: "
                    "'innvestigate/examples/images/wget_imagenet_2011_samples.sh'")

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

    # Signal
    ("deconvnet",             {},
     imgnetutils.bk_proj,       "Deconvnet"),
    ("guided_backprop",       {},
     imgnetutils.bk_proj,       "Guided Backprop",),
    ("pattern.net",           {"patterns": patterns},
     imgnetutils.bk_proj,       "PatternNet"),

    # Interaction
    ("pattern.attribution",   {"patterns": patterns},
     imgnetutils.heatmap,       "PatternAttribution"),
    ("deep_taylor.bounded",   {"low": input_range[0],
                               "high": input_range[1]}, imgnetutils.heatmap,       "DeepTaylor"),
    ("input_t_gradient",      {},
     imgnetutils.heatmap,       "Input * Gradient"),
    ("integrated_gradients",  {"reference_inputs": input_range[0],
                               "steps": 64},            imgnetutils.heatmap,       "Integrated Gradients"),
    ("lrp.z",                 {},
     imgnetutils.heatmap,       "LRP-Z"),
    ("lrp.epsilon",           {"epsilon": 1},
     imgnetutils.heatmap,       "LRP-Epsilon"),
    ("lrp.sequential_preset_a_flat", {"epsilon": 1},
     imgnetutils.heatmap,       "LRP-PresetAFlat"),
    ("lrp.sequential_preset_b_flat", {"epsilon": 1},
     imgnetutils.heatmap,       "LRP-PresetBFlat"),
]


# Create model without trailing softmax
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
    # Add batch axis.
    x = x[None, :, :, :]
    x_pp = imgnetutils.preprocess(x, net)

    # Predict final activations, probabilites, and label.
    presm = model_wo_softmax.predict_on_batch(x_pp)[0]
    prob = model.predict_on_batch(x_pp)[0]
    y_hat = prob.argmax()

    # Save prediction info:
    text.append(("%s" % label_to_class_name[y],    # ground truth label
                 "%.2f" % presm.max(),             # pre-softmax logits
                 "%.2f" % prob.max(),              # probabilistic softmax output
                 "%s" % label_to_class_name[y_hat]  # predicted label
                 ))

    for aidx, analyzer in enumerate(analyzers):
        if methods[aidx][0] == "input":
            # Do not analyze, but keep not preprocessed input.
            a = x/255
        elif analyzer:
            # Analyze.
            a = analyzer.analyze(x_pp)

            # Apply common postprocessing, e.g., re-ordering the channels for plotting.
            a = imgnetutils.postprocess(a, color_conversion, channels_first)
            # Apply analysis postprocessing, e.g., creating a heatmap.
            a = methods[aidx][2](a)
        else:
            a = np.zeros_like(image)
        # Store the analysis.
        analysis[i, aidx] = a[0]

# Prepare the grid as rectengular list
grid = [[analysis[i, j] for j in range(analysis.shape[1])]
        for i in range(analysis.shape[0])]
# Prepare the labels
label, presm, prob, pred = zip(*text)
row_labels_left = [('label: {}'.format(label[i]),
                    'pred: {}'.format(pred[i])) for i in range(len(label))]
row_labels_right = [('logit: {}'.format(presm[i]),
                     'prob: {}'.format(prob[i])) for i in range(len(label))]
col_labels = [''.join(method[3]) for method in methods]

# Plot the analysis.
eutils.plot_image_grid(grid, row_labels_left, row_labels_right, col_labels,
                       file_name=os.environ.get("plot_file_name", None))
