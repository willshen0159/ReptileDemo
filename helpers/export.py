import numpy as np
import time
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import pickle
import copy

model = keras.models.load_model("step_3.h5")
variables = model.trainable_variables
with open("model.js", "w") as fp:
    fp.write("var trainedParameters = [\n")
    for i, var in enumerate(variables):
        fp.write('    new jsnet.Tensor([%s], [%s])%s\n' % (','.join(str(x) for x in var.shape), ','.join(str(x) % x for x in var.read_value().numpy().flatten()), '' if (i == len(variables) - 1) else ','))
    fp.write("];")