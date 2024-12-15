from __future__ import annotations

import base64
import h5py
import logging
import os
import torch
import numpy as np
import requests
import pandas as pd
import importlib.resources as pkg_resources

from collections import OrderedDict

DATA_DIR = os.path.join(os.getcwd(), '.pyment')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

try:
    METADATA_DIR = pkg_resources.files('pyment').joinpath('data')
except:
    METADATA_DIR = os.path.join(DATA_DIR, 'pyment', 'data')

logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)

class WeightRepository:
    """ A class representing a repository for model weights. Allows for
    looking up and, optionally, downloading, weights for the pretrained
    models in the model zoo. """

    _BASE_URL = 'https://api.github.com/repos/estenhl/pyment-public/git/blobs'

    @staticmethod
    def _download_weight(sha: str, filename: str,
                         base_url: str = _BASE_URL) -> None:
        url = f'{base_url}/{sha}'
        logging.info('Downloading %s to %s', url, filename)

        resp = requests.get(url, timeout=30, stream=True)
        content = resp.json()['content']

        folder = os.path.dirname(filename)

        if not os.path.isdir(folder):
            os.makedirs(folder)

        with open(filename, 'wb') as f:
            f.write(base64.b64decode(content))

    @staticmethod
    def get_weights(architecture: str, name: str,
                    folder: str = MODELS_DIR) -> str:
        """ Returns the path to a file containing weights to the
        given model. If necessary, the weights are downloaded.

        Parameters
        ----------
        architecture : str
            The name of the architecture to which the weights belong.
        name : str
            The name of the pretrained model.
        folder : str
            The folder where weights are stored. Defaults to the
            MODELS_DIR set in utils.py in the top-level folder of the
            package

        Returns
        -------
        string
            The path to the weights belonging to the given architecture
            and name

        Raises
        ------
        KeyError
            If no pretrained weight corresponding to the given
            architecture and name exists
        ValueError
            If multiple pretrained weights corresponding to the given
            architecture and name exists
        """
        table = pd.read_csv(os.path.join(METADATA_DIR, 'models.csv'))
        key = (architecture, name)
        rows = table.loc[(table['architecture'] == architecture) & \
                        (table['name'] == name)]

        if len(rows) == 0:
            error = f'Unknown architecture/name combination {key}'
            logging.error(error)
            raise KeyError(error)
        elif len(rows) > 1:
            error = f'Multiple entries for architecture/name combination {key}'
            logging.error(error)
            raise ValueError(error)

        row = rows.iloc[0]
        path = os.path.join(folder, row['filename'])

        if not os.path.isfile(path):
            sha = row['sha']
            WeightRepository._download_weight(sha, path)

        return path

    @staticmethod
    def convert_to_torch(weights_path: str, model_state_dictionary: dict, model_type: str = "sfcn-reg", model_name: str = "Regression3DSFCN") -> OrderedDict[str, np.ndarray]:
        weight_dictionary = WeightRepository.__load_tf_weights(weights_path, model_name)
        # Translate it

        weight_dictionary = WeightRepository.__translate_tf_weights(tf_weights=weight_dictionary, state_dictionary=model_state_dictionary,
                                                                    model_type=model_type, model_name=model_name)
        return weight_dictionary


    @staticmethod
    def __load_tf_weights(weights_path: str, model_name: str = "Regression3DSFCN") -> OrderedDict[str, np.ndarray]:
        ordering = {
            "conv": ("kernel:0", "bias:0"),
            "norm": ("gamma:0", "beta:0", "moving_mean:0", "moving_variance:0"),
            "predictions": ("kernel:0", "bias:0"),
            f"{model_name}": ('block1', 'block2', 'block3', 'block4', 'block5', 'expand_dims', 'inputs', 'restrict', 'top', 'predictions')
        }

        def order_key(lst, main_key, ordering):
            lst = list(lst)
            if main_key in ordering:
                # Get the order of keys for the specified main_key
                main_order = ordering[main_key]
                
                # Create a helper function to extract the key part (e.g., "kernel:0" from "ciao/kernel:0")
                def get_key(item):
                    return item.split('/')[-1]
                
                # Split the list into items that match the main_key ordering and others
                ordered_items = [item for item in lst if get_key(item) in main_order]
                unordered_items = [item for item in lst if get_key(item) not in main_order]
                
                # Sort ordered items based on their position in main_order
                ordered_items.sort(key=lambda item: main_order.index(get_key(item)))
                
                # Return the concatenated list of ordered and unordered items
                return ordered_items + unordered_items
            else:
                # If main_key is not in ordering, return the original list
                return lst

        tf_weights = OrderedDict()
        with h5py.File(weights_path, 'r') as h5_file:
            def recursively_load(group: h5py.Group, prv_key="", prefix=""):
                for key in order_key(group.keys(), prv_key, ordering):
                    item = group[key]
                    full_key = f"{prefix}/{key}" if prefix else key
                    if isinstance(item, h5py.Group):
                        recursively_load(item, key, full_key)
                    else:
                        tf_weights[full_key] = np.array(item)
            recursively_load(h5_file)
        return tf_weights
    
    @staticmethod
    def __translate_tf_weights(tf_weights: dict, state_dictionary: dict, model_type: str = "sfcn-reg", model_name: str = 'Regression3DSFCN') -> OrderedDict[str, np.ndarray]:
        new_weight_dict = OrderedDict()
        with torch.no_grad():
            for tf_key, tf_value in tf_weights.items():
                pt_key = model_name + tf_key.split(model_name)[-1]
                if "kernel:0" in tf_key:
                    # Convert convolutional weights (NHWC -> NCHW)
                    if 'prediction' in tf_key:
                        pt_key = pt_key.replace("/", "_").replace("Regression3DSFCN_", f"linear.{model_type}_").replace("_kernel:0", ".weight")
                        new_weight_dict[pt_key] = torch.tensor(tf_value.transpose(1, 0), dtype=torch.float32)
                    else:
                        pt_key = pt_key.replace("/", "_").replace("Regression3DSFCN_", f"fn1.{model_type}_").replace("_kernel:0", ".weight")
                        new_weight_dict[pt_key] = torch.tensor(tf_value.transpose(4, 3, 0, 1, 2), dtype=torch.float32)
                elif "bias:0" in pt_key:
                    # Convert biases
                    if 'prediction' in tf_key:
                        pt_key = pt_key.replace("/", "_").replace("Regression3DSFCN_", f"linear.{model_type}_").replace("_bias:0", ".bias")
                    else:
                        pt_key = pt_key.replace("/", "_").replace("Regression3DSFCN_", f"fn1.{model_type}_").replace("_bias:0", ".bias")
                    new_weight_dict[pt_key] = torch.tensor(tf_value, dtype=torch.float32)
                elif "gamma:0" in pt_key:
                    # BatchNorm weight
                    pt_key = pt_key.replace("/", "_").replace("Regression3DSFCN_", f"fn1.{model_type}_").replace("_gamma:0", ".weight")
                    new_weight_dict[pt_key] = torch.tensor(tf_value, dtype=torch.float32)
                elif "beta:0" in pt_key:
                    # BatchNorm bias
                    pt_key = pt_key.replace("/", "_").replace("Regression3DSFCN_", f"fn1.{model_type}_").replace("_beta:0", ".bias")
                    new_weight_dict[pt_key] = torch.tensor(tf_value, dtype=torch.float32)
                elif "moving_mean:0" in pt_key:
                    # BatchNorm running mean
                    pt_key = pt_key.replace("/", "_").replace("Regression3DSFCN_", f"fn1.{model_type}_").replace("_moving_mean:0", ".running_mean")
                    new_weight_dict[pt_key] = torch.tensor(tf_value, dtype=torch.float32)
                elif "moving_variance:0" in pt_key:
                    # BatchNorm running variance
                    pt_key = pt_key.replace("/", "_").replace("Regression3DSFCN_", f"fn1.{model_type}_").replace("_moving_variance:0", ".running_var")
                    new_weight_dict[pt_key] = torch.tensor(tf_value, dtype=torch.float32)
            return OrderedDict({k: new_weight_dict[k] for k in state_dictionary.keys() if "num_batches_tracked" not in k})