import numpy as np
import pickle
import tensorflow as tf
import os
import sys
import urllib.request
import tarfile



def _print_download_progress(count, block_size, total_size):

    # Percentage completion.
    pct_complete = float(count * block_size) / total_size

    # Status-message. Note the \r which means the line should overwrite itself.
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()

def download():
    url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"
    data_dir = "inception/"
    print("Downloading Inception v3 Model ...")
    
    filename = url.split('/')[-1]
    file_path = os.path.join(data_dir, filename)

    if not os.path.exists(file_path):
        # Check if the download directory exists, otherwise create it.
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Download the file from the internet.
        file_path, _ = urllib.request.urlretrieve(url=url,
                                                  filename=file_path,
                                                  reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        tarfile.open(name=file_path, mode="r:gz").extractall(data_dir)

        print("Done.")
    else:
        print("Data has apparently already been downloaded and unpacked.")


class Inception:

    # Name of the tensor for the output of the Inception model.
    tensor_name_transfer_layer = "pool_3:0"

    def __init__(self):
        self.graph = tf.Graph()

        # Set the new graph as the default.
        with self.graph.as_default():

            # Open the graph-def file for binary reading.
            path = os.path.join("inception/", "classify_image_graph_def.pb")
            with tf.gfile.FastGFile(path, 'rb') as file:
                # The graph-def is a saved copy of a TensorFlow graph.
                # First we need to create an empty graph-def.
                graph_def = tf.GraphDef()

                # Then we load the proto-buf file into the graph-def.
                graph_def.ParseFromString(file.read())

                # Finally we import the graph-def to the default TensorFlow graph.
                tf.import_graph_def(graph_def, name='')


        # Get the tensor for the last layer of the graph, aka. the transfer-layer.
        self.transfer_layer = self.graph.get_tensor_by_name(self.tensor_name_transfer_layer)

        # Get the number of elements in the transfer-layer.
        self.transfer_len = self.transfer_layer.get_shape()[3]

        # Create a TensorFlow session for executing the graph.
        self.session = tf.Session(graph=self.graph)


    def transfer_values(self, image):
        feed_dict = {self.tensor_name_input_image: image}

        transfer_values = self.session.run(self.transfer_layer, feed_dict=feed_dict)

        # Reduce to a 1-dim array.
        transfer_values = np.squeeze(transfer_values)

        return transfer_values

def process_images(fn, images):

    # Number of images.
    num_images = len(images)


    result = [None] * num_images

    for i in range(num_images):
        msg = "\r- Processing image: {0:>6} / {1}".format(i+1, num_images)

        # Print the status message.
        sys.stdout.write(msg)
        sys.stdout.flush()

        # Process the image
        result[i] = fn(image=images[i])

    # Print newline.
    print()

    result = np.array(result)

    return result


def transfer_values_cache(cache_path, model, images):
    def fn():
        return process_images(fn=model.transfer_values, images=images)

    transfer_values = sauvgard(save_path=cache_path, fn=fn)

    return transfer_values

def sauvgard(save_path, fn, *args, **kwargs):
    # If the save-file exists.
    if os.path.exists(save_path):
        # Load the cached data from the file.
        with open(save_path, mode='rb') as file:
            obj = pickle.load(file)

        print("- Data loaded from cache-file: " + save_path)
    else:
        # The cache-file does not exist.

        # Execute function
        obj = fn(*args, **kwargs)

        # Save the data to a cache-file.
        with open(save_path, mode='wb') as file:
            pickle.dump(obj, file)

        print("- Data saved to file: " + save_path)

    return obj