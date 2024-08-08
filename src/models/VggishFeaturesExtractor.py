import sys
import os
import tensorflow.compat.v1 as tf

PROJECT_FOLDER = os.path.dirname(__file__).replace('/pipeline', '/models')
PARENT_PROJECT_FOLDER = os.path.dirname(PROJECT_FOLDER)
sys.path.append(PARENT_PROJECT_FOLDER)

from models.vggish_modules import vggish_input
from models.vggish_modules import vggish_params
from models.vggish_modules import vggish_slim

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class VggishFeaturesExtractor():
    def __init__(self, sample_rate):
        """
        Initialize the VggishFeaturesExtractor.
        """
        self.checkpoint_path = 'src/models/vggish_modules/vggish_model.ckpt'
        # self.target_duration = target_duration
        self.sample_rate = sample_rate
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            # Define the VGGish model in inference mode, load the checkpoint, and locate input and output tensors.
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(self.sess, self.checkpoint_path)

            self.features_tensor = self.sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
            self.embedding_tensor = self.sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
    def __call__(self, path):
        """
        Extract VGGish features from an audio file.

        Parameters:
        - path (str): The path to the audio file.

        Returns:
        - embedding_batch (numpy.ndarray): The extracted VGGish features.
        """
        # Convert the audio file to VGGish input batch
        audio_batch = vggish_input.wavfile_to_examples(path,self.sample_rate)
        # Set up a TensorFlow graph and session
        with self.graph.as_default():
            [embedding_batch] = self.sess.run([self.embedding_tensor], feed_dict={self.features_tensor: audio_batch})
            return embedding_batch
