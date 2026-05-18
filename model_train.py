

Commands
--------------

.. code-block:: shell

   # Do a "dry run" test training of the model
   mltk train keyword_spotting_pacman_v3-test

   # Train the model
   mltk train keyword_spotting_pacman_v3

   # Evaluate the trained model .tflite model
   mltk evaluate keyword_spotting_pacman_v3 --tflite

   # Profile the model in the MVP hardware accelerator simulator
   mltk profile keyword_spotting_pacman_v3 --accelerator MVP --estimates

   # Profile the model on a physical development board
   mltk profile keyword_spotting_pacman_v3  --accelerator MVP --device

   # Run the model in the audio classifier on the local PC
   mltk classify_audio keyword_spotting_pacman_v3 --verbose

   # Run the model in the audio classifier on the physical device
   mltk classify_audio keyword_spotting_pacman_v3 --device --verbose --accelerator MVP


Model Specification
---------------------

..  literalinclude:: ../../../../../../../mltk/models/siliconlabs/keyword_spotting_pacman_v3.py
    :language: python
    :lines: 590-

"""
# pylint: disable=redefined-outer-name

# Import the Tensorflow packages
# required to build the model layout
import os
import math
from typing import Tuple, Dict, List

import numpy as np
from numpy.random import RandomState
import tensorflow as tf
import mltk.core as mltk_core

# Import the AudioFeatureGeneratorSettings which we'll configure
from mltk.core.preprocess.audio.audio_feature_generator import AudioFeatureGeneratorSettings
from mltk.core.preprocess.utils import tf_dataset as tf_dataset_utils
from mltk.core.preprocess.utils import audio as audio_utils
from mltk.core.preprocess.utils import image as image_utils
from mltk.core.keras.callbacks import SteppedLearnRateScheduler
from mltk.utils.path import create_user_dir
from mltk.core.preprocess.utils import (split_file_list, shuffle_file_list_by_group)
from mltk.utils.python import install_pip_package
from mltk.models.shared import tenet
from mltk.datasets import audio as audio_datasets


##########################################################################################
# Instantiate the MltkModel instance
#

# @mltk_model
class MyModel(
    mltk_core.MltkModel,    # We must inherit the MltkModel class
    mltk_core.TrainMixin,   # We also inherit the TrainMixin since we want to train this model
    mltk_core.DatasetMixin, # We also need the DatasetMixin mixin to provide the relevant dataset properties
    mltk_core.EvaluateClassifierMixin,  # While not required, also inherit EvaluateClassifierMixin to help will generating evaluation stats for our classification model
    mltk_core.WeightsAndBiasesMixin # This allows for posting model info to https://wandb.ai
):
    pass
my_model = MyModel()

##########################################################################################
# General Settings

# For better tracking, the version should be incremented any time a non-trivial change is made
# NOTE: The version is optional and not used directly used by the MLTK
my_model.version = 1
# Provide a brief description about what this model models
# This description goes in the "description" field of the .tflite model file
my_model.description = 'Keyword spotting classifier to detect: food, water, restroom, emergency, pain, assist, noise, help'


##########################################################################################
# Training Basic Settings

# This specifies the number of times we run the training.
# We just set this to a large value since we're using SteppedLearnRateScheduler
# to control when training completes

# Specify how many samples to pass through the model
# before updating the training gradients.
# Typical values are 10-64
# NOTE: Larger values require more memory and may not fit on your GPU

my_model.epochs = 200
my_model.batch_size = 16


##########################################################################################
# Define the model architecture
#

def my_model_builder(model: MyModel) -> tf.keras.Model:
    """Build the Keras model"""
    input_shape = model.input_shape
    time_size, feature_size, _ = input_shape
    input_shape = (time_size, 1, feature_size)

    keras_model = tenet.TENet12(
        input_shape=input_shape,
        classes=model.n_classes,
        channels=24,
        blocks=3,
    )

    for layer in keras_model.layers:
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = tf.keras.regularizers.l2(0.001)

    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-8),
        metrics=['accuracy']
    )

    return keras_model

my_model.build_model_function = my_model_builder
# TENet uses a custom layer, be sure to add it to the keras_custom_objects
# so that we can load the corresponding .h5 model file
my_model.keras_custom_objects['MultiScaleTemporalConvolution'] = tenet.MultiScaleTemporalConvolution



##########################################################################################
# Training callback Settings
#


# The MLTK enables the tf.keras.callbacks.ModelCheckpoint by default.
my_model.checkpoint['monitor'] =  'val_accuracy'


# We use a custom learn rate schedule that is defined in:
# https://github.com/google-research/google-research/tree/master/kws_streaming
my_model.train_callbacks = [
    tf.keras.callbacks.TerminateOnNaN(),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
]


##########################################################################################
# Specify AudioFeatureGenerator Settings
# See https://siliconlabs.github.io/mltk/docs/audio/audio_feature_generator.html
#
frontend_settings = AudioFeatureGeneratorSettings()

frontend_settings.sample_rate_hz = 16000
frontend_settings.sample_length_ms = 1000                       # A 1s buffer should be enough to capture the keywords
frontend_settings.window_size_ms = 30
frontend_settings.window_step_ms = 10
frontend_settings.filterbank_n_channels = 104                   # We want this value to be as large as possible
                                                                # while still allowing for the ML model to execute efficiently on the hardware
frontend_settings.filterbank_upper_band_limit = 7500.0
frontend_settings.filterbank_lower_band_limit = 125.0           # The dev board mic seems to have a lot of noise at lower frequencies

frontend_settings.noise_reduction_enable = True                 # Enable the noise reduction block to help ignore background noise in the field
frontend_settings.noise_reduction_smoothing_bits = 10
frontend_settings.noise_reduction_even_smoothing =  0.025
frontend_settings.noise_reduction_odd_smoothing = 0.06
frontend_settings.noise_reduction_min_signal_remaining = 0.40   # This value is fairly large (which makes the background noise reduction small)
                                                                # But it has been found to still give good results
                                                                # i.e. There is still some background noise reduction,
                                                                # but the actual signal is still (mostly) untouched

frontend_settings.dc_notch_filter_enable = True                 # Enable the DC notch filter, to help remove the DC signal from the dev board's mic
frontend_settings.dc_notch_filter_coefficient = 0.95

frontend_settings.quantize_dynamic_scale_enable = True          # Enable dynamic quantization, this dynamically converts the uint16 spectrogram to int8
frontend_settings.quantize_dynamic_scale_range_db = 40.0


# Add the Audio Feature generator settings to the model parameters
# This way, they are included in the generated .tflite model file
# See https://siliconlabs.github.io/mltk/docs/guides/model_parameters.html
my_model.model_parameters.update(frontend_settings)


##########################################################################################
# Specify the other dataset settings
#

my_model.input_shape = frontend_settings.spectrogram_shape + (1,)

# Add the direction keywords plus a _unknown_ meta class
my_model.classes = ['food','water','restroom','emergency','pain','assist','noise','help', '_unknown_']
unknown_class_id = my_model.classes.index('_unknown_')

# Ensure the class weights are balanced during training
# https://towardsdatascience.com/why-weight-the-importance-of-training-on-balanced-datasets-f1e54688e7df
my_model.class_weights = 'balanced'


##########################################################################################
# TF-Lite converter settings
#

my_model.tflite_converter['optimizations'] = [tf.lite.Optimize.DEFAULT]
my_model.tflite_converter['supported_ops'] = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
my_model.tflite_converter['inference_input_type'] = np.int8
my_model.tflite_converter['inference_output_type'] = np.int8
# Automatically generate a representative dataset from the validation data
my_model.tflite_converter['representative_dataset'] = 'generate'


validation_split = 0.20

# Uncomment this to dump the augmented audio samples to the log directory
# DO NOT forget to disable this before training the model as it will generate A LOT of data
#data_dump_dir = my_model.create_log_dir('dataset_dump')

# This is the directory where the dataset will be extracted
dataset_dir = 'kws_dataset'


##########################################################################################
# Create the audio augmentation pipeline
#

# Install the other 3rd party packages required from preprocessing
install_pip_package('audiomentations')

import librosa
import audiomentations


def audio_pipeline_with_augmentations(
    path_batch:np.ndarray,
    label_batch:np.ndarray,
    seed:np.ndarray
) -> np.ndarray:
    """Augment a batch of audio clips and generate spectrograms

    This does the following, for each audio file path in the input batch:
    1. Read audio file
    2. Adjust its length to fit within the specified length
    3. Apply random augmentations to the audio sample using audiomentations
    4. Convert to the specified sample rate (if necessary)
    5. Generate a spectrogram from the augmented audio sample
    6. Dump the augmented audio and spectrogram (if necessary)

    NOTE: This will be execute in parallel across *separate* subprocesses.

    Arguments:
        path_batch: Batch of audio file paths
        label_batch: Batch of corresponding labels
        seed: Batch of seeds to use for random number generation,
            This ensures that the "random" augmentations are reproducible

    Return:
        Generated batch of spectrograms from augmented audio samples
    """
    batch_length = path_batch.shape[0]
    height, width = frontend_settings.spectrogram_shape
    x_shape = (batch_length, height, 1, width)
    x_batch = np.empty(x_shape, dtype=np.int8)

    # This is the amount of padding we add to the beginning of the sample
    # This allows for "warming up" the noise reduction block
    padding_length_ms = 1000
    padded_frontend_settings = frontend_settings.copy()
    padded_frontend_settings.sample_length_ms += padding_length_ms

    # For each audio sample path in the current batch
    for i, (audio_path, labels) in enumerate(zip(path_batch, label_batch)):
        class_id = np.argmax(labels)
        np.random.seed(seed[i])

        rn = np.random.random()
        # 3% of the time we want to replace the "unknown" sample with silence
        if class_id == unknown_class_id and rn < 0.03:
            original_sample_rate = frontend_settings.sample_rate_hz
            sample = np.zeros((original_sample_rate,), dtype=np.float32)
            audio_path = 'silence.wav'.encode('utf-8')
        else:
            # Read the audio file
            try:
                sample, original_sample_rate = audio_utils.read_audio_file(audio_path, return_numpy=True, return_sample_rate=True)
            except Exception as e:
                raise RuntimeError(f'Failed to read: {audio_path}, err: {e}')

        # Create a buffer to hold the padded sample
        padding_length = int((original_sample_rate * padding_length_ms) / 1000)
        padded_sample_length = int((original_sample_rate * padded_frontend_settings.sample_length_ms) / 1000)
        padded_sample = np.zeros((padded_sample_length,), dtype=np.float32)


        # Adjust the audio clip to the length defined in the frontend_settings
        out_length = int((original_sample_rate * frontend_settings.sample_length_ms) / 1000)
        sample = audio_utils.adjust_length(
            sample,
            out_length=out_length,
            trim_threshold_db=30,
            offset=np.random.uniform(0, 1)
        )
        padded_sample[padding_length:padding_length+len(sample)] += sample



        # Initialize the global audio augmentations instance
        # NOTE: We want this to be global so that we only initialize it once per subprocess
        audio_augmentations = globals().get('audio_augmentations', None)
        if audio_augmentations is None:
            audio_augmentations = audiomentations.Compose(
                p=1.0,
                transforms=[
                    audiomentations.Gain(min_gain_db=-3, max_gain_db=3, p=0.8),
                    audiomentations.TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
                    audiomentations.PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
                    audiomentations.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                    audiomentations.Shift(min_shift=-0.2, max_shift=0.2, p=0.5),
                ]
                )
            globals()['audio_augmentations'] = audio_augmentations

        # Apply random augmentations to the audio sample
        augmented_sample = audio_augmentations(padded_sample, original_sample_rate)

        # Convert the sample rate (if necessary)
        if original_sample_rate != frontend_settings.sample_rate_hz:
            augmented_sample = audio_utils.resample(
                augmented_sample,
                orig_sr=original_sample_rate,
                target_sr=frontend_settings.sample_rate_hz
            )

        # Ensure the sample values are within (-1,1)
        augmented_sample = np.clip(augmented_sample, -1.0, 1.0)

        # Generate a spectrogram from the augmented audio sample
        spectrogram = audio_utils.apply_frontend(
            sample=augmented_sample,
            settings=padded_frontend_settings,
            dtype=np.int8
        )

        # The input audio sample was padded with padding_length_ms of background noise
        # Drop the padded background noise from the final spectrogram used for training
        spectrogram = spectrogram[-height:, :]
        # The output spectrogram is 2D, add a channel dimension to make it 3D:
        # (height, width, channels=1)

        # Convert the spectrogram dimension from
        # <time, features> to
        # <time, 1, features>
        spectrogram = np.expand_dims(spectrogram, axis=-2)

        x_batch[i] = spectrogram

        # Dump the augmented audio sample AND corresponding spectrogram (if necessary)
        data_dump_dir = globals().get('data_dump_dir', None)
        if data_dump_dir:
            try:
                from cv2 import cv2
            except:
                import cv2

            fn = os.path.basename(audio_path.decode('utf-8'))

            audio_dump_path = f'{data_dump_dir}/{class_id}-{fn[:-4]}-{seed[0]}.wav'
            spectrogram_dumped = np.squeeze(spectrogram, axis=-2)
            # Transpose to put the time on the x-axis
            spectrogram_dumped = np.transpose(spectrogram_dumped)
            # Convert from int8 to uint8
            spectrogram_dumped = np.clip(spectrogram_dumped +128, 0, 255)
            spectrogram_dumped = spectrogram_dumped.astype(np.uint8)
            # Increase the size of the spectrogram to make it easier to see as a jpeg
            spectrogram_dumped = cv2.resize(spectrogram_dumped, (height*3,width*3))

            valid_sample_length = int((frontend_settings.sample_length_ms * frontend_settings.sample_rate_hz) / 1000)
            valid_augmented_sample = augmented_sample[-valid_sample_length:]
            audio_dump_path = audio_utils.write_audio_file(
                audio_dump_path,
                valid_augmented_sample,
                sample_rate=frontend_settings.sample_rate_hz
            )
            image_dump_path = audio_dump_path.replace('.wav', '.jpg')
            jpg_data = cv2.applyColorMap(spectrogram_dumped, cv2.COLORMAP_HOT)
            cv2.imwrite(image_dump_path, jpg_data)


    return x_batch


##########################################################################################
# Define the MltkDataset object
# NOTE: This class is optional but is useful for organizing the code
#
##########################################################################################
# Define the MltkDataset object
# NOTE: This class is optional but is useful for organizing the code
#
class MyDataset(mltk_core.MltkDataset):

    def __init__(self):
        super().__init__()
        self.pools = []
        self.summary = ''

    def summarize_dataset(self) -> str:
        """Return a string summary of the dataset"""
        s = self.summary
        s += mltk_core.MltkDataset.summarize_class_counts(my_model.class_counts)
        return s

    def load_dataset(
        self,
        subset: str,
        test:bool = False,
        **kwargs
    ) -> Tuple[tf.data.Dataset, None, tf.data.Dataset]:
        """Load the dataset subset"""
        if subset == 'training':
            x = self.load_subset('training', test=test)
            validation_data = self.load_subset('validation', test=test)
            return x, None, validation_data
        else:
            x = self.load_subset('validation', test=test)
            return x

    def unload_dataset(self):
        """Unload the dataset by shutting down the processing pools"""
        for pool in self.pools:
            pool.shutdown()
        self.pools.clear()

    def load_subset(self, subset:str, test:bool) -> tf.data.Dataset:
        """Load the subset"""
        if subset in ('validation', 'evaluation'):
            split = (0, validation_split)
        elif subset == 'training':
            split = (validation_split, 1)
            data_dump_dir = globals().get('data_dump_dir', None)
            if data_dump_dir:
                print(f'\n\n*** Dumping augmented samples to: {data_dump_dir}\n\n')
        else:
            split = None
            my_model.class_counts = {}

        # Using local dataset - no downloads needed
        # Dataset should be at: D:/SiliconLabs/Mughil/datasets

        # Create a tf.data.Dataset from the extracted dataset directory
        max_samples_per_class = my_model.batch_size if test else -1
        class_counts = my_model.class_counts[subset] if subset else my_model.class_counts
        features_ds, labels_ds = tf_dataset_utils.load_audio_directory(
            directory=dataset_dir,
            classes=my_model.classes,
            onehot_encode=True,
            shuffle=True,
            seed=42,
            max_samples_per_class=max_samples_per_class,
            unknown_class_percentage=0,
            split=split,
            return_audio_data=False,
            class_counts=class_counts,
            list_valid_filenames_in_directory_function=self.list_valid_filenames_in_directory,
            process_samples_function=self.add_unknown_samples
        )

        if subset:
            per_job_batch_multiplier = 1000
            per_job_batch_size = my_model.batch_size * per_job_batch_multiplier

            try:
                seed_counter = tf.data.Dataset.counter()
            except:
                seed_counter = tf.data.experimental.Counter()
            features_ds = features_ds.zip((features_ds, labels_ds, seed_counter))

            features_ds = features_ds.batch(per_job_batch_size // per_job_batch_multiplier, drop_remainder=True)
            labels_ds = labels_ds.batch(per_job_batch_size // per_job_batch_multiplier, drop_remainder=True)
            features_ds, pool = tf_dataset_utils.parallel_process(
                features_ds,
                audio_pipeline_with_augmentations,
                dtype=np.int8,
                n_jobs=4,
                name=subset,
            )
            self.pools.append(pool)
            features_ds = features_ds.unbatch()
            labels_ds = labels_ds.unbatch()
            features_ds = features_ds.prefetch(per_job_batch_size)

        ds = tf.data.Dataset.zip((features_ds, labels_ds))

        if not test:
            ds = ds.shuffle(per_job_batch_size if subset else my_model.batch_size, reshuffle_each_iteration=True)

        ds = ds.batch(my_model.batch_size)
        ds = ds.prefetch(2)

        return ds
    def list_valid_filenames_in_directory(
        self,
        base_directory:str,
        search_class:str,
        white_list_formats:List[str],
        split:float,
        follow_links:bool,
        shuffle_index_directory:str
    ) -> Tuple[str, List[str]]:
        """Return a list of valid file names for the given class

        This is called by the tf_dataset_utils.load_audio_directory() API.

        # This uses shuffle_file_list_by_group() helper function so that the same "voices"
        # are only present in a particular subset.
        """
        assert shuffle_index_directory is None, 'Shuffling the index is not supported by this dataset'

        file_list = []
        index_path = f'{base_directory}/.index/{search_class}.txt'

        # If the index file exists, then read it
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                for line in f:
                    file_list.append(line.strip())

        else:
            # Else find all files for the given class in the search directory
            class_base_dir = f'{base_directory}/{search_class}/'
            for root, _, files in os.walk(base_directory, followlinks=follow_links):
                root = root.replace('\\', '/') + '/'
                if not root.startswith(class_base_dir):
                    continue

                for fname in files:
                    if not fname.lower().endswith(white_list_formats):
                        continue
                    abs_path = os.path.join(root, fname)
                    if os.path.getsize(abs_path) == 0:
                        continue
                    rel_path = os.path.relpath(abs_path, base_directory)
                    file_list.append(rel_path.replace('\\', '/'))


                # Shuffle the voice groups
                # then flatten into list
                # This way, when the list is split into training and validation sets
                # the same voice only appears in one subset
                file_list = shuffle_file_list_by_group(file_list, get_sample_group_id_from_path)

                # Write the file list file
                mltk_core.get_mltk_logger().info(f'Generating index for "{search_class}" ({len(file_list)} samples): {index_path}')
                os.makedirs(os.path.dirname(index_path), exist_ok=True)
                with open(index_path, 'w') as f:
                    for p in file_list:
                        f.write(p + '\n')

        if len(file_list) == 0:
            raise RuntimeError(f'No samples found for class: {search_class}')


        n_files = len(file_list)
        if split[0] == 0:
            start = 0
            stop = math.ceil(split[1] * n_files)

            # We want to ensure the same person isn't in both subsets
            # So, ensure that the split point does NOT
            # split with file names with the same hash
            # recall: same hash = same person saying word

            # Get the hash of the other subset
            other_subset_hash = get_sample_group_id_from_path(file_list[stop])
            # Keep moving the 'stop' index back while
            # it's index matches the otherside
            while stop > 0 and get_sample_group_id_from_path(file_list[stop-1]) == other_subset_hash:
                stop -= 1

        else:
            start = math.ceil(split[0] * n_files)
            # Get the hash of the this subset
            this_subset_hash = get_sample_group_id_from_path(file_list[start])
            # Keep moving the 'start' index back while
            # it's index matches this side's
            while start > 0 and get_sample_group_id_from_path(file_list[start-1]) == this_subset_hash:
                start -= 1

            stop = n_files

        filenames = file_list[start:stop]

        return search_class, filenames

    def add_unknown_samples(
        self,
        directory:str,
        sample_paths:Dict[str,str],
        split:Tuple[float,float],
        follow_links:bool,
        white_list_formats:List[str],
        shuffle:bool,
        seed:int,
        **kwargs
    ):
        """Handle unknown samples from local dataset"""
        file_list = []
        
        # If you have an _unknown_ folder with .wav files
        unknown_dir = f'{dataset_dir}/_unknown_'
        if os.path.exists(unknown_dir) and os.path.isdir(unknown_dir):
            for fn in os.listdir(unknown_dir):
                if fn.endswith('.wav'):
                    file_list.append(f'_unknown_/{fn}')
        
        if len(file_list) > 0:
            file_list = sorted(file_list)
            sample_paths['_unknown_'] = split_file_list(file_list, split)
        else:
            # No unknown samples found
            sample_paths['_unknown_'] = []


def get_sample_group_id_from_path(p: str) -> str:
    """Extract a speaker/group ID from the filename.

    Ensures the same speaker does not appear in both
    training and validation subsets.
    """
    fn = os.path.basename(p)

    # Remove extension
    fn = fn.replace('.wav', '').replace('.mp3', '')

    # Case 1: Google Speech Commands
    if '_nohash_' in fn:
        return fn.split('_nohash_')[0]

    # Case 2: Common Voice
    if fn.startswith('common_voice_'):
        return fn.split('_')[-1]

    # Case 3: Cloud TTS (azure/aws/gcp)
    if fn.startswith(('gcp_', 'azure_', 'aws_')):
        return fn.split('+')[-1]

    # Case 4: Your custom dataset format
    # Example:
    # Food.6d2mt3i5.ingestion-8646c44474-hkn7f.s1.wav
    if '.ingestion-' in fn:
        return fn.split('.ingestion-')[1].split('.')[0]

    # Case 5: Fallback — use filename prefix
    # This guarantees NO crash
    return fn.split('.')[0]
my_model.dataset = MyDataset()




#################################################
# Audio Classifier Settings
#
# These are additional parameters to include in
# the generated .tflite model file.
# The settings are used by the ble_audio_classifier app
# NOTE: Corresponding command-line options will override these values.

# Controls the smoothing.
# Drop all inference results that are older than <now> minus window_duration
# Longer durations (in milliseconds) will give a higher confidence that the results are correct, but may miss some commands
my_model.model_parameters['average_window_duration_ms'] = 300
# Define a specific detection threshold for each class
#my_model.model_parameters['detection_threshold'] = 235
my_model.model_parameters['detection_threshold_list'] = list(map(lambda x: int(x*255), [.85, .85, .85, .92, .92, .92, .85, .85, 1.0]))
# Amount of milliseconds to wait after a keyword is detected before detecting the SAME keyword again
# A different keyword may be detected immediately after
my_model.model_parameters['suppression_ms'] = 700
# The minimum number of inference results to average when calculating the detection value
my_model.model_parameters['minimum_count'] = 2
# Set the volume gain scaler (i.e. amplitude) to apply to the microphone data. If 0 or omitted, no scaler is applied
my_model.model_parameters['volume_gain'] = 0.0
# This the amount of time in milliseconds between audio processing loops
# Since we're using the audio detection block, we want this to be as short as possible
my_model.model_parameters['latency_ms'] = 10
# Enable verbose inference results
my_model.model_parameters['verbose_model_output_logs'] = False



##########################################################################################
# The following allows for running this model training script directly, e.g.:
# python keyword_spotting_pacman_v3.py
#
# Note that this has the same functionality as:
# mltk train keyword_spotting_pacman_v3
#
if __name__ == '__main__':
    from mltk import cli

    # Setup the CLI logger
    cli.get_logger(verbose=True)


    # If this is true then this will do a "dry run" of the model testing
    # If this is false, then the model will be fully trained
    test_mode_enabled = False

    # Train the model
    # This does the same as issuing the command:  mltk train keyword_spotting_pacman_v3-test --clean)
    train_results = mltk_core.train_model(my_model, clean=True, test=test_mode_enabled)
    print(train_results)

    # Evaluate the model against the quantized .h5 (i.e. float32) model
    # This does the same as issuing the command: mltk evaluate keyword_spotting_pacman_v3-test
    tflite_eval_results = mltk_core.evaluate_model(my_model, verbose=True, test=test_mode_enabled)
    print(tflite_eval_results)

    # Profile the model in the simulator
    # This does the same as issuing the command: mltk profile keyword_spotting_pacman_v3-test
    profiling_results = mltk_core.profile_model(my_model, test=test_mode_enabled)
    print(profiling_results)
