'''
This script demonstrates how to add custom parameters/settings to a .tflite model.
These custom parameters can then be accessed by the embedded device using:

- Generated #defines
- Runtime API TfliteModelParameters
'''

# System includes
import sys 
import os 
import tempfile
import shutil


# Required: Import the TfliteModelParameters class
from npu_toolkit.tflite import TfliteModelParameters



def main():
    # Get the path to the .tflite file we want to add parameters to
    tflite_model_path = get_tflite_model_path()
 
    # Create the model parameters instance
    model_params = TfliteModelParameters.load(tflite_model_path)

    # Remove any old parameters
    model_params.clear()

    # Add a bool
    model_params['normalize_samples'] = True
    # Add an integer
    # NOTE: By default, the library will pack the value into the smallest valid datatype
    #       Use the put() API to force a specifc data type
    model_params.put('sample_rate', 8000, dtype='int32')
    # Add a float
    model_params['period_ms'] = 0.624
    # Add a string
    model_params['msg'] = 'This is neat!'
    # Add a list of strings
    model_params['class_labels'] = ['left', 'right', 'up', 'download', 'unknown']
    # Add a list of integers
    model_params['int_list'] = [1, 2, 3, 4, 5]
    # Add a list of integers
    model_params['float_list'] = [1.1, 2.2, 3.3, 4.4]
    # Add binary data
    model_params['blob'] = b'\xE2\x82\xAC'

    # Print a summary of the parameters
    print('Model parameters:')
    print(model_params)


    # Save the model parameters to the .tflite model file
    # When the .tflite is compiled, the parameters 
    # will be accessible to the embedded device
    print(f'Updating {tflite_model_path}')
    model_params.save(save_model=True)



def get_tflite_model_path():
    '''Check if a .tflite model path was provided on the command-line
    otherwise use the provded image_classification.tflite model
    '''

    if len(sys.argv) > 1:
        return sys.argv[1]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tflite_path = f'{script_dir}/../models/image_classification.tflite'

    temp_tflite_path = tempfile.NamedTemporaryFile(
        'wb', 
        prefix='my_model', 
        suffix='.tflite', 
        delete=False
    ).name

    # Copy the model into a new .tflite so that we can modify it without changing the original
    shutil.copy(tflite_path, temp_tflite_path)

    return temp_tflite_path


if __name__ == '__main__':
    main()