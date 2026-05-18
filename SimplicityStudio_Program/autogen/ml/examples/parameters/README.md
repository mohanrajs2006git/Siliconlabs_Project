# Model Parameters Example

This demonstrates how to add custom parameters to
a .tflite model via [Python](npu_toolkit.tflite.TfliteModelParameters) and retrieve them on
an [embedded device](npu_toolkit::TfliteModelParameters).

The associated Python script: [update_model_parameters.py](/src/cpp/apps/tflite_micro_model_examples/parameters/update_model_parameters.py)
provides an example of how to add custom parameters to a (uncompiled) .tflite
model file.
 
Once the parameters are added, the model can be compiled.


An example compiler command is as follows:

```shell
 ./npu_toolkit_compiler --platform si91x --accelerator mvpv1 --weights-paging  image_classification.tflite
 ```
 
Where:
- `--platform si91x` - Compile for a si91x target (e.g. BRD2605 development board)
- `--accelerator mvpv1` - Accelerate for the MVPv1
- `--weights-paging` - Enable weights-paging
- `image_classification.tflite` - The name/file path to your quantized .tflite model

The above command will generate a model archive file: `image_classification.zip`
which contains the compiled .tflite model file: `image_classification.mvpv1.tflite`.
It also contains a directory: __codegen__ which contains all of the generated
code to execute the compiled model. Copy the contents of this directory to
your project and ensure it is available on the C++ "includes" path.


The parameters can then be accessed with two methods:

## Compile-Time #defines
 
In the codegen direction is a header file: `<model name>_generator.parameters.h`
 
This file contains #defines corresponding to each custom parameter.
 

## Runtime API
 
The [TfliteModelParameters](npu_toolkit::TfliteModelParameters) API can also be used.
This allows for retrieving the parameters at runtime.


This example demonstrates how to use both methods.