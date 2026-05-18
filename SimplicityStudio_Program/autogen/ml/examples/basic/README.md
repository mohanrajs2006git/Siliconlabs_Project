# Basic Example

This provides a basic example of how to compile a `.tflite` model and run it on an embedded device
using the [TfliteMicroModel](npu_toolkit::TfliteMicroModel) API.


To run this example, first compile your `.tflite` model.

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

Once copied, build your application with the copied files and corresponding `main.cc` which will invoke a single inference
and print to the console using `printf()`.
