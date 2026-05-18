# Profiling Example

This builds on the basic example by enabling the profilng utility that comes with the [TfliteMicroModel](npu_toolkit::TfliteMicroModel) API.

To run this example, first compile your `.tflite` model.

An example compiler command is as follows:

```shell
 ./npu_toolkit_compiler --platform si91x --accelerator mvpv1 --weights-paging -x codegen.profiler_enabled=1 image_classification.tflite
 ```
 
Where:
- `--platform si91x` - Compile for a si91x target (e.g. BRD2605 development board)
- `--accelerator mvpv1` - Accelerate for the MVPv1
- `--weights-paging` - Enable weights-paging
- `-x codegen.profiler_enabled=1` - Enable the profiling utility with the generated code
- `image_classification.tflite` - The name/file path to your quantized .tflite model

The above command will generate a model archive file: `image_classification.zip`
which contains the compiled .tflite model file: `image_classification.mvpv1.tflite`.
It also contains a directory: __codegen__ which contains all of the generated
code to execute the compiled model. Copy the contents of this directory to
your project and ensure it is available on the C++ "includes" path.

Once copied, build your application with the copied files and corresponding `main.cc` which will invoke a single inference
and print the profiling results to the console.