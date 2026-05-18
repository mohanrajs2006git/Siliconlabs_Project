# Multi-Model Example

This demonstrates how to compiled and execute multiple models at the same time
using the [TfliteMicroModel](npu_toolkit::TfliteMicroModel) API.

Example compiler commands are as follows:

```shell
./npu_toolkit_compiler --platform si91x --accelerator mvpv1 --weights-paging \
     -x codegen.profiler_enabled=1 \
     -x codegen.model_header.namespace=image_classification \
     -x codegen.dst_dir=multi_model_sources \
     image_classification.tflite
```

```shell
./npu_toolkit_compiler --platform si91x --accelerator mvpv1 --weights-paging \
    -x codegen.profiler_enabled=1 \
    -x codegen.model_header.namespace=keyword_spotting \
    -x codegen.dst_dir=multi_model_sources \
    -x codegen.clean_dst_dir=0 \
   keyword_spotting.tflite
``` 

Where:
- `--platform si91x` - Compile for a si91x target (e.g. BRD2605 development board)
- `--accelerator mvpv1` - Accelerate for the MVPv1
- `--weights-paging` - Enable weights-paging
- `-x codegen.profiler_enabled=1` - This tells the compiler code generation to also include the profiling utility
- `-x codegen.model_header.namespace=image_classification` - This specifies the C++ namespace used when generatng the image_classification C++ header.
- `-x codegen.dst_dir=multi_model_sources` - Specifies the directory where the source code will be generated
- `-x codegen.clean_dst_dir=0` - Do NOT clean the destination directory before generation


The above command will generate the source files in the directory "multi_model_sources".
Copy the contents of this directory to
your project and ensure it is available on the C++ "includes" path.

After that, this app will instantiate both models
and invoke a single inference for both, printing to the profiling results to the console