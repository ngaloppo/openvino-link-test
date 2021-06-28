# sycl_link_test

1. Create a file `config.txt`:

```
tbb=exclude
```

2. Build a project
```
source <ONE_API_ROOT>/setvars.sh --config=config.txt

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DInferenceEngine_DIR=<OPENVINO_DIR>/deployment_tools/inference_engine/share
make -j$(nproc --all)
```

