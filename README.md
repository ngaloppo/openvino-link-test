# openvino_link_test

Reproducer for [OpenVINO issue #6454](https://github.com/openvinotoolkit/openvino/issues/6454).

Replace `<YOUR_CXX_COMPILER_EXE>` below with the compiler you'd like to test. Currently `g++` works, but any flavor of `clang++` fails (tested with `clang++-8` and `clang++-12`).

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DInferenceEngine_DIR=<OPENVINO_DIR>/deployment_tools/inference_engine/share -DCMAKE_CXX_COMPILER=<YOUR_CXX_COMPILER_EXE>
make -j$(nproc --all)
```

