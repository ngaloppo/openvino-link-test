# ocl_layer

1. Download Intel OpenVINO

2. Build a project
```
source /opt/intel/openvino_2021/bin/setupvars.sh

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc --all)
```

3. Run
```
./ocl_layer
```
