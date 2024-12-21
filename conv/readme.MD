### Cuda Matrix Multiplication

#### Setup
```
mkdir build
cd build
cmake ..
make
./conv_1d_n <vector-size> <mask-size> # Convlolve vector of size <vector-size> with mask of size <mask-size>
./conv_1d_s <vector-size> <mask-size> # Convolve using shared memory
```

#### Profile
```
nsys profile --stats=true ./conv_1d_n <vector-size> <mask-size>
nsys profile --stats=true ./conv_1d_s <vector-size> <mask-size>
```