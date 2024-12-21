### Cuda Matrix Multiplication

#### Setup
```
mkdir build
cd build
cmake ..
make
./mat_mul <row1> <col> <col2> # Multiply (row1, col) with (col, col2)
./mat_mul_tiled <row1> <col> <col2> # Multiplication using shared memory
```

#### Profile
```
nsys profile --stats=true ./matrix_mul <row1> <col> <col2>
nsys profile --stats=true ./matrix_mul_tiled <row1> <col> <col2>
```