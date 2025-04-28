#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// CUDA Kernel for vector addition
_global_ void vectorAdd(float *A, float *B, float *C, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int n;
    cout << "Enter number of elements in vectors: ";
    cin >> n;

    size_t size = n * sizeof(float);

    // Allocate host memory
    float h_A = (float)malloc(size);
    float h_B = (float)malloc(size);
    float h_C = (float)malloc(size);

    // Take input from user
    cout << "Enter elements of Vector A:\n";
    for (int i = 0; i < n; i++) {
        cin >> h_A[i];
    }

    cout << "Enter elements of Vector B:\n";
    for (int i = 0; i < n; i++) {
        cin >> h_B[i];
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy vectors from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Output result
    cout << "Result Vector (A+B):\n";
    for (int i = 0; i < n; i++) {
        cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << endl;
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}






//multiplication..................
[7:00 pm, 27/4/2025] Shruti Gaikwad: #include <iostream>
#include <cuda_runtime.h>
using namespace std;

// CUDA Kernel for vector addition
_global_ void vectorAdd(float *A, float *B, float *C, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int n;
    cout << "Enter number of elements in vectors: ";
    cin >> n;

    size_t size = n * sizeof(float);

    // Allocate host memory
    float h_A = (float)malloc(size);
    float h_B = (float)malloc(size);
    float h_C = (float)malloc(size);

    // Take input from user
    cout << "Enter elements of Vector A:\n";
    for (int i = 0; i < n; i++) {
        cin >> h_A[i];
    }

    cout << "Enter elements of Vector B:\n";
    for (int i = 0; i < n; i++) {
        cin >> h_B[i];
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy vectors from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Output result
    cout << "Result Vector (A+B):\n";
    for (int i = 0; i < n; i++) {
        cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << endl;
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
[7:00 pm, 27/4/2025] Shruti Gaikwad: #include <iostream>
#include <cuda_runtime.h>
using namespace std;

// CUDA Kernel
_global_ void matrixMul(int *A, int *B, int *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    int N;
    cout << "Enter size of square matrices (N x N): ";
    cin >> N;

    int size = N * N * sizeof(int);

    // Host memory
    int h_A = (int)malloc(size);
    int h_B = (int)malloc(size);
    int h_C = (int)malloc(size);

    // Take input for matrices
    cout << "Enter elements of Matrix A:\n";
    for (int i = 0; i < N*N; i++) {
        cin >> h_A[i];
    }

    cout << "Enter elements of Matrix B:\n";
    for (int i = 0; i < N*N; i++) {
        cin >> h_B[i];
    }

    // Device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print result matrix
    cout << "Result Matrix (A * B):\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << h_C[i*N + j] << " ";
        }
        cout << endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
