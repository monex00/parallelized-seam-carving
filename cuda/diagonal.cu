#include <cuda_runtime.h>

#define BLOCK_SIZE_X 256
#define TILE_DIM 16  // Dimensione del tile per l'elaborazione diagonale

__global__ void computeWavefrontStencil(float* input, float* output, 
                                      int width, int height, int diagonal) {
    extern __shared__ float sharedMem[];
    
    // Calcolo degli indici basati sulla diagonale
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    
    // La coordinata x parte dall'inizio della diagonale corrente
    int x = bx * TILE_DIM + tx;
    // La coordinata y viene calcolata in modo che x + y = diagonal
    int y = diagonal - x;
    
    // Controlliamo se siamo all'interno dei limiti della matrice
    if (x >= 0 && x < width && y >= 1 && y < height) {
        // Leggiamo i tre valori superiori direttamente dalla memoria globale
        // dato che sono già stati calcolati
        float topLeft = (x > 0) ? output[(y-1)*width + (x-1)] : FLT_MAX;
        float topCenter = output[(y-1)*width + x];
        float topRight = (x < width-1) ? output[(y-1)*width + (x+1)] : FLT_MAX;
        
        // Leggiamo il valore corrente
        float currentValue = input[y*width + x];
        
        // Calcolo del nuovo valore
        float minValue = min(topLeft, min(topCenter, topRight));
        output[y*width + x] = currentValue + minValue;
    }
}

void processMatrixWavefront(float* hostInput, float* hostOutput, int width, int height) {
    float *d_input, *d_output;
    size_t size = width * height * sizeof(float);
    
    // Allocazione memoria su device
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    // Copia input su device
    cudaMemcpy(d_input, hostInput, size, cudaMemcpyHostToDevice);
    
    // Copia la prima riga senza modifiche
    cudaMemcpy(d_output, d_input, width * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Numero di blocchi per diagonale
    int numBlocks = (width + TILE_DIM - 1) / TILE_DIM;
    
    // Processiamo le diagonali
    // Il numero di diagonali è width + height - 1
    for (int diagonal = 1; diagonal < width + height - 1; diagonal++) {
        // Calcoliamo quanti elementi sono nella diagonale corrente
        int elementsInDiagonal = min(min(diagonal + 1, width), height);
        int blocksNeeded = (elementsInDiagonal + TILE_DIM - 1) / TILE_DIM;
        
        computeWavefrontStencil<<<blocksNeeded, TILE_DIM>>>(
            d_input, d_output, width, height, diagonal
        );
        cudaDeviceSynchronize();
    }
    
    // Copia risultato su host
    cudaMemcpy(hostOutput, d_output, size, cudaMemcpyDeviceToHost);
    
    // Pulizia
    cudaFree(d_input);
    cudaFree(d_output);
}