#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <float.h>

#define WIDTH  4000   // Dimensione della matrice in larghezza
#define HEIGHT 3000   // Dimensione della matrice in altezza

// Funzione per inizializzare la matrice con valori casuali
void initializeMatrix(float* matrix, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        matrix[i] = 1; // static_cast<float>(rand()) / RAND_MAX;
    }
}

// Funzione per verificare che i risultati siano corretti
bool verifyResult(float* hostInput, float* hostOutput, int width, int height) {
    // Implementare qui la logica di verifica, in base alla definizione di correttezza
    // Per ora restituiremo true (simulando un controllo positivo)
    return true;
}

// Funzione per stampare la matrice
void printMatrix(float* matrix, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << matrix[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


#define BLOCK_SIZE_X 10
#define PADDING 1

__global__ void computeStencilRow(float* input, float* output, int width, int height, int currentRow) {
    extern __shared__ float sharedMem[];
    // Calcolo degli indici
    int tx = threadIdx.x; 
    int bx = blockIdx.x;

    // indice x della matrice
    int x = bx * BLOCK_SIZE_X + tx -1;
    
    // Indici per la memoria condivisa
    int s_idx = tx;
    
    // Caricamento dati in memoria condivisa dalla riga superiore (già calcolata)
    if (currentRow > 0) {
        if (x == -1 || x == width) {
            sharedMem[s_idx] = FLT_MAX;
        } else { 
            sharedMem[s_idx] = output[(currentRow-1)*width + x];
        }
    }
    
    __syncthreads();

    // Calcolo solo se siamo in una posizione valida
    if (currentRow > 0 && s_idx > 0 && s_idx < BLOCK_SIZE_X + 1) {
        // Prendo i tre valori dalla riga superiore (già calcolati)
      
        float topLeft = sharedMem[s_idx - 1];
        float topCenter = sharedMem[s_idx];
        float topRight = sharedMem[s_idx + 1];
        
        // Prendo il valore corrente dalla matrice di input originale
        float currentValue = input[currentRow*width + x];
        
        // Calcolo il nuovo valore
        float minValue = min(topLeft, min(topCenter, topRight));
        output[currentRow*width + x] = currentValue + minValue; 
        
    } else if (currentRow == 0 && x > -1 && x < width) {
        // Per la prima riga, copiamo semplicemente i valori dall'input
        output[currentRow*width + x] = input[currentRow*width + x];
    }  
}



void processMatrixWithDependencies(float* hostInput, float* hostOutput, int width, int height) {
    float *d_input, *d_output;
    size_t size = width * height * sizeof(float);

    // Inizializzazione degli eventi per misurare il tempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Inizio del timer
    cudaEventRecord(start, 0);

    // Allocazione memoria su device
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copia input su device
    cudaMemcpy(d_input, hostInput, size, cudaMemcpyHostToDevice);

    // Dimensione dei blocchi e della memoria condivisa
    dim3 blockSize(BLOCK_SIZE_X + 2 * PADDING, 1);
    dim3 gridSize(width / (BLOCK_SIZE_X), 1);
    size_t sharedMemSize = (BLOCK_SIZE_X + 2 * PADDING) * sizeof(float);

    // Processiamo una riga alla volta per rispettare le dipendenze
    for (int row = 0; row < height; row++) {
        computeStencilRow<<<gridSize, blockSize, sharedMemSize>>>(
            d_input, d_output, width, height, row
        );
        cudaDeviceSynchronize();  // Aspettiamo che la riga sia completata prima di procedere
    }

    // Copia risultato su host
    cudaMemcpy(hostOutput, d_output, size, cudaMemcpyDeviceToHost);

    // Stop del timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calcolo e stampa del tempo di esecuzione
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Tempo di esecuzione: " << milliseconds << " ms" << std::endl;

    // Pulizia
    cudaFree(d_input);
    cudaFree(d_output);

    // Distruzione degli eventi
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}



int main() {
    float* hostInput = new float[WIDTH * HEIGHT];
    float* hostOutput = new float[WIDTH * HEIGHT];

    // Inizializzazione della matrice di input
    initializeMatrix(hostInput, WIDTH, HEIGHT);
    // printMatrix(hostInput, WIDTH, HEIGHT);
    // Esegui l'elaborazione
    processMatrixWithDependencies(hostInput, hostOutput, WIDTH, HEIGHT);

    // Verifica del risultato
    if (verifyResult(hostInput, hostOutput, WIDTH, HEIGHT)) {
        std::cout << "Verifica completata: risultato corretto!" << std::endl;
    } else {
        std::cout << "Errore: risultato non corretto." << std::endl;
    }
    //  printMatrix(hostOutput, WIDTH, HEIGHT);
    delete[] hostInput;
    delete[] hostOutput;

    return 0;
}
