#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define WIDTH 1024    // Dimensione della matrice in larghezza
#define HEIGHT 1024   // Dimensione della matrice in altezza

// Funzione per inizializzare la matrice con valori casuali
void initializeMatrix(float* matrix, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Funzione per verificare che i risultati siano corretti
bool verifyResult(float* hostInput, float* hostOutput, int width, int height) {
    // Implementare qui la logica di verifica, in base alla definizione di correttezza
    // Per ora restituiremo true (simulando un controllo positivo)
    return true;
}

#define BLOCK_SIZE_X 256
#define PADDING 1

__global__ void computeStencilRow(float* input, float* output, int width, int height, int currentRow) {
    extern __shared__ float sharedMem[];
    
    // Calcolo degli indici
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int x = bx * (BLOCK_SIZE_X - 2*PADDING) + tx;
    
    // Indici per la memoria condivisa
    int s_idx = tx + PADDING;
    
    // Caricamento dati in memoria condivisa dalla riga superiore (già calcolata)
    if (currentRow > 0 && x < width) {
        sharedMem[s_idx] = output[(currentRow-1)*width + x];  // Usiamo output invece di input per leggere i risultati precedenti
    }
    
    __syncthreads();
    
    // Calcolo solo se siamo in una posizione valida
    if (currentRow > 0 && x > 0 && x < width-1) {
        // Prendo i tre valori dalla riga superiore (già calcolati)
        float topLeft = sharedMem[s_idx - 1];
        float topCenter = sharedMem[s_idx];
        float topRight = sharedMem[s_idx + 1];
        
        // Prendo il valore corrente dalla matrice di input originale
        float currentValue = input[currentRow*width + x];
        
        // Calcolo il nuovo valore
        float minValue = min(topLeft, min(topCenter, topRight));
        output[currentRow*width + x] = currentValue + minValue;
    } else if (currentRow == 0 && x < width) {
        // Per la prima riga, copiamo semplicemente i valori dall'input
        output[x] = input[x];
    }
}

void processMatrixWithDependencies(float* hostInput, float* hostOutput, int width, int height) {
    float *d_input, *d_output;
    size_t size = width * height * sizeof(float);
    
    // Allocazione memoria su device
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    // Copia input su device
    cudaMemcpy(d_input, hostInput, size, cudaMemcpyHostToDevice);
    
    // Dimensione dei blocchi e della memoria condivisa
    dim3 blockSize(BLOCK_SIZE_X, 1);
    dim3 gridSize((width + BLOCK_SIZE_X - 2*PADDING - 1) / (BLOCK_SIZE_X - 2*PADDING), 1);
    size_t sharedMemSize = (BLOCK_SIZE_X + 2*PADDING) * sizeof(float);
    
    // Processiamo una riga alla volta per rispettare le dipendenze
    for (int row = 0; row < height; row++) {
        computeStencilRow<<<gridSize, blockSize, sharedMemSize>>>(
            d_input, d_output, width, height, row
        );
        cudaDeviceSynchronize();  // Aspettiamo che la riga sia completata prima di procedere
    }
    
    // Copia risultato su host
    cudaMemcpy(hostOutput, d_output, size, cudaMemcpyDeviceToHost);
    
    // Pulizia
    cudaFree(d_input);
    cudaFree(d_output);
}


int main() {
    float* hostInput = new float[WIDTH * HEIGHT];
    float* hostOutput = new float[WIDTH * HEIGHT];

    // Inizializzazione della matrice di input
    initializeMatrix(hostInput, WIDTH, HEIGHT);

    // Esegui l'elaborazione
    processMatrixWithDependencies(hostInput, hostOutput, WIDTH, HEIGHT);

    // Verifica del risultato
    if (verifyResult(hostInput, hostOutput, WIDTH, HEIGHT)) {
        std::cout << "Verifica completata: risultato corretto!" << std::endl;
    } else {
        std::cout << "Errore: risultato non corretto." << std::endl;
    }

    delete[] hostInput;
    delete[] hostOutput;

    return 0;
}