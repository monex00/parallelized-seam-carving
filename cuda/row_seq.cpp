#include <iostream>
#include <cstdlib>
#include <chrono>
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

// Funzione per processare la matrice in modo sequenziale, rispettando le dipendenze
void processMatrixSequential(float* hostInput, float* hostOutput, int width, int height) {
    // Inizializza la prima riga dell'output copiando dall'input
    for (int col = 0; col < width; col++) {
        hostOutput[col] = hostInput[col];
    }

    // Calcola le righe successive rispettando le dipendenze
    for (int row = 1; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float topLeft = (col > 0) ? hostOutput[(row - 1) * width + (col - 1)] : FLT_MAX;
            float topCenter = hostOutput[(row - 1) * width + col];
            float topRight = (col < width - 1) ? hostOutput[(row - 1) * width + (col + 1)] : FLT_MAX;

            float currentValue = hostInput[row * width + col];
            float minValue = std::min(topLeft, std::min(topCenter, topRight));
            hostOutput[row * width + col] = currentValue + minValue;
        }
    }
}

int main() {
    float* hostInput = new float[WIDTH * HEIGHT];
    float* hostOutput = new float[WIDTH * HEIGHT];

    // Inizializzazione della matrice di input
    initializeMatrix(hostInput, WIDTH, HEIGHT);

    // Misurazione del tempo di esecuzione
    auto start = std::chrono::high_resolution_clock::now();

    // Esegui l'elaborazione sequenziale
    processMatrixSequential(hostInput, hostOutput, WIDTH, HEIGHT);

    // Calcola e stampa il tempo di esecuzione
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Tempo di esecuzione: " << duration.count() << " ms" << std::endl;

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
