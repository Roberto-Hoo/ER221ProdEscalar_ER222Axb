#include <omp.h>
#include <stdio.h>
#include <ctime>
#include <iostream>

// GSL vector suport
#include <gsl/gsl_vector.h>
#include <gsl/gsl_rng.h>


using namespace std;

#define NumThreads 8
// Variáveis Globais(shared)
char caracter;
int n = 1000000;
int M1[NumThreads];
int M2[NumThreads];


int main(int argc, char *argv[]) {

    omp_set_num_threads(NumThreads);

    // vetores
    gsl_vector *a = gsl_vector_alloc(n);
    gsl_vector *b = gsl_vector_alloc(n);

    // gerador randÃ´mico
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
    gsl_rng_set(rng, time(NULL));

    // inicializa o vetor
    for (int i = 0; i < NumThreads; i++)
        M1[i] = 0;

    // inicializa os vetores
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        gsl_vector_set(a, i, gsl_rng_uniform(rng));
        gsl_vector_set(b, i, gsl_rng_uniform(rng));
        int id = omp_get_thread_num();
        M1[id] = M1[id] + 1;
        M2[id] = i;
    }

    // produto escalar
    double dot = 0;
#pragma omp parallel for reduction(+: dot)
    for (int i = 0; i < n; i++)
        dot += gsl_vector_get(a, i) * gsl_vector_get(b, i);

    printf("\n %f", dot);

    for (int i = 0; i < NumThreads; i++)
        printf("\n M1[%d]=%d    M2[%d]=%d", i, M1[i], i, M2[i]);

    gsl_vector_free(a);
    gsl_vector_free(b);
    gsl_rng_free(rng);

    cout << "\n\n Tecle uma tecla e apos Enter para finalizar...\n";
    cin >> caracter;

    return 0;
}

