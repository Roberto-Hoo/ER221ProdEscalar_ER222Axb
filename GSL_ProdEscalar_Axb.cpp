#include <omp.h>
#include <stdio.h>
#include <ctime>
#include <iostream>

// GSL suport
#include <gsl/gsl_vector.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_matrix.h>
 #include <gsl/gsl_blas.h>



using namespace std;

#define NumThreads 8
// Variáveis Globais(shared)
char caracter;
int n = 1000000;
int M1[NumThreads], M2[NumThreads], M3[NumThreads];
// matrizes e vetores
gsl_vector *a = gsl_vector_alloc(n);
gsl_vector *b = gsl_vector_alloc(n);

int n1 = 10;
// vetores
gsl_matrix *A = gsl_matrix_alloc(n1, n1);
gsl_vector *x = gsl_vector_alloc(n1);
gsl_vector *y = gsl_vector_alloc(n1);


// gerador randômico
gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);


int main(int argc, char *argv[]) {

    /*
     * Exercícios Resolvidos
     * ER 2.2.1.
     * Escreva um código MP para computar o produto escalar entre dois vetores de
     * pontos flutuantes randômicos.

     Solução: Aqui, vamos usar o suporte a vetores e números randômicos do pacote de computação cientifica GSL.
     A solução é dada no código a seguir. */

    omp_set_num_threads(NumThreads);

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
    for (int i = 0; i < n; i++) {
        dot += gsl_vector_get(a, i) * gsl_vector_get(b, i);
        int id = omp_get_thread_num();
        M3[id] = i;
    }

    printf("\n %f", dot);

    for (int i = 0; i < NumThreads; i++)
        printf("\n M1[%d]=%d    M2[%d]=%d  M2[%d]=%d", i, M1[i], i, M2[i], i, M3[i]);

    gsl_vector_free(a);
    gsl_vector_free(b);

    /*
     * ER 2.2.2.
     * Faça um código MP para computar a multiplicação de uma matriz por um vetor de
     * elementos (pontos flutuantes randômicos). Utilize o construtor omp sections
     * para distribuir a computação entre somente dois threads.

     Solução:
     Vamos usar o suporte a matrizes, vetores, BLAS e números randômicos do pacote de
     computação científica GSL. A solução é dada no código a seguir.
     */


    // gerador randômico
    gsl_rng_set(rng, time(NULL));

    // inicialização
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n1; j++) {
            gsl_matrix_set(A, i, j, gsl_rng_uniform(rng));
        }
        gsl_vector_set(x, i, gsl_rng_uniform(rng));
    }

    //gsl_blas_dgemv(CblasNoTrans, 1.0, a, x, 0.0, y);
    // y = A*x
#pragma omp parallel sections
    {
#pragma omp section
        {
            gsl_matrix_const_view as1 = gsl_matrix_const_submatrix(A, 0, 0, n1 / 2, n1);
            gsl_vector_view ys1  = gsl_vector_subvector(y, 0, n1 / 2);
            gsl_blas_dgemv(CblasNoTrans,1.0, &as1.matrix, x, 0.0, &ys1.vector);
        }

#pragma omp section
        {
            gsl_matrix_const_view as2 = gsl_matrix_const_submatrix(A, n1 / 2, 0,(n1 - n1 / 2), n1);
            gsl_vector_view ys2 = gsl_vector_subvector(y, n1 / 2, (n1 - n1 / 2));
            gsl_blas_dgemv(CblasNoTrans,1.0, &as2.matrix, x,0.0, &ys2.vector);
        }
    }

    for (int i=0; i<n1; i++)
    printf("\n %f", gsl_vector_get(y,i));

    gsl_matrix_free(A);
    gsl_vector_free(x);
    gsl_vector_free(y);
    gsl_rng_free(rng);

    cout << "\n\n Tecle uma tecla e apos Enter para finalizar...\n";
    cin >> caracter;

    return 0;
}

