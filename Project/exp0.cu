#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <algorithm>
#include <ctime>
using namespace std;

string name = "NAIVE";

double interval(clock_t *p) {
    clock_t t = clock();
    double result = double(t - *p) / CLOCKS_PER_SEC * 1000;
    *p = t;
    return result;
}

int main(int argc, char *argv[]) {
    assert(argc == 6);

    int N = atoi(argv[1]);
    int dim = atoi(argv[2]);
    int K = atoi(argv[3]);
    double *X = new double[N * dim];
    pair<double, int> *D = new pair<double, int>[N * N];
    string title;
    clock_t p;
    
    cout << name << endl;
    cout << "N=" << N << " dim=" << dim << " K=" << K << " " << argv[4] << endl;

    p = clock();

    ifstream fin;
    fin.open(argv[4]);

    fin >> title;

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < dim; j++) fin >> X[i * dim + j];
    }

    fin.close();
    
    cout << "step 0:Data Import::" << interval(&p) << "ms" << endl;

    for(int i = 0; i < N; i++) {
        // cout << "row: " << i << endl;

        for(int j = i; j < N; j++) {
            D[i * N + j].first = 0.0;
            if (i == j) continue;
            else {
                for(int k = 0; k < dim; k++) {
                    double tmp = X[i * dim + k] - X[j * dim + k];
                    D[i * N + j].first += tmp * tmp;
                }
                
                D[i * N + j].first = sqrt(D[i * N + j].first);
                D[i * N + j].second = j;
                
                D[j * N + i].first = D[i * N + j].first;
                D[j * N + i].second = i;
            }
        }
    }

    cout << "step 1:Distance::" << interval(&p) << "ms" << endl;

    for(int i = 0; i < N; i++) sort(D + i * N, D + (i + 1) * N);

    cout << "step 2:Sort::" << interval(&p) << "ms" << endl;

    ofstream fout;
    fout.open(string(argv[5]));

    for(int i = 0; i < N; i++) {
        // omit first one -> i-i pair
        for(int j = 1; j < K + 1; j++) fout << D[i * N + j].second << " ";

        fout << endl;
    }

    fout.close();

    cout << "step 3:Export Result::" << interval(&p) << "ms" << endl;

    delete[] X;
    delete[] D;
}