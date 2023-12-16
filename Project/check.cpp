#include <fstream>
#include <iostream>
using namespace std;
int main(int argc, char *argv[]) {
    int N = atoi(argv[3]);
    int K = atoi(argv[4]);
    ifstream f1, f2;
    f1.open(string(argv[1]));
    f2.open(string(argv[2]));
    for(int i = 0; i < N * K; i++) {
        int t1, t2;
        f1 >> t1;
        f2 >> t2;
        if(t1 != t2) {
            cout << "incorrect" << endl;
            f1.close();
            f2.close();
            exit(0);
        } 
    }
    cout << "correct" << endl;
    f1.close();
    f2.close();
}