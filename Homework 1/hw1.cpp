#include <iostream>
#include <string>
#include <fstream>
#include <stdlib.h>
using namespace std;

int main() {
    string path = "D:\\C++\\Task1\\1.txt";
    ifstream file_in;
    file_in.open(path);
    if (!file_in.is_open()) {
        cout << "Error " << endl;
    }
    else {
        //cout << "Success" << endl;
        string num;
        int size = 2;
        int xn = 0;
        int yn = 0;
        double max_r = 0; // Для правого
        double max_l = 0; // Для левого
        int size_d = 2;
        int *num_m_max_left = new int[size];
        int *num_m_max_right = new int[size];
        int first = 0; // счетчик, чтобы получить 1е значение вектора
        while (!file_in.eof()) {
            int *num_m = new int[size];
            if (first == 0) {
                num = "";
                getline(file_in, num);
                // проверка нулевого символа на минус
                if (num[0] == '-') {
                    xn = (-1) * (int) (num[1] - '0'); // занесение x
                } else {
                    xn = (int) (num[0] - '0'); // занесение x
                }
                // проверка 2го символа символа на минус
                if (num[num.size() - 2] == '-') {
                    yn = (-1) * (int) (num[num.size() - 1] - '0'); // занесение y
                } else {
                    yn = (int) (num[num.size() - 1] - '0'); // занесение y
                }
                first++;
            }
            else {
                num = "";
                getline(file_in, num);
                // проверка нулевого символа на минус
                if (num[0] == '-') {
                    num_m[0] = (-1) * (int)(num[1] - '0'); // занесение x
                }
                else {
                    num_m[0] = (int)(num[0] - '0'); // занесение x
                }
                // проверка 2го символа символа на минус
                if (num[num.size() - 2] == '-') {
                    num_m[1] = (-1) * (int)(num[num.size() - 1] - '0'); // занесение y
                }
                else {
                    num_m[1] = (int)(num[num.size() - 1] - '0'); // занесение y
                }
                // Проверка на максимальный элемент + вычисление длины от вектора до точки
                double* D_LEFT = new double[size_d];
                double* D_RIGHT = new double[size_d];
                if (xn * (*(num_m + 1)) - yn * (*(num_m ) > 0.0) ) {        // слева
                    *(D_LEFT) = abs(xn * (*(num_m + 1)) - yn * (*(num_m))) / sqrt(xn ^ 2 + yn ^ 2);
                }
                else if (xn * (*(num_m + 1)) - yn * (*(num_m )) == 0.0) {
                    *(D_RIGHT) = abs(xn * (*(num_m + 1)) - yn * (*(num_m ))) / sqrt(xn ^ 2 + yn ^ 2);
                }
                else {          // справа
                    *(D_RIGHT)  = abs(xn * (*(num_m + 1)) - yn * (*(num_m ))) / sqrt(xn ^ 2 + yn ^ 2);
                }
                if (*(D_RIGHT) > max_r) {
                    max_r = *(D_RIGHT);
                    *(num_m_max_left) = *(num_m); // x_max_right
                    *(num_m_max_left+1) = *(num_m +1); // y_max_right
                }
                if (*(D_LEFT) > max_l) {
                    max_l = *(D_LEFT);
                    *(num_m_max_right) = *(num_m); // x_max_left
                    *(num_m_max_right+1) = *(num_m +1); // y_max_left
                }
                delete[] D_LEFT;
                delete[] D_RIGHT;
            }
            delete[] num_m;
        }
        cout << "Leftmost: " << *(num_m_max_left) << " " << *(num_m_max_left + 1) << endl;
        cout << "Rightmost: " << *(num_m_max_right) << " " << *(num_m_max_right + 1) << endl;
        delete[] num_m_max_left;
        delete[] num_m_max_right;
    }
    file_in.close();
    return 0;
}
