#include <iostream>
#include <string>
#include <fstream>
#include <cmath>

using namespace std;


float get_x(ifstream &input_file) // функция получения координаты x
{
    string x_cord;
    float x;
    input_file >> x_cord;
    x = stof(x_cord);
    return x;
}

float get_y(ifstream &input_file)  // функция получения координаты y
{
    string y_cord;
    float y;
    input_file >> ws >> y_cord;
    y = stof(y_cord);
    return y;
}

int main() {

    //string path = "in.txt";
    ifstream file_in("in.txt");
    //file_in.open(path);
    if (!file_in.is_open()) {
        cout << "Error " << endl;
    }
    else {
        cout << "Successful " << endl;
        int xn = 0;
        int yn = 0;
        double max_r = 0; // Для правого
        double max_l = 0; // Для левого
        int first = 0; // счетчик, чтобы получить 1е значение вектора
        int x = 0;
        int y = 0;
        int *num_m_max_left = new int[2];
        int *num_m_max_right = new int[2];

        while (!file_in.eof()) {
            if (first == 0) {
                xn = get_x(file_in);
                yn = get_y(file_in);
                first++;
            }
            else{
                x = get_x(file_in);
                y = get_y(file_in);
                // Проверка на максимальный элемент + вычисление длины от вектора до точки
                double* D_LEFT = new double[2];
                double* D_RIGHT = new double[2];
                if (xn * y - yn *x  > 0.0 ) {        // слева
                    *(D_LEFT) = abs(xn * y - yn * x) / sqrt(xn ^ 2 + yn ^ 2);
                }
                else if (xn * y - yn * x == 0.0) {
                    *(D_RIGHT) = abs(xn * y - yn * x) / sqrt(xn ^ 2 + yn ^ 2);
                }
                else {          // справа
                    *(D_RIGHT)  = abs(xn * y - yn * x) / sqrt(xn ^ 2 + yn ^ 2);
                }

                if (*(D_RIGHT) > max_r) {
                    max_r = *(D_RIGHT);
                    *(num_m_max_left) = x; // x_max_right
                    *(num_m_max_left+1) = y; // y_max_right
                }
                if (*(D_LEFT) > max_l) {
                    max_l = *(D_LEFT);
                    *(num_m_max_right) = x; // x_max_left
                    *(num_m_max_right+1) = y; // y_max_left
                }
                delete[] D_LEFT;
                delete[] D_RIGHT;
            }
        }
        cout << "Leftmost: " << *(num_m_max_left) << " " << *(num_m_max_left + 1) << endl;
        cout << "Rightmost: " << *(num_m_max_right) << " " << *(num_m_max_right + 1) << endl;
        delete[] num_m_max_left;
        delete[] num_m_max_right;
        }
    file_in.close();
    return 0;
    }



