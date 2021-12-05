#include <iostream>
#include <string>
#include <fstream>
#include <cmath>

using namespace std;


float get_x(ifstream &file_in) // функция получения координаты x
{
    string x_cord;
    float x;
    file_in >> x_cord;
    x = stof(x_cord);
    return x;
}

float get_y(ifstream &file_in)  // функция получения координаты y
{
    string y_cord;
    float y;
    file_in >> ws >> y_cord;
    y = stof(y_cord);
    return y;
}

float coordx(string str) {
    int determinator = str.find(" ");
    string X = str.substr(0, determinator);
    float x = stoi(X);
    return x;
}

float coordy(string str)
{
    int determinator = str.find(" ");
    string Y = str.substr(determinator + 1, str.length());
    float y = stoi(Y);
    return y;

}

int main() {
    double x_max_left = 0;
    double y_max_left = 0;
    double x_max_right = 0;
    double y_max_right = 0;
    double max_r = 0; // Для правого
    double max_l = 0; // Для левого
    double Dist = 0 ;
    int xn = 0;
    int yn = 0;
    int x = 0;
    int y = 0;
    int flag = 0;
    string str;

    ifstream file_in("in.txt");


        while (getline(file_in, str)) {

            if (flag == 0 ){
                xn = coordx(str);
                yn =  coordy(str);
                flag++;
            }
            else{
                x = coordx(str);
                y = coordy(str);

                Dist = abs(xn * y - yn * x / sqrt(xn ^ 2 + yn ^ 2));
                // Проверка на максимальный элемент + вычисление длины от вектора до точки

                if (xn * y - yn * x  > 0.0 ) { // слева
                    if (Dist >= max_l) {
                        max_l = Dist;
                        x_max_left = x; // x_max_right
                        y_max_left = y; // y_max_right
                    }


                }
                else {          // справа
                    if (Dist >= max_r) {
                        max_r = Dist;
                        x_max_right = x; // x_max_left
                        y_max_right = y; // y_max_left
                    }
                }
            }


        }
    file_in.close();
    cout << "Leftmost: " << x_max_left << " " << y_max_left << endl;
    cout << "Rightmost: " <<  x_max_right << " " << y_max_right << endl;
    return 0;
    }



