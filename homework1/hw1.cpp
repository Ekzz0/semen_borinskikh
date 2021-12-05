#include <iostream>
#include <string>
#include <fstream>
#include <cmath>

using namespace std;

float get_x(string num) {
    int probel = num.find(" "); // возвращается номер пробела
    //string X = num.substr(0, probel);
    string X;
    for (int i = 0; i<probel; ++i){
        X.append(1, num[i]);
    }
    float x = stoi(X);
    return x;
}

float get_y(string num)
{
    int probel = num.find(" ");
    //string Y = num.substr(probel + 1, num.length());
    string Y;
    for (int i = probel; i<num.length(); ++i){
        Y.append(1, num[i]);
    }
    float y = stoi(Y);
    return y;

}

int main() {
    double x_max_left;
    double y_max_left;
    double x_max_right;
    double y_max_right;
    double max_r = 0; // Для правого
    double max_l = 0; // Для левого
    double Dist;
    int xn;
    int yn;
    int x;
    int y;
    int flag = 0;
    string num;

    ifstream file_in("in.txt");

    if (file_in.is_open()){
        while (getline(file_in, num)) {

            if (flag == 0 ){
                xn = get_x(num);
                yn =  get_y(num);
                flag++;
            }
            else{
                x = get_x(num);
                y = get_y(num);

                Dist = abs((xn * y - yn * x) / sqrt(xn*xn + yn *yn));
                // Проверка на максимальный элемент + вычисление длины от вектора до точки

                if (xn * y - yn * x  <= 0.0 ) { // справа
                    if ( Dist >= max_r) {
                        max_r = Dist;
                        x_max_right = x; // x_max_left
                        y_max_right = y; // y_max_left
                    }

                }
                else {          // слева
                    if ( Dist >= max_l) {
                        max_l = Dist;
                        x_max_left = x; // x_max_right
                        y_max_left = y; // y_max_right
                    }
                }
            }


        }
    }
    else{
        //cout << "Error: " << endl;
    }
    file_in.close();
    cout << "Leftmost: " << x_max_left << " " << y_max_left << endl;
    cout << "Rightmost: " <<  x_max_right << " " << y_max_right << endl;
    return 0;
    }



