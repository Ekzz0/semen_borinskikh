#include <iostream>
#include <fstream>
#include <iomanip>
#include <valarray>

using namespace std;
string * StringToMass(string base_str, char delim, int size){

    size_t pos = 0;
    size_t base_str_size = base_str.size();
    //size_t delim_size = delim.size();
    size_t delim_size=1;
    string temp;
    int i=0;
    string *mass_str = new string[size];
    while (pos < base_str_size) {
        temp = temp.assign(base_str, pos, base_str.find(delim, pos) - pos);
        if (temp.size() > 0)  // проверка на пустую строку при необходимости
        {
           // cout << temp << endl;
            mass_str[i] = temp;
            i++;
            if(i==size){
                break;
            }
            //cout << i << endl;
        }
        pos += temp.size() + delim_size;

    }

    return mass_str;
}
std::tuple<int, int> step(int dividend, int divisor) {
    return  std::make_tuple(dividend / divisor, dividend % divisor);
}
int main() {
    std::string line;
    double xn,yn,fx_r,fy_r,fx_l,fy_l;
    double finde_value=1;
    double max_d_r=0;
    double max_d_l=0;
    bool isFirstLine= true;
    bool isTwiceLine_r= true,isTwiceLine_l= true;
    std::ifstream in("in.txt"); // окрываем файл для чтения
    if (in.is_open())
    {
        double x=0; double y=0;
        in >> x >> y;
        xn=(-1)*x;
        yn=(-1)*y;
        //std::<<  <<::endl;
        while (in >> x >> y)
        {
                //растояние
                double dist;
                if(xn !=0 && yn !=0){
                    dist=(abs(-1*(x)/xn+y/yn)/sqrt((1/xn)*(1/xn)+(1/yn)*(1/yn)));
					dist = std::round(dist * 10000000000.0) / 10000000000.0;
                   // dist=(abs((yn-0)*x-(xn-0)*y+xn*0-yn*0))/(sqrt(pow(xn-0,2)+pow(yn-0,2))); //вектор как прямая заданная 2мя точками
                }
                else{
                    (xn ==0)? dist =x:dist= y;
                    /*if(xn ==0){
                        dist= x;
                        //dist=sqrt(pow(x-xn,2)+pow(y-yn,2)); //как растояние между 2мя точками
                    }
                    else{
                        dist=y;
                    }*/
                }
                //<0 - справа:
                if(((xn-0)*(y-0)-(yn-0)*(x-0))>=0){
                    if(dist>=max_d_r){
                        max_d_r=dist;
                        fx_r=x;
                        fy_r=y;
                    }
                    /*if(isTwiceLine_r){
                        isTwiceLine_r= false;
                        max_d_r=dist;
                        fx_r=x;
                        fy_r=y;
                    }
                    else{
                    }*/
                }//>0 - слева
                else{
                    if(dist>=max_d_l){
                        max_d_l=dist;
                        fx_l=x;
                        fy_l=y;
                    }

                    /*if(isTwiceLine_l){
                        isTwiceLine_l= false;
                        max_d_l=dist;
                        fx_l=x;
                        fy_l=y;
                    }
                    else{
                    }*/
                }

            //delete[] values;
        }
    }
    in.close();     // закрываем файл
    std::cout << "Leftmost: " << fx_l << " " << fy_l << "\n";
    std::cout << "Rightmost: " << fx_r << " " << fy_r << "\n";

    return 0;
}
