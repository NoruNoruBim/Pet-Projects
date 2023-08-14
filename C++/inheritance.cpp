#include <iostream>
#include <cmath>

using namespace std;


class Figure {
    public:
        virtual double S() = 0;
        virtual double P() = 0;
        
        virtual void show() {
            cout << "Hi!" << endl;
        }
};

class Triangle : public Figure {
    public:
        double AB;
        double BC;
        double AC;
    public:
        
        Triangle() {
            this->AB = 0;
            this->BC = 0;
            this->AC = 0;
        }


        Triangle(double AB, double BC, double AC) : AB(AB), BC(BC), AC(AC) {}


        double S() override {
            double p = (this->AB + this->BC + this->AC) / 2;
            return sqrt(p * (p - this->AB) * (p - this->BC) * (p - this->AC));
        }
        
        double P() override {
            return this->AB + this->BC + this->AC;
        }
        
        void show() override {
            cout << this->AB << endl;
            cout << this->BC << endl;
            cout << this->AC << endl;
        }
    
        ~Triangle() {}
};

class TriangleChild : public Triangle {
    public:
        void show() override {
            cout << "child!" << endl;
        }
};


int main() {
    
    // Triangle t(3, 4, 5);
    
    // t.AB = 3;
    // t.BC = 4;
    // t.AC = 5;
    
    // cout << t.S() << " " << t.P() << endl;
    
    // t.show();
    
    // TriangleChild tc;
    
    // tc.show();
    
    
    Triangle* arr[10];
    Triangle t1 = Triangle(3, 4, 5);
    
    arr[0] = &t1;
    
    // cout << arr[0]->AB << endl;
    
    TriangleChild tc;
    
    arr[1] = (Triangle*)(&tc);
    
    arr[0]->show();
    arr[1]->show();
    
    return 0;
}
