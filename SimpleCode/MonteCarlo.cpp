#include <iostream>
#include <iomanip>
#include <cstdio>
#include <time.h>
#include <math.h>

#define TOTAL_POINTS 1000000

using namespace std;

int main(void){
    srand(time(NULL));

    int points = TOTAL_POINTS;
    double x = 0.0, y = 0.0;
    int total_points = 0;

    for(int i = 0; i<points; i++){
        x = static_cast<double>(rand())/static_cast<double>(RAND_MAX);
        y = static_cast<double>(rand())/static_cast<double>(RAND_MAX);

        double distance = (x*x) + (y*y);
        
        if(distance <= 1){
//            cout << "d: " << distance << " coords: " << x << ", " << y << endl; 

            total_points += 1; 
        }
    }

    double q_pi = 4.0 * (static_cast<double>(total_points)/static_cast<double>(TOTAL_POINTS));
    cout << "guesstimate: " << setprecision(5) << q_pi << " DELTA: " << M_PI-q_pi << endl;
}

