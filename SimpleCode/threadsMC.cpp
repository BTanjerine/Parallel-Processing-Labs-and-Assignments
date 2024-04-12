#include <iostream>
#include <cstdio>
#include <stdexcept>
//#include <threads>
//#include <mutex>
#include <cstdlib>
#include <ctime>

using namespace std;

void countPoints(int th_id, int &numPoints, unsigned long long dataSize){
    int total_points = 0;
    srand(time(nullptr) + th_id);

    for(int i=0; i<dataSize; i++){
        double x, y, dist;

        x = (double)(rand())/(double)(RAND_MAX);
        y = (double)(rand())/(double)(RAND_MAX);

        dist = (x*x) + (y*y);

        if(dist <= 1) total_points++;
    }

    lock_guard<mutex> myLock(myMutex);
    numPoints += total_points;
}


int main(int argc, char** argv){

    int numThreads = 0
    unsigned long long dataSize = 0;

    //check if there are the correct number of inputs
    if (argc <= 1 || argc > 3){
        cout << "Too many arguments!" << endl;
        cout << "tMC [1-10] [1-1000000]" << endl;
        return 0;
    }
    
    //check if arguments are in range
    try{
        if(atoi(argv[0]) > 100) throw atoi(argv[0]);
        else numThreads = atoi(argv[0]);

        if(atoi(argv[1]) > 100) throw atoi(argv[1]);
        else dataSize = atoi(argv[1]);
    }
    catch(int err_val){
        cout << "value is over range!" << endl;
        cout << "tMC [1-10] [1-1000000]" << endl;
        return 0;
    }
   
    thread t[numThreads];
    unsigned long long slice = dataSize/numThreads;
    unsigned long long num_points = 0;

    for(int i = 0; i < numThreads; i++){
        t[i] = thread(countPoints, i, &num_points, slice);
    }

    for(int i = 0; i < numThreads; i++){
        t[i].join();
    }

    long double mpi_pi = 4.0*((long double)(num_points)/(long double)(dataSize));

    cout << "pi: " << mpi << endl;   

    return 0;
}

