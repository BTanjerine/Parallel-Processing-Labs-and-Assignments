#include <iostream>
#include <iomanip>
#include <cstdio>
#include <thread>
#include <cmath>
#include <mutex>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;
mutex myMutex;

void countPoints(int th_id, long long& numPoints, long long dataSize){
    long long total_points = 0;

    // seed random value based on time and thread number
    srand(time(nullptr) + th_id);

    // loop through slice of data set
    for(long long i=0; i<dataSize; i++){
        double x, y, dist;

	//find random x and y normalized
        x = (double)(rand())/(double)(RAND_MAX);
        y = (double)(rand())/(double)(RAND_MAX);

        dist = (x*x) + (y*y);

	//check if distance is within radius of unit circle
        if(dist <= 1) total_points++;
    }

    //lock variable update till unlocked by another thread
    lock_guard<mutex> myLock(myMutex);

    //update shared variable
    numPoints += total_points;
}


int main(int argc, char** argv){

    int numThreads = 0;
    long long dataSize = 0;

    //check if there are the correct number of inputs
    if (argc <= 1 || argc > 3){
        cout << "Invalid number of arguments!" << endl;
        cout << "tMC [1-10] [1-1000000]" << endl;
        return 0;
    }
    
    //check if arguments are in range
    try{
        if(atoi(argv[1]) <= 0 || atoi(argv[1]) > 10) throw atoi(argv[1]);
        else numThreads = atoi(argv[1]);

        if(atoi(argv[2]) < 10 || atoi(argv[2]) > 1000000) throw atoi(argv[2]);
        else dataSize = atoi(argv[2]);
    }
    catch(int err_val){
        cout << "value is over range!" << endl;
        cout << "tMC [1-10] [1-1000000]" << endl;
        return 0;
    }
  
    // setup slices and threads 
    thread t[numThreads];
    long long slice = dataSize/(long long)(numThreads);
    long long num_points = 0;

    auto start = chrono::system_clock::now();

    // init threads
    for(int i = 0; i < numThreads; i++){
        t[i] = thread(countPoints, i, ref(num_points), slice);
    }

    // synchronize threads
    for(int i = 0; i < numThreads; i++){
        t[i].join();
    }

    // find final calculations
    double est_pi = 4.0*((double)(num_points)/(double)(dataSize));
    double delta_pi = M_PI - est_pi; 
    chrono::duration<double> delta_time = chrono::system_clock::now() - start;

    // print final data
    cout << "pi: " << est_pi;
    cout << " delta: " << fixed << setprecision(4) << delta_pi;
    cout << " time (ms): " << fixed << setprecision(6) << delta_time.count() << endl;   

    return 0;
}

