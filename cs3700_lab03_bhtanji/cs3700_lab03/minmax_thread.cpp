#include <chrono>
#include <iostream>
#include <mutex>
#include <random>
#include <utility>
#include <vector>
#include <thread>
#include <climits>
#include <ctime>

using namespace std;

constexpr long long value= 1000000;   
mutex myMutex;

void minmax_find(int th_id, unsigned long long& min, unsigned long long& max, const vector<int>& val, 
   unsigned long long beg, unsigned long long end){

    unsigned long long localMin = ULLONG_MAX;
    unsigned long long localMax = 0;
    
    for (auto it= beg; it < end; ++it){
	    if(static_cast<unsigned>(val[it]) < localMin){
        	localMin = val[it];
	    }
	    else if(static_cast<unsigned>(val[it]) > localMax){
        	localMax = val[it];
	    }
    }


    // print thread info
    cout << "Thread[" << th_id << "] - slice ["
         << beg << ":" << end << "]" 
	 << " loc_min: " << localMin << " loc_max: " << localMax << endl;

    //dont move to next line till you are able to recieve ok from another thread
    lock_guard<mutex> myLock(myMutex);
    if(localMin < min){
	    min = localMin;
    }
    else if(localMax > max){
	    max = localMax;
    }
}

int main(){

  cout << endl;

  // create rand value 
  vector<int> randValues;
  randValues.reserve(value);

  mt19937 engine (time(nullptr));
  uniform_int_distribution<> uniformDist(0, 10000000);
  for ( long long i=0 ; i< value ; ++i)
     randValues.push_back(uniformDist(engine));
 
  unsigned long long min = ULLONG_MAX, max = 0;
  auto start = chrono::system_clock::now();

  int threads = 4;
  thread t[threads];
  long long slice = value / threads;
  int startIdx=0;

  for (int i = 0; i < threads; ++i) {
    t[i] = thread(minmax_find, i, ref(min), ref(max), ref(randValues), startIdx, startIdx+slice-1);

    startIdx += slice;
  }

  for (int i = 0; i < threads; ++i)
     t[i].join();

  chrono::duration<double> dur= chrono::system_clock::now() - start;
  cout << "Time to find min and max: " << dur.count() << " seconds" << endl;
  cout << "Result: " << "min: " << min << " max: " << max << endl;

  cout << endl;

}
