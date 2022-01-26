#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <set>
#include <map>
#include <ctime>
#include <mpi.h>
#include <bits/stdc++.h>

// An instance holds RGB values, and if it's from the training set,
// an int qualifying if the value is a skintone or not is passed in.
class Instance {
private:
  double _r;
  double _g;
  double _b;
  int isSkin;

public:
  // Overloaded constructor for test instances.
  // Actual skin value is not stored; only displayed.
  Instance(double r, double g, double b) {
    _r = r;
    _g = g;
    _b = b;
    isSkin = 0;
  }

  // training instances instantiation.
  Instance(double r, double g, double b, int skin) {
    _r = r;
    _g = g;
    _b = b;
    isSkin = skin;
  }

  void setR(double r) { _r = r; }
  void setG(double g) { _g = g; }
  void setB(double b) { _b = b; }

  double getR() { return _r; }
  double getG() { return _g; }
  double getB() { return _b; }
  int getSkin() { return isSkin; }

  // Euclidean Distance = sqrt((a - a2)^2 + (b - b2)^2)
  double calcDistance(double r2, double g2, double b2) {
    return sqrt(pow((_r - r2), 2) + pow((_g - g2), 2) + pow((_b - b2), 2));
  }
};

// Declaring vectors of Instances.
std::vector<Instance> trainInst;
std::vector<Instance> testInst;

// k number of neighbors.
int k;

// iterates through the file, and if it sees 'char e' record to an array index.
std::vector<std::string> split(std::string input, char e) {
  std::vector<std::string> strList;
  std::string curr;

  for(int i = 0; i < input.size(); ++i) {
    if(input[i] != e) {
      curr.push_back(input[i]);
    } else {
      strList.push_back(curr), curr.clear();
    }
  }

  if(curr != "") {
    strList.push_back(curr);
  }

  return strList;
}


// determines if the test value is a 1 or 2.
int knnClassifier(double r, double g, double b, int k, std::vector<Instance> training) {
  // inits distance vals of training info to current RGB.
  // std::map is ordered by ascending keys, so i = 0 is the closest instance.
  std::map<double, int> distances;
  int class1 = 0;
  int class2 = 0;

  // populates the distances from the current RGB vals to
  // the list of all training vals.
  for(int i = 0; i < training.size(); i++) {
    double distance = training[i].calcDistance(r, g, b);
    distances.insert(std::pair<double, int>(distance, training[i].getSkin()));
  }

  // counts the values and determines the class.
  for(auto const& [key, val] : distances) {
    if(val == 1) {
      ++class1;
    } else {
      ++class2;
    }

    if(class1 + class2 == k) {
      break;
    }
  }

  if(class1 >= class2) {
    return 1;
  } else {
    return 2;
  }

  return 0;
}


int main(int argc, char **argv) {
  MPI_Init(NULL, NULL);

  int world_size;
  int rank;
  double start, mid, end;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Request requests[(world_size - 1) * 3];
  MPI_Status statuses[(world_size - 1) * 3];

  start = MPI_Wtime();

  std::string line;
  std::ifstream myfile("training.txt");

  //find min and max for standardization.
  int index = 0;
  double minR = INT_MAX;
  double maxR = INT_MIN;

  double minG = INT_MAX;
  double maxG = INT_MIN;

  double minB = INT_MAX;
  double maxB = INT_MIN;

  // go through the training data.
  if(myfile.is_open()) {
    while(getline(myfile, line)) {
      std::vector<std::string> parts = split(line, ' ');
      Instance instance(std::stod(parts[0]), std::stod(parts[1]), std::stod(parts[2]), std::stoi(parts[3]));

      // prepping standardization while reading in min/max vals
      //to avoid redundant, sequential loops.
      if(instance.getR() > maxR){
          maxR = instance.getR();
      } else if(instance.getR() < minR){
          minR = instance.getR();
      }

      if(instance.getG() > maxG){
          maxG = instance.getG();
      } else if(instance.getG() < minG){
          minG = instance.getG();
      }

      if(instance.getB() > maxB){
          maxB = instance.getB();
      } else if(instance.getB() < minB){
          minB = instance.getB();
      }

      // populate the training instances vector.
      trainInst.push_back(instance);
    }

    myfile.close();
  }

  k = sqrt(trainInst.size());

  //standardization. Sadly, we need to iterate through the whole
  // vector of training instances to find the max.
  // Can't double up with populating step bc we need the true max & min vals.
  double curr = 0;
  double regVal = 0;
  for(int i = 0; i < trainInst.size(); i++){
      curr = trainInst[i].getR();
      regVal = (curr - minR) / (maxR - minR);
      trainInst[i].setR(regVal);

      curr = trainInst[i].getG();
      regVal = (curr - minG) / (maxG - minG);
      trainInst[i].setG(regVal);

      curr = trainInst[i].getB();
      regVal = (curr - minB) / (maxB - minB);
      trainInst[i].setB(regVal);
  }

  std::string line1;
  std::ifstream myfile1("test.txt");

  if(myfile1.is_open()) {
    while(getline(myfile1, line1)) {
      std::vector<std::string> parts = split(line1, ' ');
      Instance testInstance(((std::stod(parts[0]) - minR) / (maxR - minR)), ((std::stod(parts[1]) - minG) / (maxG - minG)), ((std::stod(parts[2]) - minB) / (maxB - minB)));
      testInst.push_back(testInstance);
    }
  }

  if(rank == 0) {
    mid = MPI_Wtime();

    int index = 1;
    for(int i = 1; i < testInst.size(); i++) {
    	double r = testInst[i].getR();
      MPI_Isend(&r, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, requests + index);
      ++index;

      double g = testInst[i].getG();
      MPI_Isend(&g, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, requests + index);
      ++index;

      double b = testInst[i].getB();
      MPI_Isend(&b, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, requests + index);
    	++index;
    }

    double r = testInst[0].getR();
  	double g = testInst[0].getG();
  	double b = testInst[0].getB();
  	int pred = knnClassifier(r, g, b, k, trainInst);
  	printf("Class for %d object is: %d\n", rank + 1, pred);
  } else {
    double r;
    double g;
    double b;
    MPI_Irecv(&r, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, requests + rank + 1);
    MPI_Irecv(&g, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, requests + rank + 2);
    MPI_Irecv(&b, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, requests + rank + 3);

    MPI_Wait(requests + rank + 1, statuses + rank + 1);
    MPI_Wait(requests + rank + 2, statuses + rank + 2);
    MPI_Wait(requests + rank + 3, statuses + rank + 3);
    int pred = knnClassifier(r, g, b, k, trainInst);
    printf("Class for %d object is: %d\n", rank + 1, pred);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if(rank == 0){
    end = MPI_Wtime();
    printf("Elapsed total time: %.2f seconds.\n", (end - start));
    printf("Elapsed parallel time: %.2f seconds.\n", (end - mid));
  }
  MPI_Finalize();

  return 0;
}
