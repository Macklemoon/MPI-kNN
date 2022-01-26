# 474-Project-2
## Project 2: Parallelized kNN
The dataset, as linked below, defines a table of 245,057 entries sampling RGB values in hopes of identifying if the sampled color is a valid skintone.
*From the dataset website: Skin and Nonskin dataset is generated using skin textures from face images of diversity of age, gender, and race people.*
There are 50,859 entries that denote valid skin textures.
scikit.learn's implementation of kNN is unfortunately sequential, so in order to speed up the process, we worked on a solution involving MPI to reduce the compile time by parallelizing steps in the kNN process. The repository mentioned below was the basis for this project's skeleton, and optimizations have been made in our program to reduce the time spent compiling. See the *report.pdf* file for more information and demos images.
The project was finished in C++, and based off this github repository: https://github.com/ignatij/knn-mpi.
The training and test sets are provided here: https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation#.

## Group members:
Michael Moon m-moon@csu.fullerton.edu
Garrett Reeve garrett.reeve@csu.fullerton.edu

## To compile:
compile with the following:</br>
mpic++ -o a.out main.cpp -std=c++17
</br></br>
execute with:</br>
mpirun -n 100 -oversubscribe ./a.out
