# [CSED490C] Project Proposal
- Student ID: 20220848
- Name: 선민수
---

### 1. Description of algorithm
- Algorithm: **KNN**(K-Nearest Neighbors) for Problem-Based Benchmark Suite (PBBS)
- Description: Given `n` points in 2D or 3D, find `k` nearest neighbors for each point based on Euclidiean distance.
- Input:
    - Points: A sequence of points, each of which is a pair(2D point) or tripple(3D point) of double-precision floating-point numbers
    - `k` : The number of nearest neighbors to find for each points
- Output: A sequence of n tuples, each of which has length `k`. Each tuple identifies the indices of its `k` neaest neighbors from the input sequence.

### 2. Reference
[1] Shenshen Liang, Chen Wang, Ying Liu, Liheng Jian, "CUKNN: A PARALLEL IMPLEMENTATION OF K-NEAREST NEIGHBOR ON CUDA-ENABLED GPU", 2009 IEEE Youth Conference on Information, Computing and Telecommunication (YC-ICT), pp. 415-418, 2009
[2] Shenshen Liang, Ying Liu, Chen Wang, Liheng Jian, "A CUDA-based Parallel Implementation of K-Nearest Neighbor Algorithm", 2009 International Conference on Cyber-Enabled Distributed Computing and Knowledge Discovery (CyberC), pp. 291-296, 2009
[3] Hao Jiang, Yulin Wu, "Research on Parallelization of GPU-based K-Nearest Neighbor Algorithm", The 2017 International Conference on Cloud Technology and Communication Engineering (CTCE2017), 2017

### 3. Parallelization and Optimization
- Part 1: Calculation of Distances
    - Parallelization: Calculate the distances in parallelized manner.
    - Optimization: Reduce redundant distance calculation between two points. (e.g. Do not calculate the distance A-B and B-A twice as A,B is distinct points)
- Part 2: Finding k nearest neighbors(Sorting)
    - Parallelization: Sorting the distances in parallelized manner. (i.e. Adopting the sorting method which can be implemented in parallelized manner like bitonic sorting)
    - Optimization: Memory hiearchy optimization for optimization

### 4. Schedule
- Research: 1 week(~11/19)
- Implement Part 1: 1 week(~11/26)
- Implement Part 2: 1 week(~12/3)
- Experiment & Wrap Up: 1 week(~12/10)