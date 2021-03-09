cmake_minimum_required(VERSION 3.17)
project(ER221ProdEscalar_ER222Axb)

add_executable(ER221ProdEscalar_ER222Axb GSL_ProdEscalar_Axb.cpp)

#Acesso ao GSL-GNU
target_link_libraries(ER221ProdEscalar_ER222Axb -lgsl -lgslcblas -lm)

#Acesso ao OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp")
set(SOURCE_FILES GSL_ProdEscalar_Axb.cpp)