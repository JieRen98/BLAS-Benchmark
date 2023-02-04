#pragma once
#include <Eigen/Core>
#include <cblas.h>
#include <cstdio>

template <typename> struct GEMMDispatcher;

template <> struct GEMMDispatcher<double> {
  constexpr static auto call = cblas_dgemm;
};

template <> struct GEMMDispatcher<float> {
  constexpr static auto call = cblas_sgemm;
};

template <typename DataType> struct GEMMEnvironment {
  struct Argument {
    int size, repeat;
    Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> M, resultEigen,
        resultBLAS;
  };

  static void prepare(Argument &argument) {
    argument.M.resize(argument.size, argument.size);
    argument.M.setRandom();
    argument.resultEigen.resize(argument.size, argument.size);
    argument.resultBLAS.resize(argument.size, argument.size);
  }

  static void computeReference(Argument &argument) {
    argument.resultEigen = argument.M * argument.M;
  }

  static void compute(Argument &argument) {
    const auto &size = argument.size;
    const auto &M = argument.M;
    auto &resultBLAS = argument.resultBLAS;
    GEMMDispatcher<DataType>::call(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                   size, size, size, 1., M.data(), size,
                                   M.data(), size, 0., resultBLAS.data(), size);
  }

  static void check(Argument &argument) {
    printf("Max Error: %e\n",
           (argument.resultEigen.array() - argument.resultBLAS.array())
               .abs()
               .maxCoeff());
  }
};
