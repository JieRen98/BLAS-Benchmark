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
    int m, n, k, repeat;
    Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> A, B, resultEigen,
        resultBLAS;
  };

  static void prepare(Argument &argument) {
    argument.A.resize(argument.m, argument.k);
    argument.B.resize(argument.k, argument.n);
    argument.A.setRandom();
    argument.B.setRandom();
    argument.resultEigen.resize(argument.m, argument.n);
    argument.resultBLAS.resize(argument.m, argument.n);
  }

  static void reset(Argument &argument) {}

  static void computeReference(Argument &argument) {
    argument.resultEigen = argument.A * argument.B;
  }

  static void compute(Argument &argument) {
    const auto &M = argument.A;
    const auto &N = argument.B;
    auto &resultBLAS = argument.resultBLAS;
    GEMMDispatcher<DataType>::call(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                   argument.m, argument.n, argument.k, 1.,
                                   M.data(), argument.m, N.data(), argument.k,
                                   0., resultBLAS.data(), argument.m);
  }

  static void check(Argument &argument) {
    printf("Max Error: %e\n",
           (argument.resultEigen.array() - argument.resultBLAS.array())
               .abs()
               .maxCoeff());
  }
};
