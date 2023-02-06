#pragma once
#include <Eigen/Core>
#if USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#include <cstdio>

template <typename> struct SYRKDispatcher;

template <> struct SYRKDispatcher<double> {
  constexpr static auto call = cblas_dsyrk;
};

template <> struct SYRKDispatcher<float> {
  constexpr static auto call = cblas_ssyrk;
};

template <typename DataType> struct SYRKEnvironment {
  struct Argument {
    int m, k, repeat;
    Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> A, resultEigen,
        resultBLAS;
  };

  static void prepare(Argument &argument) {
    argument.A.resize(argument.m, argument.k);
    argument.A.setRandom();
    argument.resultEigen.resize(argument.m, argument.m);
    argument.resultBLAS.resize(argument.m, argument.m);
    argument.resultBLAS.setZero();
  }

  static void reset(Argument &argument) {}

  static void computeReference(Argument &argument) {
    argument.resultEigen = argument.A * argument.A.transpose();
  }

  static void compute(Argument &argument) {
    const auto &M = argument.A;
    auto &resultBLAS = argument.resultBLAS;
    SYRKDispatcher<DataType>::call(
        CblasColMajor, CblasLower, CblasNoTrans, argument.m, argument.k, 1.,
        M.data(), argument.m, 0., resultBLAS.data(), argument.m);
  }

  static void check(Argument &argument) {
    argument.resultEigen.template triangularView<Eigen::Upper>().setZero();
    argument.resultBLAS.template triangularView<Eigen::Upper>().setZero();

    printf("Max Error: %e\n",
           (argument.resultEigen.array() - argument.resultBLAS.array())
               .abs()
               .maxCoeff());
  }
};
