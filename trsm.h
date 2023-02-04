#pragma once
#include <Eigen/Core>
#include <cblas.h>
#include <cstdio>

template <typename> struct TRSMDispatcher;

template <> struct TRSMDispatcher<double> {
  constexpr static auto call = cblas_dtrsm;
};

template <> struct TRSMDispatcher<float> {
  constexpr static auto call = cblas_strsm;
};

template <typename DataType> struct TRSMEnvironment {
  struct Argument {
    int m, n, repeat;
    Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> A, resultBLAS;
  };

  static void prepare(Argument &argument) {
    argument.A.resize(argument.m, argument.m);
    argument.A.setRandom();
    argument.A = argument.A * argument.A.transpose();
    argument.A.template triangularView<Eigen::StrictlyUpper>().setZero();
    argument.resultBLAS.resize(argument.m, argument.n);
  }

  static void reset(Argument &argument) {
    argument.resultBLAS.setOnes();
    argument.resultBLAS = argument.A.template triangularView<Eigen::Lower>() *
                          argument.resultBLAS;
  }

  static void computeReference(Argument &) {}

  static void compute(Argument &argument) {
    const auto &M = argument.A;
    auto &resultBLAS = argument.resultBLAS;
    TRSMDispatcher<DataType>::call(CblasColMajor, CblasLeft, CblasLower,
                                   CblasNoTrans, CblasNonUnit, argument.m,
                                   argument.n, 1., M.data(), argument.m,
                                   resultBLAS.data(), argument.m);
  }

  static void check(Argument &argument) {
    printf("Max Error: %e\n",
           (argument.resultBLAS.array() - 1).abs().maxCoeff());
  }
};
