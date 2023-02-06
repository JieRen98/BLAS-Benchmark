#pragma once
#include "util.h"
#include <Eigen/Core>
#if USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
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
    int m, n;
    Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> A, resultBLAS;
  };

  static Argument prepare(const HyperParameter &hyperParameter) {
    Argument argument;
    argument.m = hyperParameter.iterator->m;
    argument.n = hyperParameter.iterator->n;
    argument.A.resize(argument.m, argument.m);
    argument.A.setRandom();
    argument.A = argument.A * argument.A.transpose();
    argument.A.template triangularView<Eigen::StrictlyUpper>().setZero();
    argument.resultBLAS.resize(argument.m, argument.n);
    return argument;
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
    //    printf("Max Error: %e\n",
    //           (argument.resultBLAS.array() - 1).abs().maxCoeff());
  }
};
