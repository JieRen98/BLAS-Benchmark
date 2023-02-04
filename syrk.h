#pragma once
#include <Eigen/Core>
#include <cblas.h>
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
    int size, repeat;
    Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> M, resultEigen,
        resultBLAS;
  };

  static void prepare(Argument &argument) {
    argument.M.resize(argument.size, argument.size);
    argument.M.setRandom();
    argument.resultEigen.resize(argument.size, argument.size);
    argument.resultBLAS.resize(argument.size, argument.size);
    argument.resultBLAS.setZero();
  }

  static void computeReference(Argument &argument) {
    argument.resultEigen = argument.M * argument.M.transpose();
  }

  static void compute(Argument &argument) {
    const auto &size = argument.size;
    const auto &M = argument.M;
    auto &resultBLAS = argument.resultBLAS;
    SYRKDispatcher<DataType>::call(CblasColMajor, CblasLower, CblasNoTrans,
                                   size, size, 1., M.data(), size, 0.,
                                   resultBLAS.data(), size);
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
