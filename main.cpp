#include "gemm.h"
#include "syrk.h"
#include "trsm.h"
#include "util.h"
#include <chrono>
#include <cstdio>
#include <gflags/gflags.h>

template <typename TestEnvironment>
std::chrono::system_clock::duration
profile(const HyperParameter &hyperParameter) {
  auto argument = TestEnvironment::prepare(hyperParameter);
  TestEnvironment::computeReference(argument);
  std::chrono::system_clock::duration duration{0};
  std::chrono::system_clock::time_point start;
  for (int i = 0; i < hyperParameter.repeat; ++i) {
    TestEnvironment::reset(argument);
    start = std::chrono::system_clock::now();
    TestEnvironment::compute(argument);
    duration += std::chrono::system_clock::now() - start;
  }
  TestEnvironment::check(argument);
  return duration;
}

template <typename EnvironmentA, typename EnvironmentB>
void compareEnvironment(const HyperParameter &hyperParameter,
                        const char *nameEnvironment, const char *nameA,
                        const char *nameB) {
  printf("======== Profiling %s ========\n", nameEnvironment);
  for (hyperParameter.iterator = hyperParameter.shape.begin();
       hyperParameter.iterator != hyperParameter.shape.end();
       ++hyperParameter.iterator) {
    std::chrono::system_clock::duration durationA, durationB;
    durationA = profile<EnvironmentA>(hyperParameter);
    durationB = profile<EnvironmentB>(hyperParameter);
    printf("(%d, %d, %d) - %s/%s: %Lf\n", hyperParameter.iterator->m,
           hyperParameter.iterator->n, hyperParameter.iterator->k, nameA, nameB,
           static_cast<long double>(durationA.count()) / durationB.count());
  }
}

DEFINE_int32(repeat, 100, "Repeat times");
DEFINE_int32(m, -1, "Matrix Size, -1 as default");
DEFINE_int32(n, -1, "Matrix Size, -1 as default");
DEFINE_int32(k, -1, "Matrix Size, -1 as default");

template <int lib> struct CblasLibrary;

template <> struct CblasLibrary<0> {
  constexpr static const char *name = "BLAS";
};

template <> struct CblasLibrary<1> {
  constexpr static const char *name = "MKL";
};

int main(int argc, char *argv[]) {
  printf("======== Simple Benchmark ========\n");
  printf("Use %s as backend\n", CblasLibrary<USE_MKL>::name);
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  const int repeat = FLAGS_repeat;

  printf("Parameter:\nRepeat: %d\nMatrix Size (m, n, k):\n", repeat);

  HyperParameter hyperParameter;
  hyperParameter.repeat = repeat;
  std::vector<int> shapeCandidate{8, 16, 32, 64, 128, 256}, mCandidate,
      nCandidate, kCandidate;
  if (FLAGS_m == -1) {
    mCandidate = shapeCandidate;
  } else {
    mCandidate.push_back(FLAGS_m);
  }
  if (FLAGS_n == -1) {
    nCandidate = shapeCandidate;
  } else {
    nCandidate.push_back(FLAGS_n);
  }
  if (FLAGS_k == -1) {
    kCandidate = shapeCandidate;
  } else {
    kCandidate.push_back(FLAGS_k);
  }

  for (const auto m : mCandidate) {
    for (const auto n : nCandidate) {
      for (const auto k : kCandidate) {
        hyperParameter.shape.push_back({m, n, k});
        printf(" | (%d, %d, %d)", m, n, k);
      }
    }
  }
  printf("\n");

  compareEnvironment<GEMMEnvironment<double>, GEMMEnvironment<float>>(
      hyperParameter, "GEMM", "double", "float");

  compareEnvironment<SYRKEnvironment<double>, SYRKEnvironment<float>>(
      hyperParameter, "SYRK", "double", "float");

  compareEnvironment<TRSMEnvironment<double>, TRSMEnvironment<float>>(
      hyperParameter, "TRSM", "double", "float");

  return 0;
}
