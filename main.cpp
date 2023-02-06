#include "gemm.h"
#include "syrk.h"
#include "trsm.h"
#include <chrono>
#include <cstdio>
#include <gflags/gflags.h>

template <typename TestEnvironment>
void test(typename TestEnvironment::Argument &argument, const char *name) {
  printf("======== Profiling %s ========\n", name);
  TestEnvironment::prepare(argument);
  TestEnvironment::computeReference(argument);
  std::chrono::system_clock::duration duration{0};
  std::chrono::system_clock::time_point start;
  for (int i = 0; i < argument.repeat; ++i) {
    TestEnvironment::reset(argument);
    start = std::chrono::system_clock::now();
    TestEnvironment::compute(argument);
    duration += std::chrono::system_clock::now() - start;
  }
  printf(
      "Elapsed: %ld us\n",
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count());
  TestEnvironment::check(argument);
}

DEFINE_int32(repeat, 1000, "Repeat times");
DEFINE_int32(m, 100, "Matrix Size");
DEFINE_int32(n, 100, "Matrix Size");
DEFINE_int32(k, 100, "Matrix Size");

template<int lib>
struct CblasLibrary;

template<>
struct CblasLibrary<0> {
  constexpr static const char *name = "OpenBlas";
};

template<>
struct CblasLibrary<1> {
  constexpr static const char *name = "MKL";
};

int main(int argc, char *argv[]) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  const int repeat = FLAGS_repeat, m = FLAGS_m, n = FLAGS_n, k = FLAGS_k;
  printf("======== Simple Benchmark ========\n");
  printf("Parameter:\nRepeat: %d, Matrix Size: (m: %d, n: %d, k: %d)\n", repeat,
         m, n, k);
  printf("Use %s as backend\n", CblasLibrary<USE_MKL>::name);

  {
    using Environment = GEMMEnvironment<double>;
    Environment::Argument argument;
    argument.m = m;
    argument.n = n;
    argument.k = k;
    argument.repeat = repeat;
    test<Environment>(argument, "GEMM<double>");
  }

  {
    using Environment = GEMMEnvironment<float>;
    Environment::Argument argument;
    argument.m = m;
    argument.n = n;
    argument.k = k;
    argument.repeat = repeat;
    test<Environment>(argument, "GEMM<float>");
  }

  {
    using Environment = SYRKEnvironment<double>;
    Environment::Argument argument;
    argument.m = m;
    argument.k = k;
    argument.repeat = repeat;
    test<Environment>(argument, "SYRK<double>");
  }

  {
    using Environment = SYRKEnvironment<float>;
    Environment::Argument argument;
    argument.m = m;
    argument.k = k;
    argument.repeat = repeat;
    test<Environment>(argument, "SYRK<float>");
  }

  {
    using Environment = TRSMEnvironment<double>;
    Environment::Argument argument;
    argument.m = m;
    argument.n = n;
    argument.repeat = repeat;
    test<Environment>(argument, "TRSM<double>");
  }

  {
    using Environment = TRSMEnvironment<float>;
    Environment::Argument argument;
    argument.m = m;
    argument.n = n;
    argument.repeat = repeat;
    test<Environment>(argument, "TRSM<float>");
  }
}
