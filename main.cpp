#include "gemm.h"
#include "syrk.h"
#include <chrono>
#include <cstdio>
#include <gflags/gflags.h>

template <typename TestEnvironment>
void test(typename TestEnvironment::Argument &argument, const char *name) {
  printf("======== Profiling %s ========\n", name);
  TestEnvironment::prepare(argument);
  TestEnvironment::computeReference(argument);
  std::chrono::system_clock::time_point start;
  start = std::chrono::system_clock::now();
  for (int i = 0; i < argument.repeat; ++i) {
    TestEnvironment::compute(argument);
  }
  auto duration = std::chrono::system_clock::now() - start;
  printf(
      "Elapsed: %ldus\n",
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count());
  TestEnvironment::check(argument);
}

DEFINE_int32(repeat, 1000, "Repeat times");
DEFINE_int32(size, 100, "Matrix Size");

int main(int argc, char *argv[]) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  const int repeat = FLAGS_repeat, size = FLAGS_size;
  printf("======== Simple Benchmark ========\n");
  printf("Parameter:\nRepeat: %d, Matrix Size: %d\n", repeat, size);

  {
    using Environment = GEMMEnvironment<double>;
    Environment::Argument argument;
    argument.size = size;
    argument.repeat = repeat;
    test<Environment>(argument, "GEMM<double>");
  }

  {
    using Environment = GEMMEnvironment<float>;
    Environment::Argument argument;
    argument.size = size;
    argument.repeat = repeat;
    test<Environment>(argument, "GEMM<float>");
  }

  {
    using Environment = SYRKEnvironment<double>;
    Environment::Argument argument;
    argument.size = size;
    argument.repeat = repeat;
    test<Environment>(argument, "SYRK<double>");
  }

  {
    using Environment = SYRKEnvironment<float>;
    Environment::Argument argument;
    argument.size = size;
    argument.repeat = repeat;
    test<Environment>(argument, "SYRK<float>");
  }
}
