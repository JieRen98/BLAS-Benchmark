#pragma once
#include <array>
#include <vector>

struct HyperParameter {
  struct Shape {
    int m, n, k;
  };
  using ShapeVector = std::vector<Shape>;
  ShapeVector shape;
  mutable ShapeVector::const_iterator iterator;
  int repeat;
};
