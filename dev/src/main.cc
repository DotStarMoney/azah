#include <iostream>
#include <memory>

struct A {
  int x;
};

struct B : public A {
  int y;
};

void f(std::unique_ptr<A> arg) {
  //
}

int main(int argc, char* argv[]) {
 

  auto q = std::make_unique<B>();

  f(std::move(q));

  return 0;
}
