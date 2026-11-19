#include <torch/torch.h>
#include <iostream>

using namespace std;

void line_fit() {
  cout << "\n--- Part 6: Learning a line ---\n";

  torch::manual_seed(0);

  torch::Tensor X = torch::linspace(-1, 1, 100);
  torch::Tensor Y = X * 3 + 2 + torch::randn({100}) * 0.1;

  torch::Tensor w = torch::randn({1}, torch::requires_grad());
  torch::Tensor b = torch::zeros({1}, torch::requires_grad());
  const double lr = 0.1;
  //const double lr = 0.001; // Slow Learning
  //const double lr = 1.5; // Fast Learning

  for (int step = 0; step < 200; ++step) {
    torch::Tensor pred = w * X + b;
    torch::Tensor loss = (pred - Y).pow(2).mean();
    loss.backward();

    {
      torch::NoGradGuard no_grad;

      w.sub_(w.grad() * lr);
      b.sub_(b.grad() * lr);
    }

    w.grad().zero_();
    b.grad().zero_();

    if (step % 20 == 0) {
      cout << "step " << step << " loss = " << loss.item<float>()
        << " w = " << w.item<float>() << " b = " << b.item<float>() << "\n";
    }
  }

  cout << "final: w = " << w.item<float>() << " (true 3), b = " << b.item<float>() << " (true 2)\n";
}

void no_grad() {
  cout << "\n--- Part 5: No Gradient ---\n";
  
  torch::Tensor w = torch::tensor(2.0, torch::requires_grad());
  torch::Tensor y = w * 3;
  y.backward();

  {
    torch::NoGradGuard no_grad;

    w.sub_(w.grad() * 0.1);
    cout << "w after step: " << w << "\n";
    cout << "tracked inside guard: " << (w * 2).requires_grad() << "\n";
  }

  cout << "tracked outside guard: " << (w * 2).requires_grad() << "\n";
}

void accumulate() {
  cout << "\n--- Part 4: Accumulation ---\n";

  torch::Tensor w = torch::tensor(2.0, torch::requires_grad());
  torch::Tensor x = torch::tensor(3.0);

  for (int i = 0; i < 3; ++i) {
    torch::Tensor y = w * x;
    y.backward();
    cout << "after backward #" << i + 1 << ": " << w.grad() << "\n";
  }

  w.grad().zero_();
  cout << "after zero_: " << w.grad() << "\n";
}

void check() {
  cout << "\n--- Part 3: Check ---\n";

  torch::Tensor w = torch::tensor(2.0, torch::requires_grad());
  torch::Tensor b = torch::tensor(1.0, torch::requires_grad());
  torch::Tensor x = torch::tensor(3.0);
  torch::Tensor t = torch::tensor(5.0); // target value
  
  torch::Tensor pred = w * x + b;
  torch::Tensor loss = (pred - t).pow(2);
  loss.backward();

  cout << "autograd: dL/dw = " << w.grad() << "dL/db = " << b.grad() << "\n";

  //same thing by hand
  float wv = 2.0f, bv = 1.0f, xv = 3.0f, tv = 5.0f;
  float err = wv * xv + bv - tv;
  float dw = 2 * err * xv;
  float db = 2 * err;
  cout << "by hand: dL/dw = " << dw << "dL/db = " << db << "\n";
}

void graph() {
  cout << "\n--- Part 2: Graph ---\n";
  
  torch::Tensor w = torch::tensor(2.0, torch::requires_grad());
  torch::Tensor x = torch::tensor(3.0);
  torch::Tensor p = w * x;
  torch::Tensor y = p + 1;

  cout << "w is leaf: " << w.is_leaf() << "\n";
  cout << "p is leaf: " << p.is_leaf() << "\n";
  cout << "x requires grad: " << x.requires_grad() << "\n";
  cout << "p.grad_fn: " << p.grad_fn()->name() << "\n";
  cout << "y.grad_fn: " << y.grad_fn()->name() << "\n";
}

void basic_grad() {
  cout << "--- Part 1: Basic Gradients ---\n";
  
  torch::Tensor w = torch::tensor(2.0, torch::requires_grad());
  torch::Tensor x = torch::tensor(3.0);

  torch::Tensor y = w * x + 1;
  y.backward();

  cout << "y = " << y << "\n";
  cout << "dy/dw = " << w.grad() << "\n";
}

int main() {
  cout << boolalpha;

  basic_grad();
  graph();
  check();
  accumulate();
  no_grad();
  line_fit();
}
