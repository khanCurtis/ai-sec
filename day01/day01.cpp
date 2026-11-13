#include <torch/torch.h>
#include <iostream>
#include <vector>

using namespace std;
using namespace torch::indexing;

void raw_mem() {
  cout << "\n--- Part 7: Raw Memory ---\n";
  vector<float> v = {1.0f, 2.0f, 3.0f};
  torch::Tensor shared  =torch::from_blob(v.data(), {3}, torch::kFloat);
  torch::Tensor owned = shared.clone();

  v[0] = 100.0f;
  cout << shared << owned;
  float* p = owned.data_ptr<float>();
  cout << p[0] << " " << p[1] << "\n";
}

void aliasing() {
  cout << "\n--- Part 6: Aliasing ---\n";
  torch::Tensor a = torch::ones({3});
  torch::Tensor alias = a;
  torch::Tensor copy = a.clone();

  a.add_(5);
  cout << a << alias << copy;
}

void shapes() {
  cout << "\n--- Part 5: Shapes ---\n";
  torch::Tensor a = torch::ones({2, 3});
  torch::Tensor b = torch::zeros({2, 3});

  torch::Tensor tall = torch::cat({a, b}, 0);
  torch::Tensor wide = torch::cat({a, b}, 1);
  cout << tall.sizes() << " " << wide.sizes() << "\n";
  
  torch::Tensor flat = tall.reshape({12});
  torch::Tensor back = flat.reshape({3, 4});
  cout << flat.sizes() << " " << back.sizes() << "\n";
}

void indexing() {
  cout << "\n--- Part 4: Indexing ---\n";
  torch::Tensor t = torch::tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});

  cout << t.index({0}) << "\n";
  cout << t.index({Slice(), 0}) << "\n";
  cout << t.index({Ellipsis, -1}) << "\n";
  cout << t.index({Slice(0, 2), Slice(1, 3)}) << "\n";
  cout << t.index({1, 1}) << "\n";

  t.index_put_({Slice(), 1}, 0);
  cout << t << "\n";
}

void math() {
  cout << "\n--- Part 3: Math ---\n";
  torch::Tensor a = torch::tensor({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  torch::Tensor b = torch::ones({2, 3}) * 10;

  cout << a + b << "\n";
  cout << a * b << "\n";
  cout << a.matmul(a.t()) << "\n";

  torch::Tensor m = torch::tensor({{1, 2, 3}, {4, 5, 6}});
  cout << m.sum() << "\n";
  cout << m.sum(0) << "\n";
  cout << m.sum(1) << "\n";
  cout << m.to(torch::kFloat).mean() << "\n";
  cout << m.max() << "\n";

  int64_t total = m.sum().item<int64_t>();
  cout << "plain C++ int: " << total << "\n";
}

void attributes() {
  cout << "\n--- Part 2: Attributes ---\n";
  torch::Tensor a = torch::tensor({{1, 2, 3}, {4, 5, 6}});

  cout << "sizes: " << a.sizes() << "\n";
  cout << "dim: " << a.dim() << "\n";
  cout << "numel: " << a.numel() << "\n";
  cout << "dtype: " << a.dtype() << "\n";
  cout << "device: " << a.device() << "\n";
  cout << "rows: " << a.size(0) << ", cols: " <<a.size(1) << "\n";

  torch::Tensor f = a.to(torch::kFloat32);
  cout << "converted dtype: " << f.dtype() << "\n";
  cout << boolalpha << "cuda: " << torch::cuda::is_available() << "\n";
}

void creation() {
  cout << "\n--- Part 1: Creation ---\n";

  torch::Tensor x = torch::tensor({1.0, 2.0, 3.0});
  cout << x << "\n";

  torch::Tensor a = torch::tensor({{1, 2, 3}, {4, 5, 6}});
  cout << a << "\n";

  torch::Tensor z = torch::zeros({2, 3});
  torch::Tensor o = torch::ones({2, 3});
  torch::Tensor r = torch::rand({2, 3});
  cout << z << "\n" << o << "\n" << r << "\n";

  torch::Tensor o2 = torch::ones_like(a);
  cout << o2 << "\n";
}

int main() {
  torch::Tensor t = torch::eye(3);
  cout << t << endl;

  torch::manual_seed(0);
  creation();
  attributes();
  math();
  indexing();
  shapes();
  aliasing();
  raw_mem();
}
