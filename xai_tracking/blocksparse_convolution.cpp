#include <torch/extension.h>
using namespace torch::indexing;

torch::Tensor blocksparse_conv2d(torch::Tensor g,
    torch::Tensor X,
    torch::Tensor W,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int groups) {
    for(int i=0; i < X.size(1); i++)
    {
        auto X_slice = X.index({Slice(), Slice(i, i+1)});
        auto W_slice = W.index({Slice(), Slice(i, i+1)});
        auto tmp = at::conv2d(X_slice, W_slice, None, dilation, padding, stride, groups).transpose(0,1);
        g.index_put_({i}, tmp.index({Slice(None, g.size(1)), Slice(None, g.size(2)), Slice(None, g.size(3)), Slice(None, g.size(4))}));
    }
    return g;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("blocksparse_conv2d", &blocksparse_conv2d, "LLTM blocksparse_conv2d");
}