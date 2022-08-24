#include <torch/extension.h>
using namespace torch::indexing;
#include <iostream>
torch::Tensor blocksparse_conv2d(torch::Tensor g,
      torch::Tensor X,
      torch::Tensor W,
      at::IntArrayRef stride,
      at::IntArrayRef padding,
      at::IntArrayRef dilation,
      int groups) {
    // X = X.transpose(0,1);
    // W = W.transpose(0,1);
    for(int i=0; i < X.size(0); i++)
    {
        auto X_slice = X.index({Slice(i, i+1)});
        auto W_slice = W.index({Slice(i, i+1)});
        // std::cout << X_slice.sizes() << " " << W_slice.sizes() << std::endl;
        auto tmp = at::cudnn_convolution_backward_weight(
          g.index({0}).sizes(),
          W_slice,
          X_slice,
          padding,
          stride,
          dilation,
          groups,
          false,
          false,
          false);
        g.index_put_({i}, tmp);
        // auto tmp = at::conv2d(X_slice, W_slice, None, dilation, padding, stride, groups).transpose(0,1);
        // g.index_put_({i}, tmp.index({Slice(None, g.size(1)), Slice(None, g.size(2)), Slice(None, g.size(3)), Slice(None, g.size(4))}));
    }
    return g;
}

torch::Tensor weights_conv2d(
      at::IntArrayRef weight_shape,
      torch::Tensor X,
      torch::Tensor W,
      at::IntArrayRef stride,
      at::IntArrayRef padding,
      at::IntArrayRef dilation,
      int groups) {
    // X = X.transpose(0,1);
    // W = W.transpose(0,1);


    return at::cudnn_convolution_backward_weight(
          weight_shape,
          W,
          X,
          padding,
          stride,
          dilation,
          groups,
          false,
          false,
          false);
        // auto tmp = at::conv2d(X_slice, W_slice, None, dilation, padding, stride, groups).transpose(0,1);
        // g.index_put_({i}, tmp.index({Slice(None, g.size(1)), Slice(None, g.size(2)), Slice(None, g.size(3)), Slice(None, g.size(4))}));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("blocksparse_conv2d", &blocksparse_conv2d, "LLTM blocksparse_conv2d");
  m.def("weights_conv2d", &weights_conv2d, "LLTM weight_conv2d");
}