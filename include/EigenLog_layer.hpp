#ifndef CAFFE_EIGENLOG_LAYER_HPP_
#define CAFFE_EIGENLOG_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class EigenLogLayer: public Layer<Dtype> {
 public:
  explicit EigenLogLayer(const LayerParameter& param)
      : Layer<Dtype>(param), diff_() {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "EigenLog"; }
  /**
   * Similar to EuclideanLossLayer, in CORALLoss we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  Blob<Dtype> diff_;
  Blob<Dtype> eig;
  Blob<Dtype> U;
  Blob<Dtype> V;
  Blob<Dtype> eig_log;
  Blob<Dtype> eig_inv;
  Blob<Dtype> eig_matx;
  Blob<Dtype> eig_log_matx;
  Blob<Dtype> eig_inv_matx;
  Blob<Dtype> diff_sys;
  Blob<Dtype> dU;
  Blob<Dtype> deigen;
  Blob<Dtype> P;
  Blob<Dtype> PT;
  Blob<Dtype> sys;
  Blob<Dtype> P_sys;
  Blob<Dtype> temp1;
  Blob<Dtype> temp2;
  Blob<Dtype> iden_matx;
  Blob<Dtype> cov;
};

}  // namespace caffe

#endif  // CAFFE_CORAL_LOSS_LAYER_HPP_
