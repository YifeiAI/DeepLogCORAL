#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layers/Movingaverage_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
namespace caffe {
template <typename Dtype>

void MOVINGAVERAGELayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  this->blobs_.resize(3);
  vector<int> sz;
  sz.push_back(bottom[0]->count()*bottom[0]->count());
  this->blobs_[0].reset(new Blob<Dtype>(sz));
  this->blobs_[1].reset(new Blob<Dtype>(sz));
  sz[0]=1;
  this->blobs_[2].reset(new Blob<Dtype>(sz));
  for (int i = 0; i<3; ++i){
  caffe_set(this->blobs_[i]->count(), Dtype(0), this->blobs_[i]->mutable_cpu_data());
  if(this->layer_param_.param_size()==i){
  ParamSpec* fixed_param_spec = this->layer_param_.add_param();
  fixed_param_spec->set_lr_mult(0.f);  
}
else{CHECK_EQ(this->layer_param_.param(i).lr_mult(),0.f)
      << "Cannot configure batch norm statistics as layer"
      << "parameters.";
}
}
}

template <typename Dtype>
void MOVINGAVERAGELayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //Layer<Dtype>::Reshape(bottom, top);
  // CHECK_EQ might not be necessary as the size of the covariance matrices of
  //source and target are always the same even if the batch size is different.
 
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int dim = count / num;
  top[0]->Reshape(bottom[0]->shape(0),bottom[0]->shape(1),bottom[0]->shape(2),bottom[0]->shape(3));
  accumulation.Reshape(bottom[0]->shape(0),bottom[0]->shape(1),bottom[0]->shape(2),bottom[0]->shape(3));

}

template <typename Dtype>
void MOVINGAVERAGELayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void MOVINGAVERAGELayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  }

#ifdef CPU_ONLY
STUB_GPU(MOVINGAVERAGELayer);
#endif

INSTANTIATE_CLASS(MOVINGAVERAGELayer);
REGISTER_LAYER_CLASS(MOVINGAVERAGE);

}  // namespace caffe
