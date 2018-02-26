#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layers/EigenLog_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
namespace caffe {

template <typename Dtype>
void EigenLogLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  const int dim = bottom[0]->num();
 diff_.Reshape(dim, dim, 1, 1); 
 top[0]->Reshape(dim,dim,1,1);

  eig.Reshape(dim,1,1,1);
  U.Reshape(dim,dim,1,1);
  V.Reshape(dim,dim,1,1);
  eig_log.Reshape(dim,1,1,1);
  eig_inv.Reshape(dim,1,1,1);
  eig_matx.Reshape(dim,dim,1,1);
  eig_log_matx.Reshape(dim,dim,1,1);
  eig_inv_matx.Reshape(dim,dim,1,1);
  diff_sys.Reshape(dim,dim,1,1);
  dU.Reshape(dim,dim,1,1);
  deigen.Reshape(dim,dim,1,1);
  P.Reshape(dim,dim,1,1);
  PT.Reshape(dim,dim,1,1);
  sys.Reshape(dim,dim,1,1);
  P_sys.Reshape(dim,dim,1,1);
  temp1.Reshape(dim,dim,1,1);
  temp2.Reshape(dim,dim,1,1);
  iden_matx.Reshape(dim,dim,1,1);
  cov.Reshape(dim,dim,1,1);  
}

template <typename Dtype>
void EigenLogLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  const int dim = bottom[0]->num();
  

  
}

template <typename Dtype>
void EigenLogLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  int dim = bottom[0]->num();

  
  }

#ifdef CPU_ONLY
STUB_GPU(EigenLogLayer);
#endif

INSTANTIATE_CLASS(EigenLogLayer);
REGISTER_LAYER_CLASS(EigenLog);

}  // namespace caffe
