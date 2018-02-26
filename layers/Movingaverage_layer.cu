#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layers/Movingaverage_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"


namespace caffe {

template <typename Dtype>
void MOVINGAVERAGELayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
      const int count = bottom[0]->count();
      this->blobs_[2]->mutable_cpu_data()[0] = this->blobs_[2]->mutable_cpu_data()[0]+1;

       if(this->blobs_[2]->cpu_data()[0]==1){
       caffe_gpu_scale(count, Dtype(1), bottom[0]->gpu_data(),top[0]->mutable_gpu_data());
       LOG(INFO)<< "MA(1, 1)" << "first iteration keeps original figure";
}
   else {
        caffe_gpu_scal(count,Dtype(0.9),this->blobs_[0]->mutable_gpu_data());
      
        caffe_gpu_scale(count,Dtype(0.1),bottom[0]->gpu_data(),accumulation.mutable_gpu_data());
        
        caffe_gpu_add(count,this->blobs_[0]->gpu_data(),accumulation.gpu_data(),top[0]->mutable_gpu_data());
        
 }
caffe_gpu_scale(count, Dtype(1), top[0]->mutable_gpu_data(), this->blobs_[0]->mutable_gpu_data());      
}

template <typename Dtype>
void MOVINGAVERAGELayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    
     const int count = bottom[0]->count(); 
     caffe_gpu_scale(count, Dtype(0.1),top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(MOVINGAVERAGELayer);

}  // namespace caffe
