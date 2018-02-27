#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layers/EigenLog_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include<iostream>
#include<stdlib.h>
#include<stdio.h>
#include <cusolverDn.h>
#include <cuda_runtime_api.h>
#include <algorithm>

namespace caffe {

template <typename Dtype>
void EigenLogLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
      const int dim = bottom[0]->num();
 // --- CUDA solver initialization
    int work_size = 0;
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);
cusolverDnSgesvd_bufferSize(solver_handle, dim, dim, &work_size);
    // --- CUDA SVD execution
    float *work;
    int *devInfo;
    cudaMalloc(&devInfo, sizeof(int));
    cudaMalloc(&work, work_size * sizeof(float));

caffe_gpu_memcpy(bottom[0]->count() * sizeof(Dtype), bottom[0]->gpu_data(), cov.mutable_gpu_data());
cusolverDnSgesvd(solver_handle, 'A', 'A', (int)dim, (int)dim, (float *)cov.mutable_gpu_data(), (int)dim, (float *)eig.mutable_gpu_data(), (float *)U.mutable_gpu_data(), (int)dim,(float *)V.mutable_gpu_data(), (int)dim, work, work_size, NULL, devInfo);

cudaDeviceSynchronize();

cusolverDnDestroy(solver_handle);

int i; 
Dtype* eig_pointer=eig.mutable_cpu_data();
for(i=0;i<dim;i++){

eig_pointer[i] = eig_pointer[i]+0.001;
caffe_log(1., &eig_pointer[i], &eig_pointer[i]);

}    //eig: get the log value of eigenvalue
caffe_gpu_set(dim*dim, Dtype(0), eig_matx.mutable_gpu_data());
Dtype* eig_matx_pointer=eig_matx.mutable_cpu_data();

for(i=0;i<dim;i++){
    eig_matx_pointer[i*dim+i]=eig_pointer[i];}   // turn log(eigenvalue) into a matrix

caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,dim,dim,dim,1.,U.gpu_data(),eig_matx.gpu_data(),0.,eig_matx.mutable_gpu_data());

caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,dim,dim,dim,1.0,eig_matx.gpu_data(),U.gpu_data(),0.,top[0]->mutable_gpu_data());  

}

template <typename Dtype>
void EigenLogLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      int dim = bottom[0]->num();
 // --- CUDA solver initialization
    int work_size = 0;
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);
cusolverDnSgesvd_bufferSize(solver_handle, dim, dim, &work_size);
    // --- CUDA SVD execution
    float *work;
    int *devInfo;
    cudaMalloc(&devInfo, sizeof(int));
    cudaMalloc(&work, work_size * sizeof(float));

caffe_gpu_memcpy(bottom[0]->count() * sizeof(Dtype), bottom[0]->gpu_data(), cov.mutable_gpu_data());
cusolverDnSgesvd(solver_handle, 'A', 'A', (int)dim, (int)dim, (float *)cov.mutable_gpu_data(), (int)dim, (float *)eig.mutable_gpu_data(), (float *)U.mutable_gpu_data(), (int)dim,(float *)V.mutable_gpu_data(), (int)dim, work, work_size, NULL, devInfo);

cudaDeviceSynchronize();

cusolverDnDestroy(solver_handle);

int i,j;

Dtype* eig_pointer=eig.mutable_cpu_data();
Dtype* eig_log_pointer=eig_log.mutable_cpu_data();
Dtype* eig_inv_pointer=eig_inv.mutable_cpu_data();
    for(i=0;i<dim;i++){
eig_inv_pointer[i]=eig_pointer[i]; 
    if(eig_pointer[i]!=0){
eig_inv_pointer[i]=1./eig_pointer[i];}            //eig_inv_pointer: get the inverse of eigenvalue
eig_log_pointer[i]=eig_pointer[i];
  
eig_pointer[i]= eig_pointer[i]+0.001;
caffe_log(1., &eig_pointer[i], &eig_log_pointer[i]);     //eig_log_pointer: get the log of eigenvalue

}  


caffe_gpu_set(dim*dim, Dtype(0), eig_matx.mutable_gpu_data());  
caffe_gpu_set(dim*dim, Dtype(0), eig_log_matx.mutable_gpu_data());
caffe_gpu_set(dim*dim, Dtype(0), eig_inv_matx.mutable_gpu_data());    
caffe_gpu_set(dim*dim, Dtype(0), iden_matx.mutable_gpu_data());    
Dtype* eig_matx_pointer=eig_matx.mutable_cpu_data();
Dtype* eig_log_matx_pointer=eig_log_matx.mutable_cpu_data();
Dtype* eig_inv_matx_pointer=eig_inv_matx.mutable_cpu_data();    
Dtype* iden_matx_pointer=iden_matx.mutable_cpu_data();

for(i=0;i<dim;i++){
eig_matx_pointer[i*dim+i]=eig_pointer[i];     //eig_matx: get eigenvalue matrix

eig_log_matx_pointer[i*dim+i]=eig_log_pointer[i];     //eig_log_matx: get log(eigenvalue) matrix

eig_inv_matx_pointer[i*dim+i]=eig_inv_pointer[i];     //eig_inv_pointer: get the inverse of eigenval matrix
iden_matx_pointer[i*dim+i]=1.;       //iden_matrix: get identity matrix
    }

caffe_gpu_gemm<Dtype>(CblasTrans,CblasNoTrans,dim,dim,dim,1.,U.gpu_data(),iden_matx.gpu_data(),0.,U.mutable_gpu_data());

caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,dim,dim,dim,0.5,top[0]->gpu_diff(),iden_matx.gpu_data(),0.,diff_sys.mutable_gpu_data());
caffe_gpu_gemm<Dtype>(CblasTrans,CblasNoTrans,dim,dim,dim,0.5,top[0]->gpu_diff(),iden_matx.gpu_data(),1.,diff_sys.mutable_gpu_data());   // diff_sys: sys(top[0]->difference)

caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,dim,dim,dim,2.,diff_sys.gpu_data(),U.gpu_data(),0.,dU.mutable_gpu_data());

caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,dim,dim,dim,1.,dU.gpu_data(),eig_log_matx.gpu_data(),0.,dU.mutable_gpu_data());     //dU

caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,dim,dim,dim,1.,eig_inv_matx.gpu_data(),U.gpu_data(),0.,deigen.mutable_gpu_data());

caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,dim,dim,dim,1.,deigen.gpu_data(),diff_sys.gpu_data(),0.,deigen.mutable_gpu_data());

caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,dim,dim,dim,1.,deigen.gpu_data(),U.gpu_data(),0.,deigen.mutable_gpu_data());  //deigen

Dtype* deigen_pointer=deigen.mutable_cpu_data();

caffe_gpu_set(dim*dim, Dtype(0), P.mutable_gpu_data());
Dtype* P_pointer=P.mutable_cpu_data();
      for(i=0;i<dim;i++){
      for(j=0;j<dim;j++){
      if(i!=j){
      if(eig_pointer[i]!=eig_pointer[j]){
      P_pointer[i*dim+j]=1./(eig_pointer[i]-eig_pointer[j]);    //P
      }}
      }}

caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,dim,dim,dim,1.,P.mutable_gpu_data(),iden_matx.gpu_data(),0.,PT.mutable_gpu_data());     //PT: P(transpose)

caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,dim,dim,dim,1.,U.gpu_data(),dU.gpu_data(),0.,sys.mutable_gpu_data());

//caffe_gpu_gemm<Dtype>(CblasTrans,CblasNoTrans,dim,dim,dim,0.5,sys.gpu_data(),iden_matx.gpu_data(),0.5,sys.mutable_gpu_data());   //sys: sys(U(transpose)*dU)
      
      
caffe_gpu_mul(dim*dim,PT.gpu_data(),sys.gpu_data(),P_sys.mutable_gpu_data());

caffe_gpu_gemm<Dtype>(CblasTrans,CblasNoTrans,dim,dim,dim,0.5,P_sys.gpu_data(),iden_matx.gpu_data(),0.5,P_sys.mutable_gpu_data());

      for(i=0;i<dim;i++){
      for(j=0;j<dim;j++){
      if(i!=j){
      deigen_pointer[i*dim+j]=0.;
      } }}

caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,dim,dim,dim,1.,U.gpu_data(),P_sys.gpu_data(),0.,temp1.mutable_gpu_data());

caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,dim,dim,dim,1.0,temp1.gpu_data(),U.gpu_data(),0.,temp1.mutable_gpu_data());  //temp1/////////////////// 1.

caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,dim,dim,dim,1.,U.gpu_data(),deigen.gpu_data(),0.,temp2.mutable_gpu_data());

caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,dim,dim,dim,1.0,temp2.gpu_data(),U.gpu_data(),0.,temp2.mutable_gpu_data());    //temp2////////////////// 1.

caffe_gpu_add(dim*dim,temp1.gpu_data(),temp2.gpu_data(),bottom[0]->mutable_gpu_diff());

}

INSTANTIATE_LAYER_GPU_FUNCS(EigenLogLayer);

}  // namespace caffe

