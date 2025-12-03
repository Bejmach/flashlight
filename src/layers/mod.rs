use flashlight_tensor::tensor::Tensor;
use async_trait::async_trait;

pub mod linear;
pub mod dropout;

pub mod activations;

pub trait Backend{
    fn forward(&mut self, input: Tensor<f32>){}
    fn backward(&mut self, grad_output: Tensor<f32>){}
}

pub struct Cpu;
pub struct Gpu;

/// Basic layer trait for layers
pub trait LayerCpu {
    /// Forward propagation for layer
    fn forward(&mut self, input: &Tensor<f32>) -> Tensor<f32>;
    /// Backward propagation for layer
    fn backward(&mut self, grad_output: &Tensor<f32>) -> Tensor<f32>;
}

#[async_trait]
pub trait LayerGpu {
    /// Forward propagation for layer
    async fn forward(&mut self, input: &Tensor<f32>) -> Tensor<f32>;
    /// Backward propagation for layer
    async fn backward(&mut self, grad_output: &Tensor<f32>) -> Tensor<f32>;
}
