use flashlight_tensor::tensor::Tensor;

pub mod linear;
pub mod dropout;

pub mod activations;

/// Basic layer trait for layers
pub trait Layer {
    /// Forward propagation for layer
    fn forward(&mut self, input: &Tensor<f32>) -> Tensor<f32>;
    /// Backward propagation for layer
    fn backward(&mut self, grad_output: &Tensor<f32>) -> Tensor<f32>;
}
