use flashlight_tensor::tensor::Tensor;

pub mod linear;
pub mod dropout;

pub mod activations;

pub trait Layer {
    fn forward(&mut self, input: &Tensor<f32>) -> Tensor<f32>;
    fn backward(&mut self, grad_output: &Tensor<f32>) -> Tensor<f32>;
}
