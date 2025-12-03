use flashlight_tensor::tensor::Tensor;

use crate::layers::LayerCpu;

/// Sigmoid activation layer
pub struct Sigmoid{
    //change it to Opt<Vec<Ten<f32>>> to store all activations in layer, and use last element in
    //backprop, and pop it.
    input_cache: Vec<Tensor<f32>>,
}

impl Sigmoid{
    /// Create new sigmoid activation layer with its own input_cache
    pub fn new() -> Self{
        Self{
            input_cache: Vec::new(),
        }
    }
    /// Return gradient output from target: &Tensor<f32>
    pub fn grad_output(&self, target: &Tensor<f32>) -> Tensor<f32>{
        if self.input_cache.is_empty(){
            panic!();
        }

        let sigmoid_out = self.input_cache.last().unwrap().sigmoid();
        sigmoid_out.tens_sub(target).unwrap()
    }
}

impl LayerCpu for Sigmoid{
    /// Forward propagation for sigmoid layer
    fn forward(&mut self, input: &Tensor<f32>) -> Tensor<f32> {
        self.input_cache.push(input.clone());
        input.sigmoid()
    }
    /// Backpropagation for sigmoid layer, uses grad_output
    fn backward(&mut self, grad_output: &Tensor<f32>) -> Tensor<f32> {
        if self.input_cache.is_empty(){
            panic!();
        }

        self.input_cache.pop().unwrap().sigmoid_der().tens_broadcast_mul(grad_output).unwrap()
    }
}
