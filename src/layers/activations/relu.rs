use flashlight_tensor::tensor::Tensor;

use crate::layers::LayerCpu;

/// ReLU activation layer
pub struct Relu{
    //change it to Opt<Vec<Ten<f32>>> to store all activations in layer, and use last element in
    //backprop, and pop it.
    input_cache: Vec<Tensor<f32>>,
}

impl Relu{
    /// Create new relu activation layer with its own input_cache
    pub fn new() -> Self{
        Self{
            input_cache: Vec::new(),
        }
    }
}

impl LayerCpu for Relu{
    /// Forward propagation for sigmoid layer
    fn forward(&mut self, input: &Tensor<f32>) -> Tensor<f32> {
        self.input_cache.push(input.clone());
        input.relu()
    }
    /// Backpropagation for sigmoid layer, uses grad_output
    fn backward(&mut self, grad_output: &Tensor<f32>) -> Tensor<f32> {
        if self.input_cache.is_empty(){
            panic!();
        }

        self.input_cache.pop().unwrap().relu_der().tens_broadcast_mul(grad_output).unwrap()
    }
}
