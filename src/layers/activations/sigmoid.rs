use flashlight_tensor::tensor::Tensor;

use crate::layers::Layer;

/// Sigmoid activation layer
pub struct Sigmoid{
    //change it to Opt<Vec<Ten<f32>>> to store all activations in layer, and use last element in
    //backprop, and pop it.
    input_cache: Option<Tensor<f32>>,
}

impl Sigmoid{
    /// Create new sigmoid activation layer with its own input_cache
    pub fn new() -> Self{
        Self{
            input_cache: None,
        }
    }
    /// Return gradient output from target: &Tensor<f32>
    pub fn grad_output(&self, target: &Tensor<f32>) -> Tensor<f32>{
        if self.input_cache.is_none(){
            panic!();
        }

        let sigmoid_out = self.input_cache.as_ref().unwrap().sigmoid();
        sigmoid_out.tens_sub(target).unwrap()
    }
}

impl Layer for Sigmoid{
    /// Forward propagation for sigmoid layer
    fn forward(&mut self, input: &Tensor<f32>) -> Tensor<f32> {
        self.input_cache = Some(input.clone());
        input.sigmoid()
    }
    /// Backpropagation for sigmoid layer, uses grad_output
    fn backward(&mut self, grad_output: &Tensor<f32>) -> Tensor<f32> {
        if self.input_cache.is_none(){
            panic!();
        }

        grad_output.tens_mul(&self.input_cache.clone().unwrap().sigmoid_der()).unwrap()
    }
}
