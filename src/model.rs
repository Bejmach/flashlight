use flashlight_tensor::prelude::*;
use rand::prelude::*;

/// Basic trait for models
pub trait Model{
    /// Forward propagation for model, returns Tensor<f32> on output
    fn forward(&mut self, input: Tensor<f32>) -> Tensor<f32>;
    /// Backward propagation for model, uses gradient_output, taken from last_activation.grad_output(target: &Tensor<f32>)
    fn backward(&mut self, grad_output: Tensor<f32>);
}

/// Returns xavier weights based on input and output neurons in layer
pub fn xavier_weights(input_neurons: u32, output_neurons: u32) -> f32{
    (6.0/((input_neurons + output_neurons) as f32)).sqrt()
}


