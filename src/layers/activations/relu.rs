use flashlight_tensor::tensor::Tensor;

use crate::layers::Layer;

pub struct Relu{
    input_cache: Option<Tensor<f32>>,
}

impl Relu{
    pub fn new() -> Self{
        Self{
            input_cache: None,
        }
    }
}

impl Layer for Relu{
    fn forward(&mut self, input: &Tensor<f32>) -> Tensor<f32> {
        self.input_cache = Some(input.clone());
        input.relu()
    }
    fn backward(&mut self, grad_output: &Tensor<f32>) -> Tensor<f32> {
        if self.input_cache.is_none(){
            panic!();
        }

        self.input_cache.clone().unwrap().relu_der().matrix_transpose().unwrap().tens_broadcast_mul(grad_output).unwrap().matrix_transpose().unwrap()
    }
}
