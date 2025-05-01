use flashlight_tensor::tensor::Tensor;

use crate::layers::Layer;

pub struct Sigmoid{
    input_cache: Option<Tensor<f32>>,
}

impl Sigmoid{
    pub fn new() -> Self{
        Self{
            input_cache: None,
        }
    }
    pub fn grad_output(&self, target: &Tensor<f32>) -> Tensor<f32>{
        if self.input_cache.is_none() || target.get_sizes()!=self.input_cache.clone().unwrap().get_sizes(){
            panic!();
        }

        let y_a = target.tens_div(&self.input_cache.clone().unwrap()).unwrap();
        let one_y_a = target.mul(-1.0).add(1.0).tens_div(&self.input_cache.clone().unwrap().mul(-1.0).add(1.0)).unwrap();

        y_a.tens_sub(&one_y_a).unwrap().mul(-1.0)
        
    }
}

impl Layer for Sigmoid{
    fn forward(&mut self, input: &Tensor<f32>) -> Tensor<f32> {
        self.input_cache = Some(input.clone());
        input.sigmoid()
    }
    fn backward(&mut self, grad_output: &Tensor<f32>) -> Tensor<f32> {
        if self.input_cache.is_none(){
            panic!();
        }

        grad_output.tens_mul(&self.input_cache.clone().unwrap().sigmoid_der()).unwrap()
    }
}
