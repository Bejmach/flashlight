use flashlight_tensor::prelude::*;
use rand::Rng;

use crate::{layers::Layer, prelude::xavier_weights};

pub struct Linear{
    pub weights: Tensor<f32>,
    pub biases: Tensor<f32>,
    learning_rate: f32,
    input_cache: Option<Tensor<f32>>,
}

impl Linear{
     pub fn new(input_size: u32, output_size: u32, learning_rate: f32) -> Self{
        let rand_range = xavier_weights(input_size, output_size);

        let mut weight_data: Vec<f32> = Vec::with_capacity((input_size * output_size) as usize);
        let mut bias_data: Vec<f32> = Vec::with_capacity(output_size as usize);

        let mut rng = rand::rng();

        for i in 0..output_size{
            bias_data.push(rng.random_range(-rand_range..rand_range));
            for j in 0..input_size{
                weight_data.push(rng.random_range(-rand_range..rand_range));
            }
        }

        Self{
            weights: Tensor::from_data(&weight_data, &[output_size, input_size]).unwrap(),
            biases: Tensor::from_data(&bias_data, &[output_size, 1]).unwrap(),
            learning_rate,
            input_cache: None,
        }
    }
}

impl Layer for Linear{
   
    fn forward(&mut self, data: &Tensor<f32>) -> Tensor<f32>{

        self.input_cache = Some(data.clone());

        let weights_data = self.weights.matrix_mul(&data).unwrap();

        let bias_data = weights_data.matrix_transpose().unwrap().tens_broadcast_add(&self.biases).unwrap().matrix_transpose().unwrap();

        bias_data
    }

    fn backward(&mut self, grad_output: &Tensor<f32>) -> Tensor<f32>{
        if self.input_cache.is_none(){
            panic!("Program paniced!\nNo input cache in layer. Perform a forward pass before backward");
        }

        let weight_grad = grad_output.matrix_mul(&self.input_cache.clone().unwrap().matrix_transpose().unwrap()).unwrap();
    
        let bias_grad = grad_output.matrix_col_sum().unwrap().mul(1.0 / self.input_cache.clone().unwrap().get_sizes()[0] as f32);

        self.weights = self.weights.tens_sub(&weight_grad.mul(self.learning_rate)).unwrap();
        self.biases = self.biases.tens_sub(&bias_grad.mul(self.learning_rate)).unwrap();

        let input_grad = self.weights.matrix_transpose().unwrap().matrix_mul(grad_output).unwrap();

        input_grad
    }
}
