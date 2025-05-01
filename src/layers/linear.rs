use flashlight_tensor::prelude::*;
use rand::Rng;

pub struct Linear{
    weights: Tensor<f32>,
    biases: Tensor<f32>,
}

impl Linear{
    pub fn new(input_size: u32, output_size: u32, rand_range: f32) -> Self{
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
        }
    }
    pub fn forward(&self, data: Tensor<f32>) -> Tensor<f32>{
        let weights_data = self.weights.matrix_transpose().unwrap().matrix_mul(&data);

        let bias_data = self.weights.matrix_transpose().unwrap().tens_broadcast_add(&self.biases).unwrap().matrix_transpose().unwrap();

        bias_data
    }

    /*pub fn backward(&self, output_der: Tensor<f32>) -> Tensor<f32>{
        
    }*/
}
