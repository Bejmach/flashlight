use flashlight_tensor::prelude::*;

use rand::seq::SliceRandom;

/// Work in progress, will be made for like 0.0.15 hopefully
pub struct InputPrePrepared{
    pub input_data: Vec<Tensor<f32>>,
    pub output_data: Vec<Tensor<f32>>,
    bach_size: u32,
}

/// Work in progress, will be made for like 0.0.15 hopefully
pub struct InputHandler{
    input_data: Tensor<f32>,
    output_data: Tensor<f32>,
    bach_size: u32,
}

impl InputPrePrepared{
    pub fn new(input_sample: &Tensor<f32>, output_sample: &Tensor<f32>) -> Self{
        Self{
            input_data: vec!{input_sample.clone()},
            output_data: vec!{output_sample.clone()},
            bach_size: 1,
        }
    }
    pub fn set_bach_size(&mut self, _bach_size: u32){
        if self.input_data.len() % _bach_size as usize == 0 {
            self.bach_size = _bach_size;
        }
    }
    pub fn append(&mut self, input_sample: &Tensor<f32>, output_sample: &Tensor<f32>){
        self.input_data.push(input_sample.clone());
        self.output_data.push(output_sample.clone());
        self.bach_size = 1;
    }

    pub fn to_handler(&mut self) -> InputHandler{
        let mut rng = rand::rng();

        let mut input_tensor = self.input_data[0].clone();
        for i in 1..self.input_data.len(){
            input_tensor = input_tensor.append(&self.input_data[i]).unwrap();
        }

        let input_mean: f32 = input_tensor.sum() / input_tensor.count_data() as f32;

        let mut input_std_dev: f32 = 0.0;
        for i in 0..input_tensor.get_data().len(){
            input_std_dev += input_tensor.get_data()[i].powi(2);
        }
        input_std_dev = (input_std_dev/input_tensor.count_data() as f32).sqrt();

        let mut normalized_vec: Vec<f32> = Vec::with_capacity(input_tensor.count_data());
        for i in 0..input_tensor.get_data().len(){
            normalized_vec.push((input_tensor.get_data()[i] - input_mean) / input_std_dev);
        }

        let normalized_tensor: Tensor<f32> = Tensor::from_data(&normalized_vec, &input_tensor.get_sizes()).unwrap();

        let mut output_tensor = self.output_data[0].clone();
        for i in 1..self.output_data.len(){
            output_tensor = output_tensor.append(&self.output_data[i]).unwrap();
        }

        InputHandler{
            input_data: normalized_tensor,
            output_data: output_tensor,
            bach_size: self.bach_size,
        }
    }
}

impl InputHandler{

    pub fn len(&self) -> u32{
        self.input_data.get_sizes()[0] / self.bach_size
    }
    pub fn input_bach(&self, n: u32) -> Tensor<f32>{
        let mut bach = self.input_data.matrix_row(n*self.bach_size).unwrap();

        for i in 1..self.bach_size{
            let mut next_col = self.input_data.matrix_row(n*self.bach_size + i).unwrap();
            bach = bach.append(&next_col).unwrap();
        }

        bach.matrix_transpose().unwrap()
    }
    pub fn output_bach(&self, n: u32) -> Tensor<f32>{
        let mut bach = self.output_data.matrix_row(n*self.bach_size).unwrap();

        for i in 1..self.bach_size{
            let mut next_col = self.output_data.matrix_row(n*self.bach_size + i).unwrap();
            bach = bach.append(&next_col).unwrap();
        }

        bach.matrix_transpose().unwrap()
    }
}
