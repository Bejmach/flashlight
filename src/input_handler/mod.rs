use flashlight_tensor::prelude::*;

use rand::seq::SliceRandom;

pub struct InputPrePrepared{
    pub input_data: Vec<Tensor<f32>>,
    pub output_data: Vec<Tensor<f32>>,
    bach_size: u32,
}

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
        self.input_data.shuffle(&mut rng);
        self.output_data.shuffle(&mut rng);

        let mut input_tensor = self.input_data[0].clone();
        for i in 1..self.input_data.len(){
            input_tensor = input_tensor.append(&self.input_data[i]).unwrap();
        }

        let mut output_tensor = self.output_data[0].clone();
        for i in 1..self.output_data.len(){
            output_tensor = output_tensor.append(&self.output_data[i]).unwrap();
        }

        InputHandler{
            input_data: input_tensor,
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
        bach = bach.transform(&[1, bach.get_sizes()[0]]).unwrap();

        for i in 1..self.bach_size{
            let mut next_col = self.input_data.matrix_row(n*self.bach_size + i).unwrap();
            next_col = next_col.transform(&[1, next_col.get_sizes()[0]]).unwrap();
            bach = bach.append(&next_col).unwrap();
        }

        bach.matrix_transpose().unwrap()
    }
    pub fn output_bach(&self, n: u32) -> Tensor<f32>{
        let mut bach = self.output_data.matrix_row(n*self.bach_size).unwrap();
        bach = bach.transform(&[1, bach.get_sizes()[0]]).unwrap();

        for i in 1..self.bach_size{
            let mut next_col = self.output_data.matrix_row(n*self.bach_size + i).unwrap();
            next_col = next_col.transform(&[1, next_col.get_sizes()[0]]).unwrap();
            bach = bach.append(&next_col).unwrap();
        }

        bach.matrix_transpose().unwrap()
    }
}
