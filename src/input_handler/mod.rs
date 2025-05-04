use flashlight_tensor::prelude::*;

use rand::seq::SliceRandom;

pub struct Sample{
    pub input_data: Tensor<f32>,
    pub output_data: Tensor<f32>,
}

/// Work in progress, will be made for like 0.0.15 hopefully
pub struct DataPreparaton{
    pub data: Vec<Sample>,
    bach_size: u32,
}

/// Work in progress, will be made for like 0.0.15 hopefully
pub struct DataHandler{
    input_data: Tensor<f32>,
    output_data: Tensor<f32>,
    bach_size: u32,
}

impl DataPreparaton{
    pub fn new() -> Self{
        Self{
            data: Vec::new(),
            bach_size: 1,
        }
    }
    pub fn set_bach_size(&mut self, _bach_size: u32){
        if self.data.len() % _bach_size as usize == 0 {
            self.bach_size = _bach_size;
        }
    }
    pub fn append(&mut self, input_sample: &Tensor<f32>, output_sample: &Tensor<f32>){
        self.data.push(Sample{
            input_data: input_sample.clone(),
            output_data: output_sample.clone(),
        });
        self.bach_size = 1;
    }

    pub fn to_handler(&mut self) -> DataHandler{
        let mut rng = rand::rng();

        self.data.shuffle(&mut rng);

        let mut input_tensor = self.data[0].input_data.clone();
        for i in 1..self.data.len(){
            input_tensor = input_tensor.append(&self.data[i].input_data).unwrap();
        }

        let mut normalized_data: Vec<f32> = Vec::with_capacity(input_tensor.count_data());

        //collums holds same element in each sample, so I use normalization across collumns
        for i in 0..input_tensor.get_sizes()[1]{
            let input_col = input_tensor.matrix_col(i).unwrap();

            let col_mean: f32 = input_col.sum() / input_col.count_data() as f32;

            let mut col_std_dev: f32 = 0.0;
            for i in 0..input_col.count_data(){
                let diff = input_col.get_data()[i] - col_mean;
                col_std_dev += diff * diff;
            }
            col_std_dev = (col_std_dev/input_col.count_data() as f32).sqrt();

            if col_std_dev < 1e-8 {
                col_std_dev = 1e-8; 
            }

            for j in 0..input_col.count_data(){
                normalized_data.push((input_col.get_data()[j] - col_mean) / col_std_dev);
            }
        }

        let normalized_tensor: Tensor<f32> = Tensor::from_data(&normalized_data, &[input_tensor.get_sizes()[1], input_tensor.get_sizes()[0]]).unwrap().matrix_transpose().unwrap();

        let mut output_tensor = self.data[0].output_data.clone();
        for i in 1..self.data.len(){
            output_tensor = output_tensor.append(&self.data[i].output_data).unwrap();
        }

        DataHandler{
            input_data: normalized_tensor,
            output_data: output_tensor,
            bach_size: self.bach_size,
        }
    }
}

impl DataHandler{

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
