use rand::prelude::*;

pub struct Tensor<T>{
    pub data: Vec<T>,
    //x, y, z, ...
    sizes: Vec<u32>,
}

impl<T: Default + Clone> Tensor<T>{
    pub fn new(_sizes: Vec<u32>) -> Self{
        let mut total_size: u32 = 1;
        for i in 0.._sizes.len(){
            total_size *= _sizes[i];
        }
        
        Self{
            data: vec![T::default(); total_size as usize],
            sizes: _sizes,
        }
    }
    pub fn vector(&self, pos: Vec<u32>) -> Option<Tensor<T>>{
        let self_dimensions = self.sizes.len();
        let selector_dimensions = pos.len();
        if self_dimensions - selector_dimensions != 1{
            return None;
        }
        let mut data_begin: u32 = 0;

        let mut stride = self.sizes[..self.sizes.len()-1].iter().product::<u32>();

        for i in 0..pos.len() {
            data_begin += pos[i] * stride;
            if i + 1 < self.sizes.len() {
                stride /= self.sizes[i];
            }
        }

        let data_end: u32 = data_begin + self.sizes[0];

        Some(Tensor{
            data: self.data[data_begin as usize..data_end as usize].to_vec(),
            sizes: self.sizes[0..1].to_vec(),
        })
    }
}

impl Tensor<f32>{
    pub fn new_f32(_sizes: Vec<u32>) -> Self{
        let mut total_size: u32 = 1;
        for i in 0.._sizes.len(){
            total_size *= _sizes[i];
        }
        
        Self{
            data: vec![0.0; total_size as usize],
            sizes: _sizes,
        }
    }
    pub fn rand_f32(_sizes: Vec<u32>, rand_range: f32) -> Self{
        let mut rng = rand::rng();

        let mut total_size: u32 = 1;
        for i in 0.._sizes.len(){
            total_size *= _sizes[i];
        }

        let mut input_vector = Vec::with_capacity(total_size as usize);

        for _i in 0..total_size{
            input_vector.push(rng.random_range(-rand_range..rand_range));
        }
        
        Self{
            data: input_vector,
            sizes: _sizes,
        }
    }
}
