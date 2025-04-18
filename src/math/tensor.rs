use rand::prelude::*;

#[derive(Clone)]
pub struct Tensor<T>{
    pub data: Vec<T>,
    //..., z, y, x
    pub sizes: Vec<u32>,
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
    pub fn value(&self, pos: &[u32]) -> Option<&T>{
        let self_dimensions = self.sizes.len();
        let selector_dimensions = pos.len();
        if self_dimensions - selector_dimensions != 0{
            return None;
        }
        
        for i in 0..pos.len(){
            if pos[i] >= *self.sizes.get(i).unwrap(){
                return None;
            }
        }
        let mut index = 0;
        let mut stride = 1;
        for i in (0..self.sizes.len()).rev() {
            index += pos[i] * stride;
            stride *= self.sizes[i];
        }

        Some(&self.data[index as usize])
    }
    pub fn vector(&self, pos: &[u32]) -> Option<Tensor<T>>{
        let self_dimensions = self.sizes.len();
        let selector_dimensions = pos.len();
        if self_dimensions - selector_dimensions != 1{
            return None;
        }
        
        for i in 0..pos.len(){
            if pos[i] >= self.sizes[i]{
                return None;
            }
        }

        let mut data_begin: u32 = 0;

        let mut stride = self.sizes[0];

        for i in 0..pos.len() {
            data_begin += pos[pos.len() - 1 - i] * stride;
            stride *= self.sizes[1+i];
        }

        let data_end: u32 = data_begin + self.sizes.get(self.sizes.len()-1).unwrap();

        Some(Tensor{
            data: self.data[data_begin as usize..data_end as usize].to_vec(),
            sizes: self.sizes[self.sizes.len()-1..self.sizes.len()].to_vec(),
        })
    }

    pub fn matrix(&self, pos: &[u32]) -> Option<Tensor<T>>{
        let self_dimensions = self.sizes.len();
        let selector_dimensions = pos.len();
        if self_dimensions - selector_dimensions != 2{
            return None;
        }
        
        for i in 0..pos.len(){
            if pos[i] >= self.sizes[i]{
                return None;
            }
        }

        let mut data_begin: u32 = 0;

        let mut stride = self.sizes[1];

        for i in 0..pos.len() {
            data_begin += pos[pos.len() - 1 - i] * stride;
            stride *= self.sizes[2+i];
        }

        let prod: u32 = self.sizes[self.sizes.len()-2..].iter().product();
        let data_end: u32 = data_begin + prod;

        Some(Tensor{
            data: self.data[data_begin as usize..data_end as usize].to_vec(),
            sizes: self.sizes[self.sizes.len()-2..].to_vec(),
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

    pub fn dot_product(&self, tens2: &Tensor<f32>) -> Option<f32>{
        if self.sizes.len() != 1{
            return None;
        }
        if self.sizes != tens2.sizes{
            return None;
        }
        
        let mut dot: f32 = 0.0;
        print!("Dot: ");
        for i in 0..self.sizes[0] as u32{
            println!("{} - {}, {} ", i, self.value(&[i]).unwrap(), tens2.value(&[i]).unwrap());
            dot += self.value(&[i]).unwrap() * tens2.value(&[i]).unwrap();
        }

        Some(dot)
    }
}
