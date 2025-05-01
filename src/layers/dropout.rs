use flashlight_tensor::tensor::Tensor;
use rand::{rng, Rng};
/// Dropout currently not implemented properly, dont use
pub struct Dropout{
    dropout: f32,
}

impl Dropout{
    /// Create a dropout layer, with a limit on how much neurons can be nullyfied, between 0-1
    /// None if value outside range
    ///
    /// # Example
    ///
    /// ```
    /// use flashlight::prelude::*;
    ///
    /// //let dropout = Dropout::new(0.5).unwrap();
    /// ```
    pub fn new(dropout: f32) -> Option<Self>{
        if dropout>1.0 || dropout < 0.0{
            return None;
        }

        Some(Self{dropout})
    }

    /// set up to all_nurown*dropout_value values in tensor to 0
    ///
    /// # Example
    ///
    /// ```
    /// use flashlight::prelude::*;
    /// use flashlight_tensor::prelude::*;
    ///
    /// //let dropout = Dropout::new(0.5).unwrap();
    /// //let tensor = Tensor::from_data(&[1.0, 2.0, 3.0], &[3])
    ///
    /// //let x = dropout.forward(tensor);
    /// ```
    pub fn forward(&self, tensor: Tensor<f32>) -> Tensor<f32>{
        let mut tensor_copy = tensor.clone();

        let mut rng = rand::rng();
        let neuron_size = tensor_copy.get_sizes()[1];

        let dropped_neurons = ((neuron_size as f32) * self.dropout) as u32;

        for sample in 0..tensor_copy.get_sizes()[0]{
            for i in 0..dropped_neurons{
                let rand_neuron = rng.random_range(0..neuron_size);
                
                tensor_copy.set(0.0, &[sample, rand_neuron]);
            }
        }
        tensor_copy
    }
}
