use crate::math::sigmoid::*;
use crate::math::derivatives::*;

use flashlight_tensor::prelude::*;
use rand::prelude::*;

pub struct Model{
    pub layers: Vec<u32>,
    pub weights: Vec<Tensor<f32>>,
    pub biases: Vec<Tensor<f32>>,
}

impl Model{
    /// Create new neural network with number of nodes on each layer and number of layers specyfied
    /// by _layers with bias and weight randomness between -range..range
    ///
    /// # Example 
    /// ```
    /// use flashlight::prelude::*;
    ///
    /// let model: Model = Model::new(&vec!{2, 3, 3, 1}, 1.0, 1.0);
    /// //nnetwork = 
    /// //    0.0, 0.0
    /// //rand, rand, rand
    /// //rand, rand, rand
    /// //       0.0
   /// ```
    pub fn new(_layers: &[u32], bias_range: f32, weight_range: f32) -> Self{

        let mut rng = rand::rng();

        let mut _weights: Vec<Tensor<f32>> = Vec::with_capacity(_layers.len()-1);
        let mut _biases: Vec<Tensor<f32>> = Vec::with_capacity(_layers.len()-1);

        for i in 1.._layers.len(){
            let mut weights_tensor: Tensor<f32> = Tensor::new(&[_layers[i-1], _layers[i]]);
            let mut biases_tensor: Tensor<f32> = Tensor::new(&[_layers[i], 1]);

            if i != _layers.len(){
                for row in 0.._layers[i]{
                    biases_tensor.set(rng.random_range(-bias_range..bias_range), &[row, 0])
                }
            }
            for row in 0.._layers[i-1]{
                for collumn in 0.._layers[i]{
                    weights_tensor.set(rng.random_range(-weight_range..weight_range), &[row, collumn]);
                }
            }
            _weights.push(weights_tensor);
            _biases.push(biases_tensor);
        }

        Self{
            layers: _layers.to_vec(),
            biases: _biases,
            weights: _weights,
        }
    }
}



