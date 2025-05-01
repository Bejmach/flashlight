use crate::math::sigmoid::*;
use crate::math::derivatives::*;

use flashlight_tensor::prelude::*;
use rand::prelude::*;

pub trait Model{
    fn forward(&mut self, input: Tensor<f32>) -> Tensor<f32>;
    fn backward(&mut self, grad_output: Tensor<f32>);
}

pub struct FlashlightModel{
    pub layers: Vec<u32>,
    pub weights: Vec<Tensor<f32>>,
    pub biases: Vec<Tensor<f32>>,
}

pub fn xavier_weights(input_neurons: u32, output_neurons: u32) -> f32{
    (6.0/((input_neurons + output_neurons) as f32)).sqrt()
}

impl FlashlightModel{
    /// Create new neural network with number of nodes on each layer and number of layers specyfied
    /// by _layers with bias and weight randomness decided using xavier
    ///
    /// # Example 
    /// ```
    /// use flashlight::prelude::*;
    ///
    /// let model: FlashlightModel = FlashlightModel::new(&vec!{2, 3, 3, 1});
    /// //nnetwork = 
    /// //    0.0, 0.0
    /// //rand, rand, rand
    /// //rand, rand, rand
    /// //       0.0
   /// ```
    pub fn new(_layers: &[u32]) -> Self{

        let mut rng = rand::rng();

        let mut _weights: Vec<Tensor<f32>> = Vec::with_capacity(_layers.len()-1);
        let mut _biases: Vec<Tensor<f32>> = Vec::with_capacity(_layers.len()-1);

        let rand_range: f32 = (6.0/((_layers[0] + _layers[_layers.len()-1]) as f32)).sqrt();

        for i in 1.._layers.len(){
            let mut weights_tensor: Tensor<f32> = Tensor::new(&[_layers[i], _layers[i-1]]);
            let mut biases_tensor: Tensor<f32> = Tensor::new(&[_layers[i], 1]);

            if i != _layers.len(){
                for row in 0.._layers[i]{
                    biases_tensor.set(rng.random_range(-rand_range..rand_range), &[row, 0])
                }
            }
            for row in 0.._layers[i]{
                for collumn in 0.._layers[i-1]{
                    weights_tensor.set(rng.random_range(-rand_range..rand_range), &[row, collumn]);
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



