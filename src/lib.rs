pub mod math;

use math::matrix::*;
use math::derivatives::*;

use rand::prelude::*;

use std::fmt;

pub struct NeuralNetwork{
    pub layers: Vec<usize>,
    pub weights: Vec<Matrix>,
    pub biases: Vec<Matrix>,
}

impl NeuralNetwork{
    pub fn new(_layers: Vec<usize>) -> Self{

        let mut rng = rand::rng();

        let mut _weights: Vec<Matrix> = Vec::with_capacity(_layers.len()-1);
        let mut _biases: Vec<Matrix> = Vec::with_capacity(_layers.len()-1);

        for i in 1.._layers.len(){
            let mut weights_matrix: Matrix = Matrix::new(_layers[i-1], _layers[i]);
            let mut biases_matrix: Matrix = Matrix::new(_layers[i], 1);
            for j in 0.._layers[i]{
                biases_matrix.set(j, 0, rng.random_range(0.0..10.0))
            }
            for j in 0.._layers[i-1]{
                for k in 0.._layers[i]{
                    weights_matrix.set(j, k, rng.random_range(0.0..10.0));
                }
            }
            _weights.push(weights_matrix);
            _biases.push(biases_matrix);
        }

        Self{
            layers: _layers,
            biases: _biases,
            weights: _weights,
        }
    }
}

impl fmt::Display for NeuralNetwork{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        
        let mut longest_layer: usize = 0;

        for var in self.layers.iter(){
            if var>&longest_layer{
                longest_layer = *var;
            }
        }

        let mut return_string: String = String::new();
        
        let layer_difference = longest_layer - self.layers[0];

        for _i in 0..layer_difference{
            return_string.push_str("   ");
        }
        for i in 0..self.layers[0]{
            return_string.push_str(" 0.00 ");
            if i == self.layers[0]-1{
                return_string.push_str("\n\n");
            }
        }

        for i in 0..self.biases.len(){    
            let layer_difference = longest_layer - self.layers[i+1];
            let bias_row: Vec<f32> = self.biases[i].row(0).unwrap();
            
            for _j in 0..layer_difference{
                return_string.push_str("   ");
            }
            for j in 0..bias_row.len(){
                let bias_value: String = format!("{:>5.2}", bias_row[j]);
                return_string.push_str(&(bias_value + " "));
                if j == bias_row.len()-1 && i != self.biases.len()-1{
                   return_string.push_str("\n\n"); 
                }
            }
        }

        write!(f, "{}", return_string)
    }
}
