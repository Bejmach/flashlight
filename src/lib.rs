pub mod math;

use math::matrix::*;
use math::derivatives::*;
use math::sigmoid::*;

use rand::prelude::*;

use std::fmt;

pub struct NeuralNetwork{
    pub layers: Vec<usize>,
    pub weights: Vec<Matrix>,
    pub biases: Vec<Matrix>,
}

impl NeuralNetwork{
    pub fn new(_layers: Vec<usize>, bias_range: f32, weight_range: f32) -> Self{

        let mut rng = rand::rng();

        let mut _weights: Vec<Matrix> = Vec::with_capacity(_layers.len()-1);
        let mut _biases: Vec<Matrix> = Vec::with_capacity(_layers.len()-1);

        for i in 1.._layers.len(){
            let mut weights_matrix: Matrix = Matrix::new(_layers[i], _layers[i-1]);
            let mut biases_matrix: Matrix = Matrix::new(_layers[i], 1);

            if i != _layers.len()-1{
                for row in 0.._layers[i]{
                    biases_matrix.set(row, 0, rng.random_range(-bias_range..bias_range))
                }
            }
            for collumn in 0.._layers[i-1]{
                for row in 0.._layers[i]{
                    weights_matrix.set(row, collumn, rng.random_range(-weight_range..weight_range));
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
    pub fn forward_propagation(&self, input_data: Vec<f32>) -> Option<Vec<f32>>{
        if input_data.len() != self.layers[0]{
            return None;
        }

        let mut output_matrix: Matrix = Matrix::from_vec(vec![input_data.clone()]).transpose();
        
        for i in 1..self.layers.len(){
            
            println!("{}\n*\n{}\n+\n{}", self.weights[i-1].clone(), output_matrix, self.biases[i-1].clone());

            output_matrix = matrix_add(matrix_mult(self.weights[i-1].clone(), output_matrix).unwrap(), self.biases[i-1].clone()).unwrap().to_sigmoid();

            println!("=\n{}", output_matrix);

            println!("_______________________________________");
        }
        println!("Propagation finished");
        Some(output_matrix.col(0).unwrap())
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
            let bias_col: Vec<f32> = self.biases[i].col(0).unwrap();
            
            for _j in 0..layer_difference{
                return_string.push_str("   ");
            }
            for j in 0..bias_col.len(){
                let bias_value: String = format!("{:>5.2}", bias_col[j]);
                return_string.push_str(&(bias_value + " "));
                if j == bias_col.len()-1 && i != self.biases.len()-1{
                   return_string.push_str("\n\n"); 
                }
            }
        }

        write!(f, "{}", return_string)
    }
}
