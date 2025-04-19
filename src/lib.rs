pub mod math;

use math::derivatives::*;
use math::sigmoid::*;

use rand::prelude::*;

use flashlight_tensor::prelude::*;

use std::fmt;

pub struct NeuralNetwork{
    pub layers: Vec<u32>,
    pub weights: Vec<Tensor<f32>>,
    pub biases: Vec<Tensor<f32>>,
}

impl NeuralNetwork{
    pub fn new(_layers: Vec<u32>, bias_range: f32, weight_range: f32) -> Self{

        let mut rng = rand::rng();

        let mut _weights: Vec<Tensor<f32>> = Vec::with_capacity(_layers.len()-1);
        let mut _biases: Vec<Tensor<f32>> = Vec::with_capacity(_layers.len()-1);

        for i in 1.._layers.len(){
            let mut weights_tensor: Tensor<f32> = Tensor::new(&[_layers[i-1], _layers[i]]);
            let mut biases_tensor: Tensor<f32> = Tensor::new(&[_layers[i], 1]);

            if i != _layers.len()-1{
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
            layers: _layers,
            biases: _biases,
            weights: _weights,
        }
    }
    pub fn forward_propagation(&self, input_data: &[f32]) -> Option<Tensor<f32>>{
        if input_data.len() != self.layers[0] as usize{
            return None;
        }

        let mut output_tensor: Tensor<f32> = Tensor::from_data(input_data, &[input_data.len() as u32, 1]).unwrap(); 
        
        for i in 1..self.layers.len(){
            

            let transposed_weights = self.weights[i-1].matrix_transpose().unwrap();
            let biases = &self.biases[i-1];
            
            println!("{}\n*\n{}\n+\n{}", transposed_weights.matrix_to_string().unwrap(), output_tensor.matrix_to_string().unwrap(), biases.matrix_to_string().unwrap());

            let multiplied_tensor = transposed_weights.matrix_mult(&output_tensor).unwrap();
    
            output_tensor = multiplied_tensor.iter_tens_add(&self.biases[i-1]).unwrap();
            
            println!("=\n{}", output_tensor.matrix_to_string().unwrap());

            for row in 0..output_tensor.get_sizes()[0]{
                for collumn in 0..output_tensor.get_sizes()[1]{
                    output_tensor.set(sigmoid(output_tensor.value(&[row, collumn]).unwrap().clone()), &[row, collumn]);
                }
            }

            println!("sigmoid\n{}", output_tensor.matrix_to_string().unwrap());

            println!("_______________________________________");

        }
        println!("Propagation finished");
        Some(output_tensor)
    }
}

///cost(y_hat(predicted answer), y(real_answer))
pub fn cost(y_hat: Tensor<f32>, y: Tensor<f32>) -> Option<f32>{
    if y_hat.get_sizes() != y.get_sizes(){
        return None;
    }

    let tensor_ones: Tensor<f32> = Tensor::fill(1.0, y_hat.get_sizes());
    
    //(y * log(y_hat))
    let y_log_y_hat: Tensor<f32> = y.iter_tens_mult(&y_hat.iter_log()).unwrap();
    //(1 - y)
    let negative_y: Tensor<f32> = tensor_ones.iter_tens_sub(&y).unwrap();
    //(1 - y_hat)
    let log_negative_y_hat: Tensor<f32> = tensor_ones.iter_tens_sub(&y_hat).unwrap().iter_log();

    let losses: Tensor<f32> = y_log_y_hat.iter_tens_add( &negative_y.iter_tens_mult(&log_negative_y_hat).unwrap() ).unwrap();

    let m: usize = y_hat.count_data();

    let const_multiplier = 1.0/m as f32;

    let mut summed_losses: f32 = 0.0;

    for i in 0..losses.get_sizes()[0]{
        summed_losses += const_multiplier * losses.matrix_row(i).unwrap().sum();
    }

    Some(-summed_losses)
}

impl fmt::Display for NeuralNetwork{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        
        let mut longest_layer: u32 = 0;

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
            let bias_col: Vec<f32> = self.biases[i].matrix_col(0).unwrap().get_data().to_vec();
            
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
