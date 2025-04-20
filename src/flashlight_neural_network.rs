use crate::math::sigmoid::*;
use crate::math::derivatives::*;

use flashlight_tensor::prelude::*;
use rand::prelude::*;

pub struct NeuralNetwork{
    pub layers: Vec<u32>,
    pub weights: Vec<Tensor<f32>>,
    pub biases: Vec<Tensor<f32>>,
}

impl NeuralNetwork{
    /// Create new neural network with number of nodes on each layer and number of layers specyfied
    /// by _layers with bias and weight randomness between -range..range
    ///
    /// # Example 
    /// ```
    /// use flashlight::prelude::*;
    ///
    /// let nnetwork: NeuralNetwork = NeuralNetwork::new(vec!{2, 3, 3, 1}, 1.0, 1.0);
    /// //nnetwork = 
    /// //    0.0, 0.0
    /// //rand, rand, rand
    /// //rand, rand, rand
    /// //       0.0
   /// ```
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
    /// Get prediction of each layer in neural network based on &[f32] input data
    /// where data need to be of length of the input layer
    ///
    /// # Example
    /// ```
    /// use flashlight::prelude::*;
    ///
    /// let nnetwork: NeuralNetwork = NeuralNetwork::new(vec!{2, 3, 3, 1}, 1.0, 1.0);
    ///
    /// nnetwork.forward_propagation(&[50.0, 150.0]);
    ///
    /// //this is not a test. It is impossible to predict the output of naural network with random
    /// //weights
    /// ```
    pub fn full_forward_propagation(&self, input_data: Tensor<f32>) -> Option<Vec<Tensor<f32>>>{
        if input_data.get_sizes().len() != 2{
            return None;
        }
        if input_data.get_sizes()[1] != self.layers[0]{
            return None;
        }

        let mut output_tensor: Tensor<f32> = input_data;
        
        let mut all_outputs: Vec<Tensor<f32>> = Vec::with_capacity(self.layers.len()-1);

        for i in 1..self.layers.len(){
            

            let transposed_weights = self.weights[i-1].matrix_transpose().unwrap();
            let biases = &self.biases[i-1];
            
            println!("{}\n*\n{}\n+\n{}", transposed_weights.matrix_to_string().unwrap(), output_tensor.matrix_to_string().unwrap(), biases.matrix_to_string().unwrap());

            let multiplied_tensor = transposed_weights.matrix_mult(&output_tensor).unwrap();
    
            output_tensor = multiplied_tensor.tens_add(&self.biases[i-1]).unwrap();
            
            println!("=\n{}", output_tensor.matrix_to_string().unwrap());

            for row in 0..output_tensor.get_sizes()[0]{
                for collumn in 0..output_tensor.get_sizes()[1]{
                    output_tensor.set(sigmoid(output_tensor.value(&[row, collumn]).unwrap().clone()), &[row, collumn]);
                }
            }

            println!("sigmoid\n{}", output_tensor.matrix_to_string().unwrap());

            println!("_______________________________________");

            all_outputs.push(output_tensor.clone());

        }
        println!("Propagation finished");
        Some(all_outputs)
    }
}

/// get cost of neural network using y_hat(predicted answers) and y(real answers)
/// where each row of y and y_hat is one evaluation
///
/// # Example
/// ```
/// use flashlight::prelude::*;
///
/// let y_hat: Tensor<f32> = Tensor::fill(0.9, &[10, 1]);
/// let y: Tensor<f32> = Tensor::fill(0.1, &[10, 1]);
///
/// let network_cost = cost(y_hat, y).unwrap();
/// ```
pub fn cost(y_hat: Tensor<f32>, y: Tensor<f32>) -> Option<f32>{
    if y_hat.get_sizes() != y.get_sizes(){
        return None;
    }

    for i in 0..y_hat.get_data().len(){
        if y_hat.get_data()[i] > 1.0 || y_hat.get_data()[i] < 0.0 || y.get_data()[i] > 1.0 || y.get_data()[i] < 0.0{
            return None
        }  
    }

    let tensor_ones: Tensor<f32> = Tensor::fill(1.0, y_hat.get_sizes());
    
    //(y * log(y_hat))
    let y_log_y_hat: Tensor<f32> = y.tens_mult(&y_hat.log()).unwrap();
    //(1 - y)
    let negative_y: Tensor<f32> = tensor_ones.tens_sub(&y).unwrap();
    //log(1 - y_hat)
    let log_negative_y_hat: Tensor<f32> = tensor_ones.tens_sub(&y_hat).unwrap().log();

    let losses: Tensor<f32> = y_log_y_hat.tens_add( &negative_y.tens_mult(&log_negative_y_hat).unwrap() ).unwrap();

    let m: usize = y_hat.count_data();

    let const_multiplier = 1.0/m as f32;

    let mut summed_losses: f32 = 0.0;

    for i in 0..losses.get_sizes()[0]{
        summed_losses += const_multiplier * losses.matrix_row(i).unwrap().sum();
    }

    Some(-summed_losses)
}

/// get value of backpropagation on last layer of neural network using input data and y(real answers)
pub fn backprop_value(network: NeuralNetwork, input: Tensor<f32>, y: Tensor<f32>) -> Option<Tensor<f32>>{
    if input.get_sizes().len() != 2{
        return None;
    }
    if input.get_sizes()[1] != network.layers[0]{
        return None;
    }
    
    //A[0-L]
    let all_predictions = network.full_forward_propagation(input).unwrap();

    let tensor_ones: Tensor<f32> = Tensor::fill(1.0, all_predictions[all_predictions.len()-1].get_sizes());

    let m: usize = all_predictions[all_predictions.len()-1].count_data();
    
    let const_multiplier = 1.0 / m as f32;

    let part_1 = (all_predictions[all_predictions.len()-1].tens_sub(&y)).unwrap();

    let part_2 = all_predictions[all_predictions.len()-2].matrix_transpose().unwrap();

    let final_prod = part_1.matrix_mult(&part_2).unwrap().mult(const_multiplier);

    Some(final_prod)
}
