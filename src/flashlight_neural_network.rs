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
    /// let input_data: Tensor<f32> = Tensor::from_data(&[50.0, 150.0, 100.0, 220.0], &[2,
    /// 2]).unwrap().matrix_transpose().unwrap();
    /// nnetwork.full_forward_propagation(&input_data).unwrap();
    ///
    /// //this is not a test. It is impossible to predict the output of naural network with random
    /// //weights
    /// ```
    pub fn full_forward_propagation(&self, input_data: &Tensor<f32>) -> Option<Vec<Tensor<f32>>>{
        if input_data.get_sizes().len() != 2{
            return None;
        }
        if input_data.get_sizes()[0] != self.layers[0]{
            return None;
        }

        let mut output_tensor: Tensor<f32> = input_data.clone();
        
        let mut all_outputs: Vec<Tensor<f32>> = Vec::with_capacity(self.layers.len()-1);

        for i in 1..self.layers.len(){
            

            let transposed_weights = self.weights[i-1].matrix_transpose().unwrap();
            let biases = &self.biases[i-1];
            
            println!("{}\n*\n{}\n=", transposed_weights.matrix_to_string().unwrap(), output_tensor.matrix_to_string().unwrap());

            let multiplied_tensor = transposed_weights.matrix_mult(&output_tensor).unwrap();
    
            println!("{}\n_________", multiplied_tensor.matrix_to_string().unwrap());
            println!("{}\n+\n{}", multiplied_tensor.matrix_to_string().unwrap(), biases.matrix_to_string().unwrap());

            output_tensor = multiplied_tensor.tens_broadcast_add(&biases).unwrap();
            
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

/// get tensor filled with data from input tensor transposed by sigmoid function
///
/// # Example
/// ```
/// use flashlight::prelude::*;
///
/// let tensor = Tensor::fill(100.0, &[10,1]);
///
/// let sigmoid_tens = tensor_to_sigmoid(&tensor);
///
/// assert_eq!(sigmoid_tens.get_data(), &vec!{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
/// ```
pub fn tensor_to_sigmoid(tensor: &Tensor<f32>) -> Tensor<f32>{
    let mut data_vector: Vec<f32> = Vec::with_capacity(tensor.get_data().len());
    for i in 0..tensor.get_data().len(){
        data_vector.push(sigmoid(tensor.get_data()[i]));
    }
    let sigmoid_input: Tensor<f32> = Tensor::from_data(&data_vector, tensor.get_sizes()).unwrap();

    sigmoid_input
}

pub struct BackpropagationValues{
    pub weights: Vec<Tensor<f32>>,
    pub biases: Vec<Tensor<f32>>,
}

/// get value of backpropagation on each layer of neural network using input data and y(real answers)
/// 
/// # Example
/// ```
/// use flashlight::prelude::*;
///
/// let nnetwork: NeuralNetwork = NeuralNetwork::new(vec!{2, 3, 3, 1}, 1.0, 1.0);
///
/// let input_data: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap().matrix_transpose().unwrap();
/// let predicted_data: Tensor<f32> = Tensor::from_data(&[3.0, 7.0, 11.0], &[1, 3]).unwrap();
///
/// let backpropagation_values = backprop_values(&nnetwork, &input_data, &predicted_data).unwrap();
/// ```
pub fn backprop_values(network: &NeuralNetwork, input: &Tensor<f32>, y: &Tensor<f32>) -> Option<BackpropagationValues>{
    if input.get_sizes().len() != 2{
        println!("input not in matrix");
        return None;
    }
    if input.get_sizes()[0] != network.layers[0]{
        println!("input does not match network input: {}, {}", input.get_sizes()[0], network.layers[0]);
        return None;
    }

    let real_y = tensor_to_sigmoid(&y);
    
    //A[0-L]
    let all_predictions = network.full_forward_propagation(input).unwrap();

    let tensor_ones: Tensor<f32> = Tensor::fill(1.0, all_predictions[all_predictions.len()-1].get_sizes());

    let m: usize = all_predictions[all_predictions.len()-1].count_data();
    
    let const_multiplier = 1.0 / m as f32;

    //weights
    println!("{}\n-\n{}", all_predictions[all_predictions.len()-1].matrix_to_string().unwrap(), real_y.matrix_to_string().unwrap());
    let part_1 = (all_predictions[all_predictions.len()-1].tens_sub(&real_y)).unwrap();
    let part_2 = all_predictions[all_predictions.len()-2].matrix_transpose().unwrap();

    println!("{}\n\n{}", part_1.matrix_to_string().unwrap(), part_2.matrix_to_string().unwrap());
    let weight_last = part_1.matrix_mult(&part_2).unwrap().mult(const_multiplier);

    //biases
    let mut bias_last: Tensor<f32> = Tensor::fill(0.0, all_predictions[all_predictions.len()-1].get_sizes());

    for i in 0..all_predictions[all_predictions.len()-1].get_sizes()[0]{
        let prediction_batch = all_predictions[all_predictions.len()-1].matrix_row(i).unwrap()
            .tens_sub(&y.matrix_row(i).unwrap()).unwrap();
        bias_last = bias_last.tens_add(&prediction_batch).unwrap();
    }

    bias_last = bias_last.mult(const_multiplier);

    //propagator
    let weitht_transpose = network.weights[network.weights.len()-1].matrix_transpose().unwrap();
    let mut delta = all_predictions[all_predictions.len()-1].tens_sub(&real_y).unwrap().mult(const_multiplier);

    let mut full_weights: Vec<Tensor<f32>> = Vec::with_capacity(network.layers.len()-1);
    let mut full_biases: Vec<Tensor<f32>> = Vec::with_capacity(network.layers.len()-1);

    full_weights.push(weight_last.clone());
    full_biases.push(bias_last.clone());

    for i in (1..network.layers.len()-1).rev(){
        let delta_part_1 = network.weights[i+1].matrix_transpose().unwrap().matrix_mult(&delta).unwrap();
        let delta_part_2 = all_predictions[i].tens_mult(&tensor_ones.tens_sub(&all_predictions[i]).unwrap()).unwrap();
        delta = delta_part_1.tens_mult(&delta_part_2).unwrap();

        let weights = delta.matrix_mult(&all_predictions[i-1].matrix_transpose().unwrap()).unwrap().mult(const_multiplier);
        let bias = delta.mult(const_multiplier);

        full_weights.push(weights);
        full_biases.push(bias);
    }
    full_weights.reverse();
    full_biases.reverse();

    Some(BackpropagationValues{
        weights: full_weights,
        biases: full_biases,
    })
}

/// get value of backpropagation on each layer of neural network using input data and y(real answers)
/// 
/// # Example
/// ```
/// use flashlight::prelude::*;
///
/// let mut nnetwork: NeuralNetwork = NeuralNetwork::new(vec!{2, 3, 3, 1}, 1.0, 1.0);
///
/// let input_data: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap().matrix_transpose().unwrap();
/// let predicted_data: Tensor<f32> = Tensor::from_data(&[3.0, 7.0, 11.0], &[3, 1]).unwrap();
///
/// backpropagation(&mut nnetwork, &input_data, &predicted_data);
/// ```
pub fn backpropagation(network: &mut NeuralNetwork, input: &Tensor<f32>, y: &Tensor<f32>){
    let backpropagation_values = backprop_values(network, input, y).unwrap();
    
    for i in 0..backpropagation_values.weights.len(){

        println!("{}\n+\n{}\n----------------------\n{}\n+\n{}", network.weights[i].matrix_to_string().unwrap(), backpropagation_values.weights[i].matrix_to_string().unwrap(), 
            network.biases[i].matrix_to_string().unwrap(), backpropagation_values.biases[i].matrix_to_string().unwrap());

        network.weights[i] = network.weights[i].tens_add(&backpropagation_values.weights[i]).unwrap();
        network.biases[i] = network.biases[i].tens_add(&backpropagation_values.biases[i]).unwrap();
    }
}
