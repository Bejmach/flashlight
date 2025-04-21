use crate::flashlight_model::*;

use flashlight_tensor::prelude::*;
use super::sigmoid::*;

pub struct BackpropagationValue{
    pub weights: Tensor<f32>,
    pub biases: Tensor<f32>,
}

impl Model{
    /// Get prediction of each layer in neural network based on &[f32] input data
    /// where data need to be of length of the input layer
    /// data returned, collumn per sample, row per neuron on layer
    /// neuron x samples
    ///
    /// Input data formated where each tensor row is one input sample.
    ///
    /// # Example
    /// ```
    /// use flashlight::prelude::*;
    ///
    /// let model: Model = Model::new(&[2, 3, 3, 1], 1.0, 1.0);
    ///
    /// let input_data: Tensor<f32> = Tensor::from_data(&[50.0, 150.0, 100.0, 220.0, 75.0, 190.0], &[3,
    /// 2]).unwrap();
    /// model.full_forward_propagation(&input_data).unwrap();
    /// ```
    pub fn full_forward_propagation(&self, input_data: &Tensor<f32>) -> Option<Vec<Tensor<f32>>>{
        if input_data.get_sizes().len() != 2{
            return None;
        }
        if input_data.get_sizes()[1] != self.layers[0]{
            return None;
        }

        let mut output_tensor: Tensor<f32> = input_data.clone().matrix_transpose().unwrap();
        
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

    /// return backpropagation values for last layer where, predictions and answers samples are
    /// stored one per row(transposed at last to be one on collumn)
    pub fn last_layer_backprop(&self, all_predictions: &Vec<Tensor<f32>>, real_answers: &Tensor<f32>) -> Option<BackpropagationValue>{
    if all_predictions.len() < 2{
        println!("To small number of layers to form backpropagation");
        return None;
    }
    //check if rows are equal
    if real_answers.get_sizes()[0] != all_predictions[all_predictions.len()-1].get_sizes()[0]{
        println!("Number of samples not equal number of real outputs");
        return None;
    }

    let const_multiplier: f32 = 1.0 / real_answers.count_data() as f32;

    let last_minus_real: Tensor<f32> = all_predictions[all_predictions.len()-1].tens_sub(real_answers).unwrap();
    let transposed_not_last: Tensor<f32> = all_predictions[all_predictions.len() - 2].matrix_transpose().unwrap();

    let backprop_weights: Tensor<f32> = last_minus_real.matrix_mult(&transposed_not_last).unwrap().mult(const_multiplier);

    let mut bias_sum: Tensor<f32> = Tensor::new(&[1, real_answers.get_sizes()[1]]);
    for i in 0..real_answers.get_sizes()[1]{
        let predicted_row = all_predictions[all_predictions.len()-1].matrix_row(i).unwrap();
        let real_row = real_answers.matrix_row(i).unwrap();
        bias_sum = bias_sum.tens_add(&predicted_row.tens_sub(&real_row).unwrap()).unwrap();
    }

    let backprop_biases = bias_sum.mult(const_multiplier).matrix_transpose().unwrap();

    Some(BackpropagationValue{
        weights: backprop_weights,
        biases: backprop_biases,
    })

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


