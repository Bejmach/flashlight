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
    /// Input data formated where each tensor collumn is one input sample.
    ///
    /// # Example
    /// ```
    /// use flashlight::prelude::*;
    ///
    /// let model: Model = Model::new(&[2, 3, 3, 1], 1.0, 1.0);
    ///
    /// let input_data: Tensor<f32> = Tensor::from_data(&[50.0, 150.0, 100.0, 220.0, 75.0, 190.0], &[3,
    /// 2]).unwrap().matrix_transpose().unwrap();
    /// model.full_forward_propagation(&input_data).unwrap();
    /// ```
    pub fn full_forward_propagation(&self, input_data: &Tensor<f32>) -> Option<Vec<Tensor<f32>>>{
        if input_data.get_sizes().len() != 2{
            println!("input not matrix");
            return None;
        }
        if input_data.get_sizes()[0] != self.layers[0]{
            println!("bach size and input layer differ, FFP");
            return None;
        }

        let mut output_tensor: Tensor<f32> = input_data.clone();
        
        let mut all_outputs: Vec<Tensor<f32>> = Vec::with_capacity(self.layers.len()-1);

        all_outputs.push(output_tensor.clone());

        for i in 1..self.layers.len(){
            

            let weights = &self.weights[i-1];
            let biases = &self.biases[i-1];
            
            //prints left for future debuging
            //println!("{}\n*\n{}\n=", weights.matrix_to_string().unwrap(), output_tensor.matrix_to_string().unwrap());

            let multiplied_tensor = weights.matrix_mult(&output_tensor).unwrap();
    
            //println!("{}\n_________", multiplied_tensor.matrix_to_string().unwrap());
            //println!("{}\n+\n{}", multiplied_tensor.matrix_to_string().unwrap(), biases.matrix_to_string().unwrap());

            output_tensor = multiplied_tensor.tens_broadcast_add(&biases).unwrap();
            
            //println!("=\n{}", output_tensor.matrix_to_string().unwrap());

            for row in 0..output_tensor.get_sizes()[0]{
                for collumn in 0..output_tensor.get_sizes()[1]{
                    output_tensor.set(sigmoid(output_tensor.value(&[row, collumn]).unwrap().clone()), &[row, collumn]);
                }
            }

            //println!("sigmoid\n{}", output_tensor.matrix_to_string().unwrap());

            //println!("_______________________________________");

            all_outputs.push(output_tensor.clone());

        }
        //println!("Propagation finished");
        Some(all_outputs)
    }
    
    /// Perform a backpropagation on model using input data and real answers, with learning_rate
    /// learning rate betweem 0.01 - 0.05 is advised
    ///
    /// # Example
    /// ```
    /// use flashlight::prelude::*;
    ///
    /// let mut model: Model = Model::new(&vec!{2, 3, 3, 1}, 1.0, 1.0);
    ///
    /// let input_data: Tensor<f32> = Tensor::from_data(&[10.0, 15.0, 25.0, 30.0, 80.0, 20.0], &[3,2]).unwrap().matrix_transpose().unwrap();
    ///let expected_output: Tensor<f32> = Tensor::from_data(&[sigmoid((10.0+15.0)/100.0), sigmoid((25.0+30.0)/100.0), sigmoid((80.0+20.0)/100.0)], &[3, 1]).unwrap().matrix_transpose().unwrap();
    ///
    /// model.cross_entropy_backprop_loop(&input_data, &expected_output, 0.05);
    /// ```
    pub fn cross_entropy_backprop_loop(&mut self, input_data: &Tensor<f32>, real_answers: &Tensor<f32>, learning_rate: f32) {

        if input_data.get_sizes()[1] != real_answers.get_sizes()[1]{
            println!("input and output bach count differ");
            return;
        }
        if input_data.get_sizes()[0] != self.layers[0]{
            println!("bach size and input layer size differ, CEBP");
            return;
        }
            
        let current_layer = self.layers.len()-1;

        let all_activations = self.full_forward_propagation(input_data).unwrap();

        let mut backprop_values: Vec<BackpropagationValue> = Vec::with_capacity(self.layers.len()-1);

        //C/Z[L] 
        let mut delta: Tensor<f32> = all_activations[current_layer].tens_sub(real_answers).unwrap();

        //println!("Activation len: {}, weight len: {}", all_activations.len(), self.weights.len());

        //println!("Delta:\n{}\n", delta.matrix_to_string().unwrap());

        //1/m
        let const_multiplier = 1.0/real_answers.get_sizes()[1] as f32;
        
        // get average of weight backpropagation calculated by using C/Z[L] * A[L-1] * 1/m
        let weight_backprop = delta.matrix_mult(&all_activations[current_layer-1].matrix_transpose().unwrap()).unwrap().mult(const_multiplier);
        //println!("{}, {}/ {}", delta.get_sizes()[0], delta.get_sizes()[1], delta.get_sizes().len());
        
        //average of sum of all bach bias backpropagations calculated using sum(C/Z[L]) * 1/m
        let mut bias_backprop: Tensor<f32> =  Tensor::fill(0.0, &[delta.get_sizes()[0], 1]);
        for i in 0..delta.get_sizes()[1]{
            //println!("{}\n\n{}", bias_backprop.matrix_to_string().unwrap(), &delta.matrix_to_string().unwrap());
            //println!("{}", &delta.matrix_col(i).unwrap().transform(&[delta.get_sizes()[0], 1]).unwrap().matrix_to_string().unwrap());
            bias_backprop = bias_backprop.tens_add(&delta.matrix_col(i).unwrap().transform(&[delta.get_sizes()[0], 1]).unwrap()).unwrap();
        }
        bias_backprop = bias_backprop.mult(const_multiplier);

        //println!("\nlayer_backprop:\n{}\n\n{}", weight_backprop.matrix_to_string().unwrap(), bias_backprop.matrix_to_string().unwrap());

        let sigmoid_deriv = all_activations[current_layer].tens_mult(&all_activations[current_layer].mult(-1.0).add(1.0)).unwrap();
        //println!("sigmoid_deriv:\n{}\n", sigmoid_deriv.matrix_to_string().unwrap());

        self.weights[current_layer-1] = self.weights[current_layer-1].tens_sub(&weight_backprop.mult(learning_rate)).unwrap();
        self.biases[current_layer-1] = self.biases[current_layer-1].tens_sub(&bias_backprop.mult(learning_rate)).unwrap();

        for i in 1..self.layers.len()-1{
            let current_layer = current_layer-i;

            //println!("Current layer: {}", current_layer);

            //println!("last delta:\n{}\n", delta.matrix_to_string().unwrap());

            let weight = &self.weights[current_layer].matrix_transpose().unwrap();
            //println!("weight:\n{}\n", weight.matrix_to_string().unwrap());

            let activation_derivative = all_activations[current_layer].tens_mult(&all_activations[current_layer].mult(-1.0).add(1.0)).unwrap();
            //println!("activation_derivative:\n{}\n", activation_derivative.matrix_to_string().unwrap());

            delta = weight.matrix_mult(&delta).unwrap().tens_mult(&activation_derivative).unwrap();
            //println!("new_delta:\n{}\n", delta.matrix_to_string().unwrap());
            
            //println!("last weight:\n{}\n", self.weights[current_layer-1].matrix_to_string().unwrap());
            let weight_backprop = delta.matrix_mult(&all_activations[current_layer-1].matrix_transpose().unwrap()).unwrap().mult(const_multiplier);
            //println!("weight_backprop:\n{}\n", weight_backprop.matrix_to_string().unwrap());

            let mut bias_backprop: Tensor<f32> =  Tensor::fill(0.0, &[delta.get_sizes()[0], 1]);
            for i in 0..delta.get_sizes()[1]{
                //println!("{}\n\n{}", bias_backprop.matrix_to_string().unwrap(), &delta.matrix_to_string().unwrap());
                //println!("{}", &delta.matrix_col(i).unwrap().transform(&[delta.get_sizes()[0], 1]).unwrap().matrix_to_string().unwrap());
                bias_backprop = bias_backprop.tens_add(&delta.matrix_col(i).unwrap().transform(&[delta.get_sizes()[0], 1]).unwrap()).unwrap();
            }
            bias_backprop = bias_backprop.mult(const_multiplier);
            //println!("last bias:\n{}\n", self.biases[current_layer-1].matrix_to_string().unwrap());
            //println!("bias_backprop:\n{}\n", bias_backprop.matrix_to_string().unwrap());

            self.weights[current_layer-1] = self.weights[current_layer-1].tens_sub(&weight_backprop.mult(learning_rate)).unwrap();
            self.biases[current_layer-1] = self.biases[current_layer-1].tens_sub(&bias_backprop.mult(learning_rate)).unwrap();
        }
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
/// let network_cost = cross_entropy_cost(&y_hat, &y).unwrap();
/// ```
pub fn cross_entropy_cost(y_hat: &Tensor<f32>, y: &Tensor<f32>) -> Option<f32>{
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


