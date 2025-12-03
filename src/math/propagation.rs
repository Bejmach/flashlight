use flashlight_tensor::prelude::*;

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
    if y_hat.get_shape() != y.get_shape(){
        return None;
    }

    for i in 0..y_hat.get_data().len(){
        if y_hat.get_data()[i] > 1.0 || y_hat.get_data()[i] < 0.0 || y.get_data()[i] > 1.0 || y.get_data()[i] < 0.0{
            return None
        }  
    }

    let mut y_hat_fixed_data: Vec<f32> = Vec::with_capacity(y_hat.count_data());

    for i in 0..y_hat.get_data().len(){
        if y_hat.get_data()[i] == 1.0{
            y_hat_fixed_data.push(0.999999);
        }
        else if y_hat.get_data()[i] == 0.0{
            y_hat_fixed_data.push(0.000001);
        }
        else {
            y_hat_fixed_data.push(y_hat.get_data()[i]);
        }
    }

    let y_hat_fixed: Tensor<f32> = Tensor::from_data(&y_hat_fixed_data, y_hat.get_shape()).unwrap();

    let tensor_ones: Tensor<f32> = Tensor::fill(1.0, y_hat.get_shape());
    
    //(y * log(y_hat))
    let y_log_y_hat: Tensor<f32> = y.tens_mul(&y_hat_fixed.nlog()).unwrap();
    //(1 - y)
    let negative_y: Tensor<f32> = tensor_ones.tens_sub(&y).unwrap();
    //log(1 - y_hat)
    let log_negative_y_hat: Tensor<f32> = tensor_ones.tens_sub(&y_hat_fixed).unwrap().nlog();

    let losses: Tensor<f32> = y_log_y_hat.tens_add( &negative_y.tens_mul(&log_negative_y_hat).unwrap() ).unwrap();

    let m: usize = y_hat_fixed.count_data();

    let const_multiplier = 1.0/m as f32;

    let mut summed_losses: f32 = 0.0;

    for i in 0..losses.get_shape()[0]{
        summed_losses += const_multiplier * losses.matrix_row(i).unwrap().sum();
    }

    Some(-summed_losses)
}


