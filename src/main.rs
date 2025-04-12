mod math;

use math::matrix::*;

fn main() {
    let mat1 = Matrix::from_vec(vec![
        vec![1,2,3],
        vec![4,5,6]
    ]);

    let mat2 = Matrix::from_vec(vec![
        vec![1,2,5],
        vec![3,4,6],
    ]);

    println!("{}", mat2.transpose());
    println!("{}", matrix_mult(mat1, mat2.transpose()).unwrap());
}
