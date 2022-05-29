// use dl::twd::zeros;
use autograd::ndarray::Array2;

fn main() {
    // pub type Array2<A> = ndarray::Array<A, ndarray::Ix2>;
    // let x = Array2::<f32>::zeros((3, 2));
    // let x = zeros(&[3, 2]);
    // let x = zeros((3, 2));
    // eprintln!("x = {:#?}", x);
    let x = Array2::zeros(autograd::ndarray::Ix2(2, 3));
    // let x = Array2::zeros::<>((3, 2));
    eprintln!("&x = {:#?}", &x);
}
