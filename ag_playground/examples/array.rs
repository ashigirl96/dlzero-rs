use autograd::ndarray::array;

fn main() {
    let arr = array![[1., 2., 3.], [10., 100., 20.]];
    for i in &arr {
        eprintln!("i = {:#?}", i);
    }
}
