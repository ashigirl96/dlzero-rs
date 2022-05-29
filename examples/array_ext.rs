use autograd::ndarray::{array, Array2, ArrayBase, Dim, Dimension};
use autograd::ndarray_ext::ArrayRng;
use autograd::rand::prelude::StdRng;
use autograd::rand::SeedableRng;
use autograd::{array_gen, rand, NdArray};
use dl::array_ext::QuantileExt;

fn main() {
    // use ndarray::array;
    // use ndarray_stats::QuantileExt;

    let a = array![[1., 3., 5.], [2., 0., 6.]];
    assert_eq!(a.argmin(), Ok((1, 1)));
    // let qs = Array2::zeros::<f32>(&[3, 2]);
    // // let x = qs.argmin().unwrap();
    // // let rng = ArrayRng::<f32>::default();
    // let mut x = array_gen::zeros::<f32>(&[3, 2]);
    // eprintln!("x = {:#?}", &x.argmin().unwrap() == 2);

    // let my_rng = ArrayRng::new(StdRng::seed_from_u64(42));
    // let random: NdArray<f32> = my_rng.standard_normal(&[2, 3]);
    // // eprintln!("random = {:#?}", random.argmin().unwrap());
    // eprintln!("random.first() = {:#?}", &random);
    // eprintln!("random.first() = {:#?}", &random.first());
    // let hoge = rng.random_uniform(&[3, 2, 4], 0.0, 1.0);
    // eprintln!("&hoge[0] = {:#?}", hoge.argmin().unwrap()[0]);
    //
    // eprintln!(
    //     "rng.random_uniform(&[3, 2, 4], 0.0, 1.0) = {:#?}",
    //     hoge.argmin()
    // );
    //
    // // eprintln!("ns = {:#?}", ns.argmin());
    // eprintln!("x = {:#?}", x);

    // eprintln!("&a = {:#?}", &a.max());
}
