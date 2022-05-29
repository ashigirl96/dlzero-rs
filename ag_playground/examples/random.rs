use autograd as ag;
use autograd::ndarray_ext::ArrayRng;
use autograd::rand::prelude::StdRng;
use autograd::rand::{Rng, SeedableRng};
use autograd::{rand, NdArray};

fn main() {
    eprintln!("rand::random::<i32>() = {:#?}", rand::random::<i32>());
    eprintln!("rand::random::<f32>() = {:#?}", rand::random::<f32>());

    let mut rng = rand::thread_rng();

    // faster than rand::random cause, can be made faster by caching thread_rng.
    eprintln!("rng.gen(); = {:#?}", rng.gen::<i32>());
    eprintln!("rng.gen(); = {:#?}", rng.gen::<f32>());

    eprintln!("rng.gen_range(1..10) = {:#?}", rng.gen_range(1..10));

    let rng = ArrayRng::<f32>::default();
    eprintln!(
        "rng.random_uniform(&[3, 2, 4], 0.0, 1.0) = {:#?}",
        rng.random_uniform(&[3, 2, 4], 0.0, 1.0)
    );

    let mut rng0 = StdRng::seed_from_u64(42);
    let my_rng: ArrayRng<f32, StdRng> = ArrayRng::new(rng0);
    eprintln!(
        "my_rng.standard_normal(&[2, 3]) = {:#?}",
        my_rng.standard_normal(&[2, 3])
    );
}
