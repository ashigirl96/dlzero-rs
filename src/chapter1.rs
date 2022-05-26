use autograd as ag;
use autograd::array_gen::zeros;
use autograd::rand;
use autograd::rand::prelude::StdRng;
use autograd::rand::{Rng, SeedableRng};

pub fn avg1() {
    println!("avg1");
    let mut rng = StdRng::seed_from_u64(42);
    let mut rewards = zeros::<f32>(&[10]);
    for n in 1..=10 {
        let reward = rng.gen::<f32>();
        rewards[n - 1] = reward;
        let q_value = rewards.sum() / (n as f32);
        eprintln!("q_value = {:#?}", q_value);
    }
}

pub fn avg2() {
    println!("avg2");
    let mut rng = StdRng::seed_from_u64(42);
    let mut q_value: f32 = 0.0;
    for n in 1..=10 {
        let reward = rng.gen::<f32>();
        q_value = q_value + (reward - q_value) / (n as f32);
        eprintln!("q_value = {:#?}", q_value);
    }
}
