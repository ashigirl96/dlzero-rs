use autograd as ag;
use autograd::array_gen::zeros;
use autograd::rand;
use autograd::rand::Rng;

pub fn avg() {
    let mut rng = rand::thread_rng();
    let mut rewards = zeros::<f32>(&[10]);
    for n in 1..=10 {
        let reward = rng.gen::<f32>();
        rewards[n - 1] = reward;
        let q_value = rewards.sum() / (n as f32);
        eprintln!("q_value = {:#?}", q_value);
    }
}
