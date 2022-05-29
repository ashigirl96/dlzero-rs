use dl::chapter1::bandit::Agent;
use dl::chapter1::{avg1, avg2, Bandit};

fn main() {
    let action_size = 10;
    let epsilon = 0.1;
    let mut bandit = Bandit::new(action_size);
    let mut agent = Agent::new(epsilon, action_size);

    let mut rates = vec![];
    let mut total_reward = 0.0;
    for step in 1..1000 {
        let action = agent.step();
        let reward = bandit.step(action);
        agent.learn(action, reward);
        total_reward += reward;
        eprintln!("&total_reward = {:#?}", &total_reward);
        rates.push(total_reward / (step + 1) as f32);
    }
    eprintln!("rates = {:#?}", rates);
}
