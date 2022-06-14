import random
import numpy as np
import gym
import yaml
from gym_qRacing.envs.functions import Helper


# initializing global logging lists
log_episodes = []
log_loss = []
log_reward = []
log_result = []


def simulate():
    #* loading global config from yaml file
    with open('racesim_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    #* initializing variables for Q-Learning
    episode_counter = 0 
    env = gym.make("qRacing-base-v0", config=config)
    num_box = tuple(((env.observation_space.high[0], env.observation_space.high[1]) + np.ones(env.observation_space.shape)).astype(int))
    q_table = np.zeros(num_box + (env.action_space.n,))


    #! The actual simulation loop. Iterating over the defined amount of episodes.
    for episode in range(config["QLEARNING"]["ENV_EPISODES"]):
        # logging
        Helper.global_logging(config["LOGGING"], "ENVIRONMENT", "\n[bold blue]Starting episode #{}[/bold blue]".format(episode+1))

        # initialize environment
        state = env.reset()
        total_reward = 0
        t_loss = 0

        # AI tries up to env_config["env_maxTry"] times
        for t in range(config["QLEARNING"]["ENV_MAXTRY"]):

            # In the beginning, do random action to learn
            if random.uniform(0, 1) < config["QLEARNING"]["ENV_EPSILON"]:
                action = env.action_space.sample()
            else:
                #print("\nstate: {} \n".format(state))
                action = np.argmax(q_table[state])

            # Do action and get result
            next_state, reward, done, _ = env.step(action)
            total_reward += reward


            # Get correspond q value from state, action pair
            #print("action index {}".format(action))
            q_value = q_table[state][action]
            best_q = np.max(q_table[next_state])
            t_loss += (q_value - reward)

            # Q(state, action) <- (1 - a)Q(state, action) + a(reward + rmaxQ(next state, all actions))
            q_table[state][action] = (1 - config["QLEARNING"]["ENV_LEARNINGRATE"]) * q_value + config["QLEARNING"]["ENV_LEARNINGRATE"] * (reward + config["QLEARNING"]["ENV_GAMMA"] * best_q)

            # Setup for the next episode iteration
            state = next_state

            
            
            # Check if episode is finished
            if done or t >= config["QLEARNING"]["ENV_MAXTRY"] - 1:
                # check if logging is enabled
                if episode_counter % config["LOGGING"]['EPISODE_INTERVAL'] == 0:
                    # log results to output
                    if config['LOGGING']['AGENT']['RESULTS']:
                        Helper.global_logging(config["LOGGING"], "ENVIRONMENT", "\n[bold blue]Agent results of episode #{}[/bold blue]".format(episode+1))
                        print("Position: %i \nReward: %f\n" % (state[0], total_reward))

                    # save results to logging list
                    log_loss.append([episode_counter, t_loss])
                    log_reward.append([episode_counter, total_reward])
                    log_result.append([episode_counter, state[0]])
                    
                episode_counter += 1
                break

        
        
        # exploration-rate decay
        if config["QLEARNING"]["ENV_EPSILON"] >= 0.005:
            config["QLEARNING"]["ENV_EPSILON"] *= config["QLEARNING"]["ENV_EPSILONDECAY"]


if __name__ == "__main__":
    simulate()
