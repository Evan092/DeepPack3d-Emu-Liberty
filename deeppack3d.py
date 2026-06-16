import numpy as np
import os, shutil, time
import seaborn as sns
import tensorflow as tf
from env import *
from agent import *
import pandas as pd

@tf.keras.utils.register_keras_serializable()
class CompatibleDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

def parse_args():
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('method', metavar='method', 
                        type=str, choices=['rl', 'bl', 'baf', 'bssf', 'blsf'], 
                        help='choose the method from {"rl", "bl", "baf", "bssf", "blsf"}.')
    
    parser.add_argument('lookahead', metavar='lookahead', 
                        type=int,
                        help='choose the lookahead value.')
    
    parser.add_argument('--data', metavar='', 
                        type=str, default='generated', choices=['generated', 'input', 'file'], 
                        help='choose the input source from {"generated", "input", "file"} (default: generated).')
    
    parser.add_argument('--path', metavar='', 
                        type=str, default=None, 
                        help='set the file path, only used if --data is "file" (default: None).')
    
    parser.add_argument('--n_iterations', metavar='', 
                        type=int, default=100, 
                        help='set the number of iterations, only used if --data is "generated" (default: 100).')
    
    parser.add_argument('--seed', metavar='', 
                        type=str, default=None, 
                        help='set the random seed for reproducibility, only used if --data is "generated" (default: None).')
    
    parser.add_argument('--verbose', metavar='', 
                        type=int, default=1, 
                        help='set verbose level (default: 1).')
    
    parser.add_argument('--train', 
                        action='store_true', 
                        help='enable training mode, only used if method is "rl" (default: False).')
    
    parser.add_argument('--batch_size', metavar='', 
                        type=int, default=32, 
                        help='set batch_size, only used if train is True (default: 32).')
    
    parser.add_argument('--visualize', 
                        action='store_true', 
                        help='enable visualization mode (default: False).')
    
    parser.add_argument('--unity', metavar='', 
                        type=str, default=None, 
                        help='path to PackingSim_Linux.x86_64 for physics eval (default: None).')
    
    return parser.parse_args()

import numpy as np
import os, shutil, time
import seaborn as sns
from env import *
from agent import *

heuristics = {
    'bl': bottom_left,
    'baf': best_area_fit, 
    'bssf': best_short_side_fit, 
    'blsf': best_long_side_fit, 
}

def deeppack3d(method, lookahead, *, n_iterations=100, seed=None, verbose=1, data='generated', path=None, train=False, visualize=False, batch_size=32, unity_exe=None):
    reset_rng(seed)
    
    env = MultiBinPackerEnv(n_bins=1, 
                            max_bins=1, 
                            size=(32, 32, 32), 
                            k=lookahead, 
                            prealloc_items=100, 
                            verbose=verbose)

    if data == 'file':
        file_conveyor = FileConveyor(k=env.k, path=path)
        env.conveyor = file_conveyor.reset()
        # If the file specifies a pallet size, rescale the env to match
        if file_conveyor.bin_size is not None:
            env.size = file_conveyor.bin_size
            # Store inverse scale for converting grid coords back to real-world units
            env.inv_scale = file_conveyor.inv_scale
            # Recompute v_pad_headroom for the grid bin height
            W, H, D = env.size
            env.v_pad_headroom = (H + 1) * env.v_pad
            env.packers = [SpacePartitioner(env.size, floor_height=env.v_pad,
                                            v_pad_headroom=env.v_pad_headroom)
                           for _ in range(env.n_bins)]
            if verbose > 0:
                print(f'Pallet {file_conveyor.pallet_size} \u2192 '
                      f'grid bin {env.size} '
                      f'(scale={file_conveyor.scale:.6f}, '
                      f'1 grid unit = {file_conveyor.inv_scale:.3f} mm)')
    elif data == 'input':
        env.conveyor = InputConveyor(k=env.k).reset()

    if visualize:
        if os.path.exists('./outputs'):
            shutil.rmtree('./outputs')
        os.makedirs('./outputs')

    if train:
        print(f'Training with method "{method}" and lookahead {lookahead}...')
        
        if method != 'rl':
            raise Exception('training mode can only be used if method is "rl"')

        # env = BinPackerEnv(size=(32, 32, 32), k=env.k, bin_size=(32, 32, 32))


        model_path = f'./agent.h5'
        memory_path = model_path.replace('.h5', '_memory.pkl')
        agent = Agent(env, train=True, verbose=verbose > 0, visualize=visualize, batch_size=batch_size)
        # Load pre-trained model into both networks if it exists
        if os.path.exists(model_path):
            agent.q_net = tf.keras.models.load_model(model_path, compile=False, custom_objects={'Dense': CompatibleDense, 'CompatibleDense': CompatibleDense})
            agent.q_net_target = tf.keras.models.load_model(model_path, compile=False, custom_objects={'Dense': CompatibleDense, 'CompatibleDense': CompatibleDense})
            agent.q_net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

            if os.path.exists(memory_path) and True:
                agent.load_memory(memory_path)

        agent.eps = 1.0
        if os.path.exists(model_path):
            # ... load model ...
            agent.eps = 0.025  # or load from a saved epsilon value
        else:
            agent.eps = 1.0
        # Initialize Unity evaluator if exe path provided
        unity_evaluator = None
        if unity_exe is not None:
            unity_evaluator = UnityEvaluator(
                exe_path=unity_exe,
                placements_dir='./unity_sims',
                max_parallel=6,
                timeout=600,
                reward_scale=100.0,
            )
            print(f'Unity physics eval enabled: {unity_exe}')

        # Initialize lists to store episode data
        episode_data = []
        iteration_data = []
        stability_data = []
        
        for i in range(1, n_iterations):
            print(f'Iteration {i}')
            start_time = time.time()
            
            yield from agent.run(100, verbose=verbose > 1, unity_evaluator=unity_evaluator, ep_offset=(i - 1) * 100)

            # Wait for any Unity sims still running at the end of the iteration,
            # then apply their stability rewards to the replay buffer before logging.
            if unity_evaluator is not None:
                print(f'Waiting for remaining Unity sims...')
                unity_evaluator.wait_all()
                agent.apply_unity_results(unity_evaluator)

            #agent.eps = max(agent.eps * 0.95, 0.025)
            agent.eps = max(agent.eps * 0.975, 0.025)
            
            # Log episode data for this iteration
            for ep_idx, (utils, n_bins, ep_reward, items_packed) in enumerate(agent.ep_history[-100:]):
                for bin_idx, util in enumerate(utils):
                    episode_data.append({
                        'iteration': i,
                        'episode': ep_idx,
                        'bin': bin_idx,
                        'util_percent': util * 100,
                        'items_packed': items_packed,
                        'reward': ep_reward,
                        'n_bins': n_bins
                    })
            
            # Collect any Unity stability results that finished this iteration
            if unity_evaluator is not None:
                for ep_id, unity_reward in unity_evaluator.drain_log():
                    stability_score = (unity_reward / unity_evaluator.reward_scale) + 0.8
                    stability_data.append({
                        'iteration': ep_id // 100 + 1,
                        'episode': ep_id % 100,
                        'global_episode': ep_id,
                        'unity_reward': round(unity_reward, 4),
                        'stability_score': round(stability_score, 4),
                    })

            # Calculate iteration averages
            recent_episodes = agent.ep_history[-100:]
            utils_flat = [util for utils, n_bins, ep_reward, items_packed in recent_episodes for util in utils]
            items_packed_list = [items_packed for utils, n_bins, ep_reward, items_packed in recent_episodes]
            
            rewards_list = [ep_reward for utils, n_bins, ep_reward, items_packed in recent_episodes]

            iteration_data.append({
                'iteration': i,
                'mean_util': np.mean(utils_flat),
                'min_util': np.min(utils_flat),
                'max_util': np.max(utils_flat),
                'mean_items_packed': np.mean(items_packed_list),
                'min_items_packed': np.min(items_packed_list),
                'max_items_packed': np.max(items_packed_list),
                'mean_reward': np.mean(rewards_list),
                'min_reward': np.min(rewards_list),
                'max_reward': np.max(rewards_list),
            })
            
            # Save data every 10 iterations or on last iteration
            if (i + 1) % 10 == 0 or i == n_iterations - 1:
                # Save episode data
                episodes_df = pd.DataFrame(episode_data)
                episodes_df.to_excel('./training_episodes.xlsx', index=False)
                
                # Save iteration data
                iterations_df = pd.DataFrame(iteration_data)
                iterations_df.to_excel('./training_iterations.xlsx', index=False)

                # Save Unity stability data
                if stability_data:
                    stability_df = pd.DataFrame(stability_data)
                    stability_df.to_excel('./training_stability.xlsx', index=False)
                
                print(f'Data saved after iteration {i}')
            
            # Save model and memory after each iteration
            print(f'Saving model and memory after iteration {i}...')
            agent.q_net.save(model_path)
            
            memory_path = model_path.replace('.h5', '_memory.pkl')
            if hasattr(agent, 'save_memory'):
                agent.save_memory(memory_path)
            
            print(f'Iteration {i} completed in {time.time() - start_time:.2f} seconds, Avg Util: {np.mean(utils_flat)*100:.2f}%, Avg Items: {np.mean(items_packed_list):.2f}')
            
        if unity_evaluator is not None:
            unity_evaluator.shutdown(wait=True)

        data = np.asarray([utils for utils, n_bins, ep_reward, items_packed in agent.ep_history])
            
        data = np.asarray([utils for utils, n_bins, ep_reward, items_packed in agent.ep_history])
        # y = np.ones(100)
        # data = np.convolve(data, y, 'valid') / len(y)
        sns.lineplot(data=data)
        plt.savefig(f'./util.jpg')
        plt.show()
        
        data = np.asarray([ep_reward for utils, n_bins, ep_reward, items_packed in agent.ep_history])
        # y = np.ones(100)
        # data = np.convolve(data, y, 'valid') / len(y)
        sns.lineplot(data=data)
        plt.savefig(f'./ep_reward.jpg')
        plt.show()

        import uuid
        uid = uuid.uuid4()
        print(f'saved model at ./{uid}.h5')
        agent.q_net.save(f'{uid}.h5')
    else:
        if verbose > 0:
            print(f'Testing with method "{method}" and lookahead {lookahead}...')
        
        if method == 'rl':
            model_path = f'./models/k={lookahead}.h5'
            agent = Agent(env, train=False, verbose=verbose > 0, visualize=visualize, batch_size=batch_size)
            agent.q_net = tf.keras.models.load_model(model_path, compile=False)
            agent.eps = 0.0
        else:
            agent = HeuristicAgent(heuristics[method], env, verbose=verbose > 0, visualize=visualize)
        
        start_time = time.time()
        
        try:
            yield from agent.run(n_iterations, verbose=verbose > 1)
        except Exception as e:
            if np.all(np.array(env.conveyor.reset().peek()) == None):
                if verbose > 0:
                    print('\n=====the end of conveyor line=====')
            else:
                print(e)

        if verbose > 0:
            print()
            next_items = np.array(env.conveyor.reset().peek()).tolist()
            avg_util = np.mean([util for utils, n_bins, ep_reward, items_packed in agent.ep_history[:] for util in utils[:]])
            used_items = np.sum([n_bins for utils, n_bins, ep_reward, items_packed in agent.ep_history[:] for util in utils[:]])
            avg_items_packed = np.mean([items_packed for utils, n_bins, ep_reward, items_packed in agent.ep_history[:]])
            
            print(f'Used time: {int(time.time() - start_time)} seconds')
            print(f'Next items: {next_items}')
            print(f'Average space util: {avg_util}')
            print(f'Used bins: {used_items}')
            print(f'Average items packed: {avg_items_packed:.2f}')

def main():
    args = parse_args()
    
    reset_rng(args.seed)

    for _ in deeppack3d(args.method, 
                        args.lookahead, 
                        n_iterations=args.n_iterations, 
                        seed=args.seed, 
                        train=args.train, 
                        verbose=args.verbose, 
                        data=args.data, 
                        path=args.path,
                        visualize=args.visualize, 
                        batch_size=args.batch_size,
                        unity_exe=args.unity):
        pass

if __name__ == "__main__":
    main()