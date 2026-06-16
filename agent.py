import os
import concurrent.futures
import queue
import subprocess

import numpy as np

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LeakyReLU, Lambda, Conv2D, GlobalAveragePooling2D, Flatten, Dense, MaxPooling2D, Conv2DTranspose, RepeatVector, Reshape, concatenate, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import orthogonal
import json
from env import *
import collections, itertools

class UnityEvaluator:
    """Runs Unity headless physics sim in background threads.
    Submit placements at episode end; drain results each episode to inject into replay buffer."""

    def __init__(self, exe_path, max_parallel=1, timeout=600, reward_scale=10.0, placements_dir='./unity_sims'):
        """
        exe_path:      path to PackingSim_Linux.x86_64
        max_parallel:  simultaneous Unity processes (each is heavy; 1-2 recommended)
        timeout:       per-sim hard cap in seconds
        reward_scale:  magnitude of unity reward (matched to existing terminal reward scale)
        placements_dir: directory for per-episode input/output files (avoids conflicts)
        """
        self.exe_path = exe_path
        self.timeout = timeout
        self.reward_scale = reward_scale
        self.placements_dir = placements_dir
        os.makedirs(placements_dir, exist_ok=True)
        self._result_queue = queue.Queue()
        self._log_queue = queue.Queue()  # separate copy for external logging
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel)
        self._inflight_futures = []

    def submit(self, placements, all_buf_indices, episode_id):
        """Non-blocking. Launches Unity sim in a background thread.
        all_buf_indices: list of (buf_idx, slot_ver) for every transition in the episode,
        in step order, so per-step stability rewards can be credited to the right transitions."""
        fut = self._pool.submit(self._run, list(placements), all_buf_indices, episode_id)
        self._inflight_futures.append(fut)

    def _run(self, placements, all_buf_indices, episode_id):
        agg_reward = 0.0
        per_step_rewards = []
        inp = os.path.join(self.placements_dir, f'placements_ep{episode_id}.jsonl')
        out = os.path.join(self.placements_dir, f'stability_ep{episode_id}.json')
        try:
            with open(inp, 'w') as f:
                for p in placements:
                    f.write(json.dumps(p) + '\n')
            proc = subprocess.run(
                [self.exe_path, '-batchmode', '-nographics', '-headlessEval',
                 '-input', inp, '-output', out],
                capture_output=True, timeout=self.timeout
            )
            if proc.returncode == 0 and os.path.exists(out):
                with open(out) as f:
                    data = json.load(f)
                agg_reward, per_step_rewards = self._parse(data)
            else:
                stderr = proc.stderr.decode('utf-8', errors='replace')[:500]
                print(f'[Unity] ep {episode_id}: sim failed (rc={proc.returncode}): {stderr}')
        except subprocess.TimeoutExpired:
            print(f'[Unity] ep {episode_id}: timed out after {self.timeout}s')
        except Exception as e:
            print(f'[Unity] ep {episode_id}: error: {e}')

        self._result_queue.put((episode_id, per_step_rewards, all_buf_indices))
        self._log_queue.put((episode_id, agg_reward))

    def _parse(self, result):
        """Compute per-step stability rewards using a composite score.

        Composite score per step combines three signals:
          - Tilt score:     continuous tilt angle normalized to [0,1] (0 deg=1.0, 45+ deg=0.0)
          - Discrete score: fraction of stable boxes, counting wobbling at half-weight
          - Motion score:   penalises linear speed and recent drift (impending instability)

        Per-step reward = clip(composite[i] - composite[i-1], min=-1, max=0) * reward_scale
        Only stability *drops* are penalised; recoveries are not rewarded, avoiding false
        credit to the placement that happened to land after a pile settled.

        Returns (agg_reward, per_step_rewards):
          agg_reward       -- scalar based on final_stability_score, used for logging
          per_step_rewards -- list aligned to placement steps (length == num steps in sim)
        """
        steps = result.get('steps', [])
        agg_score = float(result.get('final_stability_score', 0.5))
        agg_reward = self.reward_scale * (agg_score - 0.8)

        if not steps:
            return agg_reward, [agg_reward]

        per_step_rewards = []
        prev_composite = 1.0  # perfect stability before any boxes are placed

        for step in steps:
            n = max(step.get('num_boxes', 1), 1)

            # 1. Continuous tilt angle: 0 deg -> 1.0, 45+ deg -> 0.0
            mean_tilt = step.get('mean_tilt_angle_degrees', 0.0)
            tilt_score = max(0.0, 1.0 - mean_tilt / 45.0)

            # 2. Discrete fraction: tipped boxes count fully, wobbling boxes count as half
            n_tipped = step.get('num_tipped_boxes', 0)
            n_wobbling = step.get('num_wobbling_boxes', 0)
            discrete_score = 1.0 - (n_tipped + 0.5 * n_wobbling) / n

            # 3. Motion score: high linear speed or drift signals impending collapse
            linear_speed = step.get('mean_linear_speed', 0.0)
            drift = step.get('mean_recent_drift', 0.0)
            motion_score = max(0.0, 1.0 - linear_speed / 0.05 - drift / 0.02)

            composite = 0.4 * tilt_score + 0.4 * discrete_score + 0.2 * motion_score

            # Delta: how much did THIS placement harm stability?
            # Clip positive deltas to 0 so recoveries don't reward the wrong step.
            delta = composite - prev_composite
            step_reward = self.reward_scale * min(delta, 0.0)

            per_step_rewards.append(step_reward)
            prev_composite = composite

        return agg_reward, per_step_rewards

    def wait_all(self, timeout=None):
        """Block until all in-flight simulations have finished.
        Call this at the end of an iteration to ensure every submitted sim is done
        before draining queues and logging results."""
        concurrent.futures.wait(self._inflight_futures, timeout=timeout)
        self._inflight_futures.clear()

    def drain(self):
        """Collect all finished results without blocking.
        Returns list of (episode_id, per_step_rewards, all_buf_indices)."""
        results = []
        while True:
            try:
                results.append(self._result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def drain_log(self):
        """Collect completed (episode_id, unity_reward) pairs for external logging.
        Independent of drain() — consuming one does not affect the other."""
        results = []
        while True:
            try:
                results.append(self._log_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def shutdown(self, wait=True):
        """Shut down thread pool. Pass wait=False to abandon in-flight sims."""
        self._pool.shutdown(wait=wait, cancel_futures=not wait)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer using sum-tree for O(log n) sampling."""
    def __init__(self, maxlen=1000000, alpha=0.6):
        self.maxlen = maxlen
        self.alpha = alpha  # priority exponent: 0 = uniform, 1 = full prioritization
        self.buffer = []
        self.priorities = np.zeros(maxlen, dtype=np.float64)
        self._slot_version = np.zeros(maxlen, dtype=np.int64)  # increments each time a slot is overwritten
        self.pos = 0
        self.size = 0

    def __len__(self):
        return self.size

    def extend(self, transitions):
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
        for t in transitions:
            self.buffer.append(t) if self.size < self.maxlen else self.buffer.__setitem__(self.pos, t)
            self.priorities[self.pos] = max_priority
            self._slot_version[self.pos] += 1
            self.pos = (self.pos + 1) % self.maxlen
            self.size = min(self.size + 1, self.maxlen)

    def sample(self, batch_size, beta=0.4):
        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()

        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        # Importance-sampling weights to correct for non-uniform sampling bias
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalize so max weight = 1

        return samples, indices, np.float32(weights)

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = abs(td) + 1e-6

    def to_list(self):
        """Serialize buffer contents for saving."""
        return {
            'buffer': list(self.buffer),
            'priorities': self.priorities[:self.size].tolist(),
            'pos': self.pos,
            'size': self.size,
            'alpha': self.alpha,
            'maxlen': self.maxlen
        }

    @classmethod
    def from_dict(cls, data):
        """Reconstruct buffer from saved data."""
        buf = cls(maxlen=data['maxlen'], alpha=data['alpha'])
        buf.buffer = data['buffer']
        buf.size = data['size']
        buf.pos = data['pos']
        buf.priorities[:buf.size] = np.array(data['priorities'])
        return buf

def q_net(k=1):
    weight_decay = 0.0005
    hmap_in = Input((32, 32, 1))
    amap_in = Input((32, 32, 1))
    umap_in = Input((32, 32, 1))  # utilization map: fill density below surface
    imap_in = Input((k, 3))
    imap_x = Flatten()(imap_in)
    const_in = Input((32, 32, 1))
    # Explicit placement scalars: (x/W, y/H, z/D, w/W, h/H, d/D)
    # These bypass GlobalAveragePooling entirely, giving the network a direct,
    # undiluted signal about WHERE and HOW HIGH the item is placed.
    placement_in = Input((6,))
    
    x = concatenate([hmap_in, amap_in, umap_in, const_in], axis=-1)
    
    x = Conv2D(64, 11, strides=1, activation='relu', kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, 9, strides=1, activation='relu', kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, 7, strides=1, activation='relu', kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, 5, strides=1, activation='relu', kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, 3, strides=1, activation='relu', kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(2048, 2, strides=1, activation='relu', kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    
    x = GlobalAveragePooling2D()(x)
    
    emb = Dense(256, kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(imap_x)
    placement_emb = Dense(64, activation='relu', kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(placement_in)
    
    x = concatenate([x, emb, placement_emb], axis=-1)
    
    x = Dense(1000, activation='relu', kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    
    x = Dense(100, activation='relu', kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    
    x = Dense(1, activation='linear')(x)
    
    outputs = x
    model = Model([const_in, hmap_in, amap_in, umap_in, imap_in, placement_in], outputs)
    return model

class Agent:
    def __init__(self, env=MultiBinPackerEnv(n_bins=2, max_bins=-1, size=(32, 32, 32), k=10, verbose=True), train=True, verbose=True, visualize=False, batch_size=32):
        self.env = env
        
        self.gamma = 0.99
        
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.99
        
        self.ep_history = []
        
        self.warmup_epochs = 20
        self.warmup_lr = 1e-3
        self.learning_rate = 1e-3
        self.lr_min = 1e-5
        self.lr_drop = 10000
        self.epoch = 0
        self.update_epochs = 10

        self.n_step = 12  # n-step return horizon

        self.batch_size = batch_size
        
        self.__train = train

        self.verbose = verbose
        self.visualize = visualize
        
        if self.__train:
            self.q_net = q_net(k=env.k - 1)
            self.q_net_target = q_net(k=env.k - 1)
            # Initialize target network with same weights as online network
            self.q_net_target.set_weights(self.q_net.get_weights())
            self.q_optimizer = tf.keras.optimizers.Adam(learning_rate=self.warmup_lr)
            self.memory = PrioritizedReplayBuffer(maxlen=200000, alpha=0.6)
            self.per_beta = 0.4        # IS correction exponent, annealed toward 1
            self.per_beta_max = 1.0
            self.per_beta_anneal = 0.001
        else:
            self.q_net = None
            self.q_net_target = None
            self.q_optimizer = None
            self.memory = None
    
    def select(self, state):
        items, h_maps, u_maps, actions = state
        action_space = indices(actions)
        
#         print('actions: ', len(action_space))
        
        r = np.random.random()
        if r < self.eps:
            # 30% heuristic, 70% random
            if np.random.random() < 0.3:
                heuristic = np.random.choice(
                    [bottom_left, best_area_fit, best_short_side_fit, best_long_side_fit, spread_first, floor_first],
                    p=[0.05, 0.1, 0.1, 0.1, 0.3, 0.35]  # 65% weight on spreading/floor heuristics
                )
                action = tuple(heuristic(actions))
            else:
                action = action_space[np.random.choice(len(action_space))]
            r = 0
        else:
            q = self.Q(state)
            action = action_space[np.argmax(q)]
            r = np.max(q)
            
        return action, r, r != 0
    
    def save_memory(self, path):
        """Save replay buffer to disk (atomic write to avoid corruption on crash)"""
        import pickle, tempfile
        dir_name = os.path.dirname(path) or '.'
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
        try:
            with os.fdopen(fd, 'wb') as f:
                pickle.dump(self.memory.to_list(), f)
            os.replace(tmp_path, path)  # atomic on same filesystem
        except BaseException:
            os.unlink(tmp_path)  # clean up temp file on failure
            raise
        print(f'Saved {len(self.memory)} transitions to {path}')

    def load_memory(self, path):
        """Load replay buffer from disk"""
        import pickle, os
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f'Memory file {path} is empty or missing.  starting with fresh replay buffer')
            return
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except (EOFError, pickle.UnpicklingError, Exception) as e:
            print(f'Memory file {path} is corrupted ({e}).  starting with fresh replay buffer')
            return
        if isinstance(data, dict):
            # Check for old-format transitions (5-tuple with full states)
            if data.get('buffer') and len(data['buffer']) > 0 and len(data['buffer'][0]) == 5:
                print(f'Memory file {path} contains old-format transitions.  starting with fresh replay buffer')
                return
            self.memory = PrioritizedReplayBuffer.from_dict(data)
        else:
            print(f'Memory file {path} contains old-format data.  starting with fresh replay buffer')
            return
        print(f'Loaded {len(self.memory)} transitions from {path}')

    @staticmethod
    def _resize_map(m, size=(32, 32)):
        """Resize a 2D map to the neural net's expected input size."""
        if m.shape == size:
            return m
        # tf.image.resize expects (H, W, C)
        resized = tf.image.resize(m[..., np.newaxis], size, method='bilinear')
        return resized.numpy()[..., 0]

    def Q_inputs(self, state, action=None):
        W, H, D = self.env.size
        
        items, h_maps, u_maps, actions = state
        if action is None:
            action_space = indices(actions)
        else:
            i, j, k = action
            action_space = [(i, j, k)]
            
        imaps = [self.env.i_map(i, items) for i in range(len(self.env.packers))]
            
        hmap_in = []
        amap_in = []
        umap_in = []
        imap_in = []
        placement_in = []
        
        # item, bin, rotation_placement
        for i, j, k in action_space:
            _, (x, y, z), (w, h, d), _ = actions[i][j][k]
            amap = self.env.p_map(j, (x, y, z, w, h, d))
            amap = np.where(amap == 0, h_maps[j], y + h) / H

            # Actual heightmap so the model can see surface topology
            hmap = h_maps[j] / H

            # Utilization map: fill density below surface
            umap = u_maps[j]

            # Resize to 32x32 for neural net
            amap = self._resize_map(amap)
            hmap = self._resize_map(hmap)
            umap = self._resize_map(umap)

            imap = imaps[j][np.arange(len(items)) != i]

            hmap_in.append(hmap)
            amap_in.append(amap)
            umap_in.append(umap)
            imap_in.append(imap)
            # Explicit placement scalars bypass GlobalAveragePooling dilution.
            # The item footprint is ~2-4% of the 32x32 map; after GAP the height
            # signal at the footprint is drowned out by the other 96-98% of pixels.
            placement_in.append([x / W, y / H, z / D, w / W, h / H, d / D])
            
        hmap_in, amap_in, umap_in, imap_in = map(np.asarray, (hmap_in, amap_in, umap_in, imap_in))
        hmap_in = hmap_in[..., None]
        amap_in = amap_in[..., None]
        umap_in = umap_in[..., None]
        const_in = np.ones(hmap_in.shape, dtype=np.float32)
        placement_in = np.array(placement_in, dtype=np.float32)
        
        return [const_in, hmap_in, amap_in, umap_in, imap_in, placement_in]
    
    def Q(self, state, action=None, net=None):
        if net is None:
            net = self.q_net
        const_in, hmap_in, amap_in, umap_in, imap_in, placement_in = self.Q_inputs(state, action)
        
        batch_size = self.batch_size
        sections = np.cumsum([self.batch_size] * int(np.ceil(const_in.shape[0] / batch_size) - 1))
        batches = map(lambda data: map(lambda x: x.copy(), np.split(data, sections, axis=0)), (const_in, hmap_in, amap_in, umap_in, imap_in, placement_in))
        
        outputs = []
        for const_in, hmap_in, amap_in, umap_in, imap_in, placement_in in zip(*batches):
            q = net([const_in, hmap_in, amap_in, umap_in, imap_in, placement_in])
            outputs.append(q)
        q = np.concatenate(outputs, axis=0)
        return q

    def lr_scheduler(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.warmup_lr
        else:
            lr = self.learning_rate * (0.5 ** (epoch / self.lr_drop))
        return max(self.lr_min, lr)
            
    def train(self, history, is_weights=None):
        q_inputs = []
        q_targets = []
        
        for compact_inputs, n_step_return, reward, done in history:
            q_inputs.append(list(compact_inputs))
            q_targets.append([n_step_return])
            
        const_in, hmap_in, amap_in, umap_in, imap_in, placement_in = zip(*q_inputs)
        q_inputs = [np.asarray(const_in), np.asarray(hmap_in), np.asarray(amap_in), np.asarray(umap_in), np.asarray(imap_in), np.asarray(placement_in)]
        q_targets = np.asarray(q_targets)
        return self.fit(q_inputs, q_targets, is_weights=is_weights)
    
    def fit(self, q_inputs, q_targets, is_weights=None):
        if is_weights is None:
            is_weights = np.ones(q_targets.shape[0], dtype=np.float32)
        is_weights_tf = tf.constant(is_weights[:, None], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # Train online network, use target network for targets
            q = self.q_net(q_inputs, training=True)  # training=True: BatchNorm uses batch stats
            td_errors = q_targets - q  # per-sample TD errors
            # Weighted MSE: importance-sampling weights correct for PER bias
            loss = tf.reduce_mean(is_weights_tf * tf.square(td_errors))

            # Height regularization: off-policy contamination makes n-step returns
            # for floor placements and tower placements converge to similar values
            # (shown empirically: floor step gets -27, tower step gets -25.6).
            # This anchors Q-values to be strictly decreasing in y/H, which is
            # always correct regardless of policy. We have y/H in placement_in[1].
            placement_in_tf = tf.cast(q_inputs[5], tf.float32)  # (batch, 6)
            y_over_H = placement_in_tf[:, 1:2]  # (batch, 1)
            height_reg_target = -8.0 * y_over_H  # floor -> ~0, top of bin -> -8
            height_reg_loss = tf.reduce_mean(tf.square(q - height_reg_target))
            loss += 0.3 * height_reg_loss
        
        grad = tape.gradient(loss, self.q_net.trainable_variables)
        self.q_optimizer.apply_gradients(zip(grad, self.q_net.trainable_variables))
        
        self.q_optimizer.learning_rate.assign(self.lr_scheduler(self.epoch))
        
        self.epoch += 1
        
        # Periodically sync target network with online network
        if self.epoch % self.update_epochs == 0:
            print('update')
            self.q_net_target.set_weights(self.q_net.get_weights())
        
        return loss, td_errors.numpy().flatten()
    
    def apply_unity_results(self, unity_evaluator):
        """Drain completed Unity results and patch the replay buffer in-place.
        Safe to call at any time: processes whatever has finished so far."""
        if self.memory is None:
            return
        for ep_id, per_step_rewards, all_buf_indices in unity_evaluator.drain():
            patched = 0
            max_pri = self.memory.priorities[:self.memory.size].max() if self.memory.size > 0 else 1.0
            for k, (buf_idx, slot_ver) in enumerate(all_buf_indices):
                step_reward = per_step_rewards[k] if k < len(per_step_rewards) else 0.0
                if step_reward == 0.0:
                    continue  # neutral steps don't need patching
                if buf_idx < self.memory.size and self.memory._slot_version[buf_idx] == slot_ver:
                    ci, ns_ret, r, d = self.memory.buffer[buf_idx]
                    self.memory.buffer[buf_idx] = (ci, ns_ret + step_reward, r + step_reward, d)
                    self.memory.priorities[buf_idx] = max_pri
                    patched += 1
            print(f'[Unity] ep {ep_id}: patched {patched}/{len(all_buf_indices)} transitions with per-step stability rewards')

    def run(self, max_ep=1, verbose=False, train=None, unity_evaluator=None, ep_offset=0):
        if train is None:
            train = self.__train
            
        iters = (i for i, _ in enumerate(iter(bool, True))) if max_ep == -1 else range(max_ep)

        for ep in iters:
            # === DRAIN COMPLETED UNITY RESULTS ===
            if unity_evaluator is not None and train:
                self.apply_unity_results(unity_evaluator)

            if verbose:
                print(f'ep {ep}:')
                
            state = self.env.reset()
            ep_reward = 0
            items_packed = 0
            
            history = []
            ep_placements = []  # placements accumulated for Unity eval
            
            for step in itertools.count():
                if verbose:
                    print(f'\nstep {step}')
                    
                items, h_map, u_maps, actions = state
                if len(actions) == 0:
                    break
                action, r, exploited = self.select(state)
                
                if verbose:
                    print(f'possible actions: {len(actions)}')
                    print(f'action: {action}')
                    print(f'placement: {actions[action[0]][action[1]][action[2]]}')
                
                yield actions[action[0]][action[1]][action[2]]
                items_packed += 1
                
                next_state, reward, done = self.env.step(action)

                # Track placement for Unity physics eval
                if unity_evaluator is not None:
                    bin_index, (px, py, pz), (pw, ph, pd), _ = actions[action[0]][action[1]][action[2]]
                    s = 1
                    ep_placements.append({
                        'bin': int(bin_index),
                        'x': round(float(px * s), 3), 'y': round(float(py * s), 3), 'z': round(float(pz * s), 3),
                        'w': round(float(pw * s), 3), 'h': round(float(ph * s), 3), 'd': round(float(pd * s), 3),
                    })
                
                if train:
                    # Store compact NN inputs instead of full states (~17KB vs ~589KB per transition)
                    q_in = self.Q_inputs(state, action)
                    compact_inputs = tuple(arr[0].copy() for arr in q_in)
                    history.append((compact_inputs, reward, done))
                
                if self.visualize:
                    next_items, _, _, _ = next_state
                    if not os.path.exists(f'./outputs/bin{ep}/'):
                        os.makedirs(f'./outputs/bin{ep}/', exist_ok=True)
                    with open(f"./outputs/bin{ep}/conveyor.jsonl", "a") as f:
                        f.write(json.dumps({
                            "step": step,
                            "items": [{"w": round(float(it[0]), 3), "h": round(float(it[1]), 3), "d": round(float(it[2]), 3)} if it is not None else None for it in next_items]
                        }) + "\n")
                    for i, packer in enumerate(self.env.packers):
                        packer.render(conveyor_items=next_items, conveyor_colors=self.env._item_colors).savefig(f'./outputs/bin{ep}/{step}_{i}.jpg')
                        with open(f"./outputs/bin{ep}/placements.jsonl", "a") as f:
                            placement = actions[action[0]][action[1]][action[2]]
                            bin_index, (x, y, z), (w, h, d), split = placement
                            s = 1#getattr(self.env, 'inv_scale', 1.0)
                            f.write(json.dumps({
                                "bin": int(bin_index),
                                "x": round(float(x * s), 3), "y": round(float(y * s), 3), "z": round(float(z * s), 3),
                                "w": round(float(w * s), 3), "h": round(float(h * s), 3), "d": round(float(d * s), 3)
                            }) + "\n")
                        with open(f"./outputs/bin{ep}/boxRewards.jsonl", "a") as f:
                            placement = actions[action[0]][action[1]][action[2]]
                            bin_index, (x, y, z), (w, h, d), split = placement
                            s = 1
                            entry = {
                                "step": step,
                                "bin": int(bin_index),
                                "x": round(float(x * s), 3), "y": round(float(y * s), 3), "z": round(float(z * s), 3),
                                "w": round(float(w * s), 3), "h": round(float(h * s), 3), "d": round(float(d * s), 3),
                                "reward": round(float(reward), 4),
                                "agent": str(exploited),
                            }
                            if hasattr(self.env, '_reward_breakdown'):
                                entry['reward_breakdown'] = self.env._reward_breakdown
                            f.write(json.dumps(entry) + "\n")
                    
                ep_reward += reward
                if done:
                    break
                state = next_state
                
            loss = None
            if train:
                # Compute n-step returns (truncated at end of episode, no full-MC bias)
                n_step_returns = [0.0] * len(history)
                for t in range(len(history)):
                    G = 0.0
                    for k in range(min(self.n_step, len(history) - t)):
                        G += (self.gamma ** k) * history[t + k][1]
                    n_step_returns[t] = G
                transitions = [(ci, ns_ret, r, d) for (ci, r, d), ns_ret in zip(history, n_step_returns)]
                self.memory.extend(transitions)
                # Record (buf_idx, slot_version) for every transition in this episode so that
                # per-step stability rewards can be credited to the exact right buffer slot.
                ep_len = len(transitions)
                all_buf_indices = [
                    (
                        (self.memory.pos - ep_len + k) % self.memory.maxlen,
                        self.memory._slot_version[(self.memory.pos - ep_len + k) % self.memory.maxlen],
                    )
                    for k in range(ep_len)
                ]

                # Submit Unity eval AFTER committing to buffer (non-blocking, ~4 min in background)
                # When it finishes, per-step stability rewards are distributed to each transition.
                if unity_evaluator is not None and ep_placements:
                    unity_evaluator.submit(ep_placements, all_buf_indices, ep_offset + ep)
                    print(f'[Unity] ep {ep_offset + ep}: submitted {len(ep_placements)} placements for physics eval')

                if len(self.memory) > 1000:
                    # Train multiple times per episode, proportional to new data collected
                    # Standard DQN does ~1 gradient step per 4 env steps
                    n_updates = max(1, len(history) // 4)
                    print(f'update model x{n_updates} ({len(history)} new transitions)')
                    for _ in range(n_updates):
                        beta = min(self.per_beta_max, self.per_beta + self.epoch * self.per_beta_anneal)
                        sampled_history, sample_indices, is_weights = self.memory.sample(128, beta=beta)
                        loss, td_errors = self.train(sampled_history, is_weights=is_weights)
                        self.memory.update_priorities(sample_indices, td_errors)
            
            self.ep_history.append(([packer.space_utilization() for packer in self.env.used_packers], self.env.used_bins, ep_reward, items_packed))
            
            yield None
            
            utils = [round(packer.space_utilization() * 100, 2) for packer in self.env.used_packers]
            if self.verbose: print(f'Episode {ep}, util: {utils}, used bins: {self.env.used_bins}, ep_reward: {ep_reward:.2f}, memory: {len(self.memory) if self.memory is not None else None}, eps: {self.eps:.2f}, loss: {loss}, lr: {self.q_optimizer.learning_rate.numpy() if self.q_optimizer is not None else None}')

class HeuristicAgent:
    def __init__(self, heuristic, env=MultiBinPackerEnv(n_bins=2, max_bins=-1, size=(32, 32, 32), k=10, verbose=True), verbose=True, visualize=False):
        self.env = env
        
        self.heuristic = heuristic
        self.ep_history = []

        self.verbose = verbose
        self.visualize = visualize
    
    def select(self, state):
        # state = (items, h_map, u_maps, actions)
        
        items, h_map, u_maps, actions = state
#         print(len(indices(actions)))
        action = self.heuristic(actions)
            
        return action
    
    def run(self, max_ep=1, verbose=False):
        iters = (i for i, _ in enumerate(iter(bool, True))) if max_ep == -1 else range(max_ep)
        
        for ep in iters:
            if verbose:
                print(f'ep {ep}:')
                
            state = self.env.reset()
            ep_reward = 0
            items_packed = 0
            
            history = []
            
            for step in itertools.count():
                if verbose:
                    print(f'\nstep {step}')
                    
                items, h_map, u_maps, actions = state
                if len(actions) == 0:
                    break
                
                action = self.select(state)
                
                if verbose:
                    print(f'action: {action}')
                
                next_state, reward, done = self.env.step(action)
                
                history.append((state, action, next_state, reward, done))
                
                if verbose:
                    print(f'actions: {actions}')
                    print(f'reward: {reward}, done: {done}')
                    print(f'placement: {actions[action[0]][action[1]][action[2]]}')
                
                yield actions[action[0]][action[1]][action[2]]
                items_packed += 1
                
                ep_reward += reward
                
                if self.visualize:
                    next_items_h, _, _, _ = next_state
                    for i, packer in enumerate(self.env.packers):
                        packer.render(conveyor_items=next_items_h).savefig(f'./outputs/{ep}_{step}_{i}.jpg')
                        
                if done:
                    break
                
                state = next_state
            
            self.ep_history.append(([packer.space_utilization() for packer in self.env.used_packers], self.env.used_bins, ep_reward, items_packed))
            
            yield None
            
            utils = [round(packer.space_utilization() * 100, 2) for packer in self.env.used_packers]
            if self.verbose: print(f'Episode {ep}, util: {utils}, used bins: {self.env.used_bins}, ep_reward: {ep_reward:.2f}')

def bottom_left(actions):
    scores = []
    for i, item in enumerate(actions):
        for j, bin_ in enumerate(item):
            for k, placement in enumerate(bin_):
                item, (x, y, z), (w, h, d), _ = placement
                y = y + h
                x = x + w
                z = z + d
                scores.append(([y, x, z, i, j, k], [i, j, k]))
        
    indices = sorted(range(len(scores)), key=lambda i: scores[i][0])
    return scores[indices[0]][1]

def best_short_side_fit(actions):
    scores = []
    for i, item in enumerate(actions):
        for j, bin_ in enumerate(item):
            for k, placement in enumerate(bin_):
                item, (x, y, z), (w, h, d), split = placement
                W, H = split.width, split.height
                scores.append(((min(W - w, H - h), i, j, k), [i, j, k]))
        
#     print(scores)
    indices = sorted(range(len(scores)), key=lambda i: scores[i][0])
    return scores[indices[0]][1]

def best_area_fit(actions):
    scores = []
    for i, item in enumerate(actions):
        for j, bin_ in enumerate(item):
            for k, placement in enumerate(bin_):
                item, (x, y, z), (w, h, d), split = placement
                W, H = split.width, split.height
                scores.append(((split.volume, min(W - w, H - h), i, j, k), [i, j, k]))
        
#     print(scores)
    indices = sorted(range(len(scores)), key=lambda i: scores[i][0])
    return scores[indices[0]][1]

def best_long_side_fit(actions):
    scores = []
    for i, item in enumerate(actions):
        for j, bin_ in enumerate(item):
            for k, placement in enumerate(bin_):
                item, (x, y, z), (w, h, d), split = placement
                W, H = split.width, split.height
                scores.append(((max(W - w, H - h), i, j, k), [i, j, k]))
        
    indices = sorted(range(len(scores)), key=lambda i: scores[i][0])
    return scores[indices[0]][1]

def spread_first(actions):
    """Heuristic that prefers placements on the floor with the largest footprint,
    encouraging even coverage rather than stacking."""
    scores = []
    for i, item in enumerate(actions):
        for j, bin_ in enumerate(item):
            for k, placement in enumerate(bin_):
                _, (x, y, z), (w, h, d), _ = placement
                # Strongly prefer floor placements (low y)
                # Then prefer large footprint items (spread faster)
                footprint = w * d
                scores.append(((y, -footprint, i, j, k), [i, j, k]))
    
    indices = sorted(range(len(scores)), key=lambda idx: scores[idx][0])
    return scores[indices[0]][1]

def floor_first(actions):
    """Heuristic that strictly prioritizes floor-level placements.
    Among floor placements, picks the largest footprint to maximize coverage.
    Only stacks when no floor space is available."""
    scores = []
    for i, item in enumerate(actions):
        for j, bin_ in enumerate(item):
            for k, placement in enumerate(bin_):
                _, (x, y, z), (w, h, d), _ = placement
                # Binary: is this a floor placement?
                is_floor = 0 if y < 1.0 else 1
                footprint = w * d
                scores.append(((is_floor, -footprint, y, i, j, k), [i, j, k]))
    
    indices = sorted(range(len(scores)), key=lambda i: scores[i][0])
    return scores[indices[0]][1]