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

    def submit(self, placements, all_buf_indices, episode_id, supporters=None):
        """Non-blocking. Launches Unity sim in a background thread.
        all_buf_indices: list of (buf_idx, slot_ver) for every transition in the episode,
        in step order, so per-step stability rewards can be credited to the right transitions.
        supporters: per placement, {supporting placement index: share of load-bearing area}."""
        fut = self._pool.submit(self._run, list(placements), all_buf_indices, episode_id, supporters)
        self._inflight_futures.append(fut)

    def _run(self, placements, all_buf_indices, episode_id, supporters=None):
        agg_reward = 0.0
        per_step_rewards = []
        succeeded = False
        inp = os.path.join(self.placements_dir, f'placements_ep{episode_id}.jsonl')
        out = os.path.join(self.placements_dir, f'stability_ep{episode_id}.json')
        try:
            with open(inp, 'w') as f:
                for p in placements:
                    f.write(json.dumps(p) + '\n')
            if os.path.exists(out):
                os.remove(out)
            proc = subprocess.run(
                [self.exe_path, '-batchmode', '-nographics', '-headlessEval',
                 '-input', inp, '-output', out],
                capture_output=True, timeout=self.timeout
            )
            if proc.returncode == 0 and os.path.exists(out):
                with open(out) as f:
                    data = json.load(f)
                agg_reward, per_step_rewards = self._parse(data, len(placements), supporters)
                succeeded = True
            else:
                stderr = proc.stderr.decode('utf-8', errors='replace')[:500]
                print(f'[Unity] ep {episode_id}: sim failed (rc={proc.returncode}): {stderr}')
        except subprocess.TimeoutExpired:
            print(f'[Unity] ep {episode_id}: timed out after {self.timeout}s')
        except Exception as e:
            print(f'[Unity] ep {episode_id}: error: {e}')

        self._result_queue.put((episode_id, per_step_rewards, all_buf_indices))
        self._log_queue.put((episode_id, agg_reward, succeeded))

    def _parse(self, result, n_placements, supporters=None):
        """Charge each box that ends the sim tipped to its own placement, and half as much to the boxes it rested on.

        Unity box ids are placement indices. Returns (agg_reward for logging, per_step_rewards aligned to placements).
        """
        agg_score = float(result.get('final_stability_score', 0.5))
        agg_reward = self.reward_scale * (agg_score - 0.8)

        per_step_rewards = [0.0] * n_placements
        steps = result.get('steps', [])
        if not steps:
            return agg_reward, per_step_rewards

        for box in steps[-1].get('box_results', []):
            box_id = int(box.get('id', -1))
            if not box.get('tipped') or not 0 <= box_id < n_placements:
                continue
            per_step_rewards[box_id] -= self.reward_scale
            box_supporters = supporters[box_id] if supporters and box_id < len(supporters) else {}
            for supporter, share in box_supporters.items():
                if 0 <= supporter < n_placements:
                    per_step_rewards[supporter] -= 0.5 * self.reward_scale * share

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
        """Collect completed (episode_id, unity_reward, succeeded) results.
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
    """Prioritized Experience Replay buffer."""
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
    boundary_in = Input((32, 32, 1))  # seam-aware boundary map for interweaving
    imap_in = Input((k, 3))
    imap_x = Flatten()(imap_in)
    # Explicit placement scalars: (x/W, y/H, z/D, w/W, h/H, d/D)
    # These bypass GlobalAveragePooling entirely, giving the network a direct,
    # undiluted signal about WHERE and HOW HIGH the item is placed.
    placement_in = Input((6,))

    x = concatenate([hmap_in, amap_in, umap_in, boundary_in], axis=-1)

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
    model = Model([boundary_in, hmap_in, amap_in, umap_in, imap_in, placement_in], outputs)
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
        self.warmup_lr = 1e-5
        self.learning_rate = 1e-3
        self.lr_min = 1e-5
        self.lr_drop = 10000
        self.epoch = 0
        self.update_epochs = 10
        self.updates_per_episode = 1
        self.regularization_loss_scale = 0.0

        self.n_step = 12  # n-step return horizon
        # Unity penalties already arrive charged to the tipped box and the boxes under it,
        # so spreading them back over earlier steps in time is off by default.
        self.unity_backprop_steps = 0
        self.unity_backprop_discount = 0.85

        self.batch_size = batch_size

        self.__train = train

        self.verbose = verbose
        self.visualize = visualize

        # Phase 0 diagnostic: last argmax-Q vector + its action space (None when exploring)
        self._last_q = None
        self._last_action_space = None

        if self.__train:
            self.q_net = q_net(k=env.k - 1)
            self.q_net_target = q_net(k=env.k - 1)
            # Initialize target network with same weights as online network
            self.q_net_target.set_weights(self.q_net.get_weights())
            self.q_optimizer = tf.keras.optimizers.Adam(learning_rate=self.warmup_lr, global_clipnorm=1.0)
            self.memory = PrioritizedReplayBuffer(maxlen=200000, alpha=0.6)
            self.per_beta = 0.4        # IS correction exponent, annealed toward 1
            self.per_beta_max = 1.0
            self.per_beta_anneal = 1e-4
        else:
            self.q_net = None
            self.q_net_target = None
            self.q_optimizer = None
            self.memory = None

    def select(self, state):
        items, h_maps, u_maps, boundary_maps, actions = state
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
            self._last_q = None  # exploration step: no argmax ranking to diagnose
            self._last_action_space = None
        else:
            q = self.Q(state)
            action = action_space[np.argmax(q)]
            r = np.max(q)
            self._last_q = np.asarray(q).reshape(-1)
            self._last_action_space = action_space

        return action, r, r != 0

    def _placement_choice_record(self, step, state, actual_reward, actual_maps):
        # Phase 0 diagnostic: compare the argmax-Q pick against the best-supported
        # candidate to tell whether Q mis-ranks well-supported placements.
        # actual_maps must be the physical height maps from BEFORE the step.
        q = self._last_q
        action_space = self._last_action_space
        if q is None or action_space is None or len(action_space) == 0:
            return None

        actions = state[-1]

        def describe(m):
            i, j, k = action_space[m]
            bin_index, (x, y, z), (w, h, d), _ = actions[i][j][k]
            footprint = actual_maps[bin_index][int(z):int(z + d), int(x):int(x + w)]
            rest_y = float(np.max(footprint))
            support = float(np.count_nonzero(footprint == rest_y)) / footprint.size
            return bin_index, (x, rest_y + self.env.v_pad, z), (w, h, d), support

        # Rank of each candidate by Q (0 = highest Q)
        rank_of = {int(idx): r for r, idx in enumerate(np.argsort(-q))}

        supports = [(m,) + describe(m) for m in range(len(action_space))]
        chosen_idx = int(np.argmax(q))
        chosen = supports[chosen_idx]
        # Best support; break ties toward the higher-Q candidate.
        best = max(supports, key=lambda t: (t[4], q[t[0]]))

        def entry(t, include_reward=False):
            m, bin_index, (x, y, z), (w, h, d), support = t
            e = {
                "q": round(float(q[m]), 4),
                "q_rank": int(rank_of[m]),
                "support": round(support, 4),
                "overhang": round(1.0 - support, 4),
                "x": round(float(x), 3), "y": round(float(y), 3), "z": round(float(z), 3),
                "w": round(float(w), 3), "h": round(float(h), 3), "d": round(float(d), 3),
            }
            if include_reward:
                e["reward"] = round(float(actual_reward), 4)
            return e

        return {
            "step": step,
            "bin": int(chosen[1]),
            "num_candidates": len(action_space),
            "chosen": entry(chosen, include_reward=True),
            "best_support": entry(best),
            "chosen_is_best_support": bool(chosen[0] == best[0]),
            "max_support_available": round(float(max(t[4] for t in supports)), 4),
        }

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
            if data.get('buffer') and len(data['buffer'][0]) != 6:
                print(f'Memory file {path} contains incompatible transitions.  starting with fresh replay buffer')
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

        items, h_maps, u_maps, boundary_maps, actions = state
        if action is None:
            action_space = indices(actions)
        else:
            i, j, k = action
            action_space = [(i, j, k)]

        imaps = [self.env.i_map(i, items) for i in range(len(self.env.packers))]

        hmap_in = []
        amap_in = []
        umap_in = []
        boundary_in = []
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

            # Boundary map: seam-aware mask of visible top-layer edges
            boundary = boundary_maps[j]

            # Resize to 32x32 for neural net
            amap = self._resize_map(amap)
            hmap = self._resize_map(hmap)
            umap = self._resize_map(umap)
            boundary = self._resize_map(boundary)

            imap = imaps[j][np.arange(len(items)) != i]

            hmap_in.append(hmap)
            amap_in.append(amap)
            umap_in.append(umap)
            boundary_in.append(boundary)
            imap_in.append(imap)
            # Explicit placement scalars bypass GlobalAveragePooling dilution.
            # The item footprint is ~2-4% of the 32x32 map; after GAP the height
            # signal at the footprint is drowned out by the other 96-98% of pixels.
            placement_in.append([x / W, y / H, z / D, w / W, h / H, d / D])

        hmap_in, amap_in, umap_in, boundary_in, imap_in = map(np.asarray, (hmap_in, amap_in, umap_in, boundary_in, imap_in))
        hmap_in = hmap_in[..., None]
        amap_in = amap_in[..., None]
        umap_in = umap_in[..., None]
        boundary_in = boundary_in[..., None]
        placement_in = np.array(placement_in, dtype=np.float32)

        return [boundary_in, hmap_in, amap_in, umap_in, imap_in, placement_in]

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

    def _build_n_step_transitions(self, history):
        transitions = []

        for start, (compact_inputs, _, _, _) in enumerate(history):
            reward_sum = 0.0
            horizon = 0
            terminal = False
            bootstrap_state = None

            for offset in range(min(self.n_step, len(history) - start)):
                _, reward, done, next_state = history[start + offset]
                reward_sum += (self.gamma ** offset) * reward
                horizon = offset + 1
                bootstrap_state = next_state
                if done:
                    terminal = True
                    break

            bootstrap_inputs = None
            bootstrap_discount = 0.0
            if not terminal and horizon == self.n_step and bootstrap_state is not None:
                action_space = indices(bootstrap_state[-1])
                if action_space:
                    online_q = self.Q(bootstrap_state, net=self.q_net).reshape(-1)
                    bootstrap_action = action_space[int(np.argmax(online_q))]
                    q_in = self.Q_inputs(bootstrap_state, bootstrap_action)
                    bootstrap_inputs = tuple(arr[0].copy() for arr in q_in)
                    bootstrap_discount = self.gamma ** horizon

            immediate_reward = history[start][1]
            transitions.append((
                compact_inputs,
                reward_sum,
                immediate_reward,
                bootstrap_inputs,
                bootstrap_discount,
                terminal,
            ))

        return transitions

    def train(self, history, is_weights=None):
        q_inputs = []
        q_targets = []
        bootstrap_rows = []
        bootstrap_inputs = []
        bootstrap_discounts = []

        for row, (compact_inputs, reward_sum, _, bootstrap_input, bootstrap_discount, _) in enumerate(history):
            q_inputs.append(list(compact_inputs))
            q_targets.append([reward_sum])
            if bootstrap_input is not None and bootstrap_discount > 0.0:
                bootstrap_rows.append(row)
                bootstrap_inputs.append(bootstrap_input)
                bootstrap_discounts.append(bootstrap_discount)

        if bootstrap_inputs:
            target_inputs = [
                np.asarray(values)
                for values in zip(*bootstrap_inputs)
            ]
            bootstrap_q = self.q_net_target(target_inputs, training=False).numpy().reshape(-1)
            for row, discount, target_q in zip(bootstrap_rows, bootstrap_discounts, bootstrap_q):
                q_targets[row][0] += discount * float(target_q)

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
            td_loss = tf.reduce_mean(is_weights_tf * tf.square(td_errors))
            regularization_loss = (
                tf.add_n(self.q_net.losses)
                if self.q_net.losses
                else tf.constant(0.0, dtype=td_loss.dtype)
            )
            applied_regularization_loss = self.regularization_loss_scale * regularization_loss
            loss = td_loss + applied_regularization_loss

        self._last_td_loss = float(td_loss.numpy())
        self._last_regularization_loss = float(applied_regularization_loss.numpy())
        self._last_raw_regularization_loss = float(regularization_loss.numpy())

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
        Safe to call at any time: processes whatever has finished so far.

        Unity evaluates an entire episode after its transitions have entered replay.
        Penalties arrive already attributed to the placements of tipped boxes and the
        boxes beneath them; unity_backprop_steps optionally also spreads them back in time.
        Every replay write is guarded by the episode's captured slot version, so
        overwritten slots are skipped rather than patched incorrectly.
        """
        if self.memory is None:
            return
        for ep_id, per_step_rewards, all_buf_indices in unity_evaluator.drain():
            active_priorities = self.memory.priorities[:self.memory.size]
            patch_priority = float(np.percentile(active_priorities, 90)) if self.memory.size > 0 else 1.0
            patch_priority = max(patch_priority, 1e-6)
            episode_len = len(all_buf_indices)
            adjustments = np.zeros(episode_len, dtype=np.float64)
            direct_penalties = 0

            # Aggregate first so overlapping backward windows update each immediate
            # reward once and n-step returns can be recomputed from final values.
            for k, step_reward in enumerate(per_step_rewards[:episode_len]):
                if step_reward >= 0.0:
                    continue
                direct_penalties += 1
                first_cause = max(0, k - self.unity_backprop_steps)
                for t in range(first_cause, k + 1):
                    distance = k - t
                    adjustments[t] += step_reward * (self.unity_backprop_discount ** distance)

            if not np.any(adjustments):
                print(f'[Unity] ep {ep_id}: no stability penalties to patch')
                continue

            slot_is_current = np.array([
                buf_idx < self.memory.size and self.memory._slot_version[buf_idx] == slot_ver
                for buf_idx, slot_ver in all_buf_indices
            ], dtype=bool)
            stale_skips = int(np.count_nonzero(~slot_is_current & (adjustments != 0.0)))
            immediate_patches = 0

            for t, adjustment in enumerate(adjustments):
                if adjustment == 0.0 or not slot_is_current[t]:
                    continue
                buf_idx, _ = all_buf_indices[t]
                ci, reward_sum, reward, bootstrap_input, bootstrap_discount, done = self.memory.buffer[buf_idx]
                self.memory.buffer[buf_idx] = (
                    ci, reward_sum, reward + float(adjustment),
                    bootstrap_input, bootstrap_discount, done,
                )
                self.memory.priorities[buf_idx] = patch_priority
                immediate_patches += 1

            # A late reward correction changes every n-step return whose reward
            # window includes that transition. Recompute only valid contiguous
            # episode windows; stale replay slots may now belong to another episode.
            affected_returns = set()
            for adjusted_step in np.flatnonzero(adjustments):
                first_return = max(0, adjusted_step - self.n_step + 1)
                affected_returns.update(range(first_return, adjusted_step + 1))

            return_patches = 0
            for start in affected_returns:
                start_buf_idx, _ = all_buf_indices[start]
                transition = self.memory.buffer[start_buf_idx]
                horizon = min(self.n_step, episode_len - start)
                stop = start + horizon
                if not np.all(slot_is_current[start:stop]):
                    continue
                reward_sum = 0.0
                for offset, t in enumerate(range(start, stop)):
                    buf_idx, _ = all_buf_indices[t]
                    reward_sum += (self.gamma ** offset) * self.memory.buffer[buf_idx][2]

                buf_idx, _ = all_buf_indices[start]
                ci, _, reward, bootstrap_input, bootstrap_discount, done = transition
                self.memory.buffer[buf_idx] = (
                    ci, reward_sum, reward,
                    bootstrap_input, bootstrap_discount, done,
                )
                self.memory.priorities[buf_idx] = patch_priority
                return_patches += 1

            print(
                f'[Unity] ep {ep_id}: {direct_penalties} direct penalties, '
                f'{immediate_patches} immediate/backward patches, '
                f'{return_patches} n-step returns recomputed, '
                f'{stale_skips} stale slots skipped, '
                f'total adjustment {adjustments.sum():.4f}'
            )

    def run(self, max_ep=1, verbose=False, train=None, unity_evaluator=None, unity_rewards=True, ep_offset=0):
        if train is None:
            train = self.__train

        iters = (i for i, _ in enumerate(iter(bool, True))) if max_ep == -1 else range(max_ep)

        for ep in iters:
            # === DRAIN COMPLETED UNITY RESULTS ===
            if unity_evaluator is not None and train and unity_rewards:
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

                items, h_map, u_maps, boundary_maps, actions = state
                if len(actions) == 0:
                    break
                action, r, exploited = self.select(state)

                if verbose:
                    print(f'possible actions: {len(actions)}')
                    print(f'action: {action}')
                    print(f'placement: {actions[action[0]][action[1]][action[2]]}')

                yield actions[action[0]][action[1]][action[2]]
                items_packed += 1

                bin_index, (px, py, pz), (pw, ph, pd), _ = actions[action[0]][action[1]][action[2]]
                # The padded py drifts 0.5 per layer; physics gets a constant one-v_pad drop instead
                release_y = self.env.release_height(bin_index, px, pz, pw, pd)
                actual_maps = [p.actual_height_map.copy() for p in self.env.packers] if self.visualize else None

                next_state, reward, done = self.env.step(action)

                # Track placement for Unity physics eval
                if unity_evaluator is not None:
                    s = 1
                    ep_placements.append({
                        'bin': int(bin_index),
                        'x': round(float(px * s), 3), 'y': round(release_y * s, 3), 'z': round(float(pz * s), 3),
                        'w': round(float(pw * s), 3), 'h': round(float(ph * s), 3), 'd': round(float(pd * s), 3),
                    })

                if train:
                    # Store compact NN inputs instead of full states (~17KB vs ~589KB per transition)
                    q_in = self.Q_inputs(state, action)
                    compact_inputs = tuple(arr[0].copy() for arr in q_in)
                    history.append((compact_inputs, reward, done, next_state))

                if self.visualize:
                    next_items, _, _, _, _ = next_state
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
                                "x": round(float(x * s), 3), "y": round(release_y * s, 3), "z": round(float(z * s), 3),
                                "w": round(float(w * s), 3), "h": round(float(h * s), 3), "d": round(float(d * s), 3)
                            }) + "\n")
                        with open(f"./outputs/bin{ep}/boxRewards.jsonl", "a") as f:
                            placement = actions[action[0]][action[1]][action[2]]
                            bin_index, (x, y, z), (w, h, d), split = placement
                            s = 1
                            entry = {
                                "step": step,
                                "bin": int(bin_index),
                                "x": round(float(x * s), 3), "y": round(release_y * s, 3), "z": round(float(z * s), 3),
                                "w": round(float(w * s), 3), "h": round(float(h * s), 3), "d": round(float(d * s), 3),
                                "reward": round(float(reward), 4),
                                "agent": str(exploited),
                            }
                            if hasattr(self.env, '_reward_breakdown'):
                                entry['reward_breakdown'] = self.env._reward_breakdown
                            f.write(json.dumps(entry) + "\n")

                    choice_record = self._placement_choice_record(step, state, reward, actual_maps)
                    if choice_record is not None:
                        with open(f"./outputs/bin{ep}/placementChoices.jsonl", "a") as f:
                            f.write(json.dumps(choice_record) + "\n")

                ep_reward += reward
                if done:
                    break
                state = next_state

            loss = None
            all_buf_indices = []
            if train:
                transitions = self._build_n_step_transitions(history)
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
                unity_evaluator.submit(ep_placements, all_buf_indices, ep_offset + ep,
                                       supporters=list(self.env.placement_supporters))
                print(f'[Unity] ep {ep_offset + ep}: submitted {len(ep_placements)} placements for physics eval')

            if train and len(self.memory) > 1000:
                n_updates = self.updates_per_episode
                print(f'update model x{n_updates} ({len(history)} new transitions)')
                for _ in range(n_updates):
                    beta = min(self.per_beta_max, self.per_beta + self.epoch * self.per_beta_anneal)
                    sampled_history, sample_indices, is_weights = self.memory.sample(128, beta=beta)
                    loss, td_errors = self.train(sampled_history, is_weights=is_weights)
                    self.memory.update_priorities(sample_indices, td_errors)

            self.ep_history.append(([packer.space_utilization() for packer in self.env.used_packers], self.env.used_bins, ep_reward, items_packed))

            yield None

            utils = [round(packer.space_utilization() * 100, 2) for packer in self.env.used_packers]
            if self.verbose:
                td_loss = getattr(self, '_last_td_loss', None)
                regularization_loss = getattr(self, '_last_regularization_loss', None)
                print(
                    f'Episode {ep}, util: {utils}, used bins: {self.env.used_bins}, '
                    f'ep_reward: {ep_reward:.2f}, memory: {len(self.memory) if self.memory is not None else None}, '
                    f'eps: {self.eps:.2f}, loss: {loss}, td_loss: {td_loss}, '
                    f'l2_loss: {regularization_loss}, '
                    f'lr: {self.q_optimizer.learning_rate.numpy() if self.q_optimizer is not None else None}'
                )

class HeuristicAgent:
    def __init__(self, heuristic, env=MultiBinPackerEnv(n_bins=2, max_bins=-1, size=(32, 32, 32), k=10, verbose=True), verbose=True, visualize=False):
        self.env = env

        self.heuristic = heuristic
        self.ep_history = []

        self.verbose = verbose
        self.visualize = visualize

    def select(self, state):
        # state = (items, h_map, u_maps, actions)

        items, h_map, u_maps, boundary_maps, actions = state
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

                items, h_map, u_maps, boundary_maps, actions = state
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
