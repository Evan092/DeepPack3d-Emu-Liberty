import numpy as np
from conveyor import *
from SpacePartitioner import *
from geometry import *
from binpacker import *
import math

class Env:
    def __init__(self, verbose):
        self.verbose = verbose
        self._state = None
        
    def reset(self):
        raise Exception('not implemented')
        
    def state(self, step=False):
        raise Exception('not implemented')
        
    def step(self, action):
        raise Exception('not implemented')
        
    def actions(self):
        raise Exception('not implemented')
        
def indices(actions):
    return [
        (i, j, k) 
        for i in range(len(actions)) 
        for j in range(len(actions[i])) 
        for k in range(len(actions[i][j]))
    ]

def dead_space(a_map, rows, cols, min_dim, pad):
    """Volume of pits along the given rows/columns that are wider than the padding gap but too narrow to fill."""
    dead = 0.0
    for line in [a_map[r] for r in rows] + [a_map[:, c] for c in cols]:
        n = line.size
        edges = np.flatnonzero(np.diff(line)) + 1
        for s, e in zip(np.r_[0, edges], np.r_[edges, n]):
            # A pit is a flat run lower than both neighbours; the pallet edge counts as a wall
            rim = min(line[s - 1] if s > 0 else np.inf, line[e] if e < n else np.inf)
            # Between two boxes the left box's clearance is inside the run, plus the new box's own clearance
            need = min_dim + pad * (2 if 0 < s and e < n else 1)
            if np.isfinite(rim) and rim > line[s] and pad < e - s < need:
                dead += (e - s) * (rim - line[s])
    return dead

class MultiBinPackerEnv(Env):
    def __init__(self, n_bins, size, k=1, max_bins=None, max_items=None, replace='all', verbose=False, prealloc_bins=0, prealloc_items=0, shuffle=False, use_rotate=True, use_skip=True, h_pad=1, v_pad=0.5, min_support=0.5):
        super().__init__(verbose)
        self.n_bins = n_bins
        self.size = size
        self.k = k
        
        # Padding configuration
        # h_pad: horizontal padding (x, z) between boxes, in whole grid cells. realistic for real-world replication
        # v_pad: vertical padding (y) between stacked boxes and below first layer. minimal for physics sim
        # v_pad does NOT count against bin height: items "fall" into place in the real world
        self.h_pad = h_pad
        self.v_pad = v_pad

        # Minimum fraction of a box's footprint that must rest at its physical support height
        self.min_support = min_support
        
        # Scale factor for converting grid coords back to real-world units
        # Default 1.0, overridden when loading from file
        self.inv_scale = 1.0
        
        W, H, D = size
        self.v_pad_headroom = (H + 1) * v_pad
        
        self.packers = [SpacePartitioner(size, floor_height=v_pad, v_pad_headroom=self.v_pad_headroom, h_pad_headroom=h_pad) for _ in range(n_bins)]
        self.conveyor = Conveyor(k, 
                                 max_items=max_items, 
                                 prealloc_bins=prealloc_bins, 
                                 prealloc_items=prealloc_items, 
                                 shuffle=shuffle
                                )
        
        self.max_bins = -1 if max_bins is None else max_bins
        self.used_bins = n_bins
        
        self.used_packers = [] + self.packers
        self.replace = replace # {'max', 'min', 'all'}
        
        self.use_rotate = use_rotate
        self.use_skip = use_skip
        
        # Track items placed in current bin for scheduler
        self.items_in_current_bin = 0

        # Persistent color per conveyor slot so placed boxes keep their color
        self._item_colors = []

        # Per-bin seam/bridging observation and backend maps
        self.boundary_maps = [np.zeros((size[2], size[0]), dtype=np.float32) for _ in range(n_bins)]
        self.id_maps = [np.full((size[2], size[0]), -1, dtype=np.int32) for _ in range(n_bins)]
        self.item_counter = 0
        # History of interweave calculations for debugging/analysis
        self._interweave_log = []
        # Per placement: {placement index of a supporting box: share of the load-bearing area}
        self.placement_supporters = []
        # Per bin: last value of each state-quality potential, for potential-based shaping
        self._potentials = [{} for _ in range(n_bins)]
        
    def reset(self):
        for packer in self.packers:
            packer.reset()
        self.conveyor.reset()
        
        self._state = None
        self.used_bins = self.n_bins
        self.used_packers = []
        self.items_in_current_bin = 0
        self._item_colors = []
        self.boundary_maps = [np.zeros((self.size[2], self.size[0]), dtype=np.float32) for _ in range(self.n_bins)]
        self.id_maps = [np.full((self.size[2], self.size[0]), -1, dtype=np.int32) for _ in range(self.n_bins)]
        self.item_counter = 0
        self._interweave_log = []
        self.placement_supporters = []
        self._potentials = [{} for _ in range(self.n_bins)]
        return self.state()

    def release_height(self, j, x, z, w, d):
        """Height to release a box from for physics: one v_pad above where it will physically rest."""
        footprint = self.packers[j].actual_height_map[int(z):int(z + d), int(x):int(x + w)]
        return float(np.max(footprint)) + self.v_pad
    
    @staticmethod
    def _random_item_color():
        r, g, b = np.random.random(3) * 0.65 + 0.35
        return (float(r), float(g), float(b), 0.75)

    def state(self, step=False):
        if step or self._state is None:
            items = list(self.conveyor.peek())
            # Assign a stable color to each newly visible conveyor slot
            while len(self._item_colors) < len(items):
                self._item_colors.append(self._random_item_color())
            h_maps = np.array(self._height_maps())
            u_maps = self._utilization_maps()
            boundary_maps = np.array(self._boundary_maps())
#             print(items)
#             print(self.actions(items, h_maps, rotate=True, skip=True))
            actions = list(self.actions(items, h_maps, rotate=self.use_rotate, skip=self.use_skip))
    
            self._state = (items, h_maps, u_maps, boundary_maps, actions)
        
        return self._state
    
    def _height_maps(self):
        return [packer.height_map.copy() for packer in self.packers]
    
    def _actual_height_maps(self):
        return [packer.actual_height_map.copy() for packer in self.packers]
    
    def _boundary_maps(self):
        return [boundary_map.copy() for boundary_map in self.boundary_maps]
    
    def _utilization_maps(self):
        """For each bin, compute a 2D map where each cell = fraction of space
        below the height map surface that is actually filled by items.
        1.0 = solid column, 0.0 = empty below surface. Empty = 1.0"""
        u_maps = []
        for packer in self.packers:
            h_map = packer.actual_height_map
            W, H, D = packer.size
            rd, rw = h_map.shape  # (D, W)
            
            # Count filled height at each (x, z) column from placed items
            filled = np.zeros((rd, rw), dtype=np.float32)
            for item in packer.actual_splits:
                ix, iy, iz = int(item.x), item.y, int(item.z)
                iw, ih, id_ = int(item.width), item.height, int(item.depth)
                # Clamp to bin bounds
                x1 = min(ix + iw, rw)
                z1 = min(iz + id_, rd)
                filled[iz:z1, ix:x1] += ih
            
            # Utilization = filled height / surface height
            u_map = np.where(h_map > self.v_pad, filled / (h_map + 1e-9), 1.0)
            u_map = np.clip(u_map, 0.0, 1.0)
            u_maps.append(u_map.copy())
        
        return u_maps
    
    def placeable_coords_old(self, packer, h_map, size):
        w, h, d = size
        w_pad = w + self.h_pad
        h_pad = h + self.v_pad
        d_pad = d + self.h_pad
        padded_size = (w_pad, h_pad, d_pad)
        
        xz = []
        splits = {}
        for split in packer.free_splits:
            if (split.top < self.size[1]) or (not split.fit(padded_size)):
                continue
            sx, sy, sz = split.coord
            
            # The 4 corners inside the free split for the box to sit flush against the boundaries.
            sw = split.width
            sd = split.depth
            
            split_candidates = [
                (sx, sz),                                      # Bottom-Left (Flush Left/Back)
                (sx + sw - w_pad, sz),                         # Bottom-Right (Flush Right/Back)
                (sx, sz + sd - d_pad),                         # Top-Left (Flush Left/Front)
                (sx + sw - w_pad, sz + sd - d_pad)             # Top-Right (Flush Right/Front)
            ]
            
            for cx, cz in split_candidates:
                # Snap to integer grid (items are integer-sized, height map is integer-indexed)
                x = int(np.ceil(cx))
                z = int(np.ceil(cz))
                
                # Re-check padded item still fits within bin after snapping
                if x < 0 or z < 0 or x + w_pad > self.size[0] or z + d_pad > self.size[2]:
                    continue
                
                xz.append((x, z))
                if (x, z) not in splits:
                    splits[(x, z)] = split
        
        # Free-split corners capture the anchors per split region.
        # Placed-item corners are the natural flat-landing starts: the top surface
        # of every already-placed item is a valid floor for the next item.
        # Add all four horizontal corners of each placed item so we don't miss
        # positions that are inside a large free split but far from its corner.
        for placed in packer.actual_splits:
            for cx in (int(placed.x), int(placed.right)):
                for cz in (int(placed.z), int(placed.front)):
                    if cx + w_pad <= self.size[0] and cz + d_pad <= self.size[2]:
                        xz.append((cx, cz))
                        # map to whichever free split owns this (x,z); packer.fit() will validate
                        if (cx, cz) not in splits:
                            splits[(cx, cz)] = None

        # Removes any duplicate candidates generated from overlapping splits or identical box/split corners
        xz = set(xz)
        
        xyz = []
        for x, z in xz:
            placement = h_map[z:z + d, x:x + w]
            y = np.amax(placement)
            # Check actual (non-padded) height fits within bin.
            actual_placement = packer.actual_height_map[z:z + d, x:x + w]
            actual_y = np.amax(actual_placement)
            if actual_y + h > self.size[1]:
                continue
            # verify the padded cuboid actually fits in the 3D space partitioner
            padded_cuboid = Cuboid(x, y, z, w_pad, h_pad, d_pad)
            if not packer.fit(padded_cuboid):
                continue
            # placement and stability constraints
            if np.count_nonzero(placement == y) / (d * w) > 0.5:
                split = splits.get((x, z))
                if split is None:
                    # Item-corner candidate: find the free split that contains it
                    for fs in packer.free_splits:
                        if fs.contain(padded_cuboid):
                            split = fs
                            splits[(x, z)] = fs
                            break
                if split is None:
                    continue  # no containing split found (shouldn't happen after packer.fit passed)
                xyz.append((x, y, z, split))
        
        return xyz


    def placeable_coords(self, packer, h_map, size):
        w, h, d = size
        w_pad, h_pad, d_pad = w + self.h_pad, h + self.v_pad, d + self.h_pad
        padded_size = (w_pad, h_pad, d_pad)
        
        # Pre-calculate base area for the stability check
        base_area = w * d 
        
        # Drop the `xz` list. A dictionary's keys naturally act as a set, 
        # so we can build it directly and inherently prevent duplicates.
        splits = {}
        
        for split in packer.free_splits:
            if (split.top < self.size[1]) or (not split.fit(padded_size)):
                continue
            
            sx, sy, sz = split.coord
            sw, sd = split.width, split.depth
            
            # Near corners round up and far corners round down so snapping never pushes a box out of its split
            for x in (math.ceil(sx), math.floor(sx + sw - w_pad)):
                for z in (math.ceil(sz), math.floor(sz + sd - d_pad)):
                    # Only the physical box must be on the pallet; its clearance may hang past the edge
                    if 0 <= x and 0 <= z and x + w <= self.size[0] and z + d <= self.size[2]:
                        splits.setdefault((x, z), split)
        
        for placed in packer.actual_splits:
            for cx in (int(placed.x), int(placed.right)):
                for cz in (int(placed.z), int(placed.front)):
                    if cx + w <= self.size[0] and cz + d <= self.size[2]:
                        splits.setdefault((cx, cz), None)

        xyz = []
        
        # Iterate directly over the deduplicated dictionary items
        for (x, z), split in splits.items():
            
            # 1. CHEAPEST CHECK: Does the unpadded height fit?
            actual_placement = packer.actual_height_map[z:z + d, x:x + w]
            actual_y = np.amax(actual_placement)
            if actual_y + h > self.size[1]:
                continue
                
            # 2. CHEAP CHECK: Is enough of the footprint resting at the physical support height?
            # Padded heights drift 0.5 per layer, so they can't be used to judge support
            if np.count_nonzero(actual_placement == actual_y) / base_area < self.min_support:
                continue

            placement = h_map[z:z + d, x:x + w]
            y = np.amax(placement)

            # 3. EXPENSIVE CHECK: Does the padded 3D cuboid collide with anything?
            padded_cuboid = Cuboid(x, y, z, w_pad, h_pad, d_pad)
            if not packer.fit(padded_cuboid):
                continue
            
            # 4. RESOLVE: Find the containing split if it was a placed-item corner
            if split is None:
                for fs in packer.free_splits:
                    if fs.contain(padded_cuboid):
                        split = fs
                        break
                if split is None:
                    continue  
                    
            xyz.append((x, y, z, split))
        
        return xyz
    
    def actions(self, items, h_maps, rotate, skip):
        actions = []
        
        for item in items:
            if item is None:
                continue
            
            bin_actions = []
            for i, packer in enumerate(self.packers):
                item_actions = []
#                 print(items)
                for size in rotated_sizes(item, rotate):
                    for x, y, z, split in self.placeable_coords(packer, h_maps[i], size):
                        item_actions.append((i, (x, y, z), size, split))
                bin_actions.append(item_actions)
            actions.append(bin_actions)
                
            # always pick the first available item
            if not skip:
                return actions
            
        return actions
    











    def step(self, action):
            items, h_maps, u_maps, boundary_maps_state, actions = self.state()
            
            # item, bin, rotation_placement
            i, j, k = action
            _, (x, y, z), (w, h, d), _ = actions[i][j][k]
            
            packer = self.packers[j]

            # Physical heights BEFORE placing the item; reward terms never compare padded heights
            old_a_map = packer.actual_height_map.copy()
            rest_y = float(np.max(old_a_map[z:z + d, x:x + w]))

            actual_cuboid = Cuboid(x, y, z, w, h, d)
            padded_cuboid = Cuboid(x, y, z, w + self.h_pad, h + self.v_pad, d + self.h_pad)
            item_color = self._item_colors.pop(i) if i < len(self._item_colors) else None
            if not packer.add(padded_cuboid, actual_cuboid=actual_cuboid, color=item_color):
                raise Exception(f'invalid space {actual_cuboid}')

            self.conveyor.grab(i)
            
            # --- 6. INTERWEAVING (Seam Bridging) ---
            # Calculate seam bridging reward BEFORE the new placement overwrites the id_map
            boundary_map = self.boundary_maps[j]
            id_map = self.id_maps[j]
            
            # Track only the physical footprint. Including clearance padding here can
            # pull in an adjacent box ID that does not support the placed item.
            z0 = int(math.floor(z))
            z1 = int(math.ceil(z + d))
            x0 = int(math.floor(x))
            x1 = int(math.ceil(x + w))

            # Clamp to bounds (upper bounds may equal shape for slicing)
            z0 = max(0, min(z0, id_map.shape[0] - 1))
            z1 = max(z0, min(z1, id_map.shape[0]))
            x0 = max(0, min(x0, id_map.shape[1] - 1))
            x1 = max(x0, min(x1, id_map.shape[1]))

            # If the physical footprint is empty after clamping, or the placement
            # sits on the floor, there is no interweaving benefit.
            if z1 <= z0 or x1 <= x0:
                interweave_score = 0.0
            elif rest_y <= 0:
                interweave_score = 0.0 # Floor placements don't bridge seams
            else:
                support_ids = id_map[z0:z1, x0:x1]
                # Only count cells that actually bear load at THIS placement's height.
                # Without this check, a footprint spanning multiple ids at very
                # different (lower) heights would still score as "bridging",
                # rewarding overhangs instead of genuine seam bridging.
                support_heights = old_a_map[z0:z1, x0:x1]
                load_bearing_mask = np.isclose(support_heights, rest_y)
                valid_mask = (support_ids >= 0) & load_bearing_mask
                if np.any(valid_mask):
                    unique_ids = np.unique(support_ids[valid_mask])
                    if unique_ids.size > 1:
                        counts = np.bincount(support_ids[valid_mask], minlength=self.item_counter + 1)
                        counts = counts[unique_ids]
                        area_frac = counts / counts.sum()
                        evenness = 1.0 - np.sum((area_frac - (1.0 / unique_ids.size)) ** 2) / max(1e-9, 1.0 - (1.0 / unique_ids.size))
                        # Scale by how much of the footprint is actually load-bearing at
                        # this height, so partially-overhanging bridges score lower.
                        total_footprint_cells = max(1, (z1 - z0) * (x1 - x0))
                        support_fraction = counts.sum() / total_footprint_cells
                        # Square both factors so a perfect bridge (full support, even split)
                        # is rewarded disproportionately more than a near-perfect one.
                        interweave_score = (support_fraction ** 2) * (0.6 * (evenness ** 2) + 0.4)
                    else:
                        interweave_score = 0.0
                else:
                    interweave_score = 0.0

            # Log interweave diagnostics for debugging
            try:
                unique_ids_list = [int(u) for u in unique_ids]
            except Exception:
                unique_ids_list = None
            try:
                counts_list = [int(c) for c in counts]
            except Exception:
                counts_list = None
            try:
                area_frac_list = [float(a) for a in area_frac]
            except Exception:
                area_frac_list = None
            evenness_val = float(evenness) if 'evenness' in locals() else None

            record = {
                'bin': j,
                'item_counter': int(self.item_counter),
                'z0': int(z0), 'z1': int(z1), 'x0': int(x0), 'x1': int(x1),
                'unique_ids': unique_ids_list,
                'counts': counts_list,
                'area_frac': area_frac_list,
                'evenness': evenness_val,
                'interweave_score': float(interweave_score),
            }
            self._interweave_log.append(record)
            if self.verbose:
                print('Interweave:', record)

            # Boxes this placement rests on, weighted by load-bearing area, so Unity can blame weak foundations
            supporters = {}
            if rest_y > 0:
                bearing_ids = id_map[z0:z1, x0:x1][np.isclose(old_a_map[z0:z1, x0:x1], rest_y)]
                bearing_ids = bearing_ids[bearing_ids >= 0]
                if bearing_ids.size:
                    ids, areas = np.unique(bearing_ids, return_counts=True)
                    supporters = {int(s): float(a) / bearing_ids.size for s, a in zip(ids, areas)}
            self.placement_supporters.append(supporters)

            # Vectorized boundary-map update: clear the old footprint, then draw the new inner border
            boundary_map[z0:z1, x0:x1] = 0.0
            if z1 - z0 > 0 and x1 - x0 > 0:
                boundary_map[z0, x0:x1] = 1.0
                boundary_map[z1 - 1, x0:x1] = 1.0
                boundary_map[z0:z1, x0] = 1.0
                boundary_map[z0:z1, x1 - 1] = 1.0

            # Vectorized id-map update
            id_map[z0:z1, x0:x1] = self.item_counter
            self.item_counter += 1

            # Increment items placed in current bin
            self.items_in_current_bin += 1
            
            # Capture next state (which now includes the correctly updated boundary_maps)
            next_state = self.state(step=True)
            items, h_maps, u_maps, boundary_maps_state, actions = next_state
            
            item = items[i]
            volume = np.sum([split.volume for split in packer.actual_splits])

            # === REWARD COMPONENTS ===
            
            W, H, D = packer.size
            actual_h_map = packer.actual_height_map
            max_h = np.max(actual_h_map)

            # 1. Surface Variance
            if max_h == 0:
                surface_flatness = 0.0
            else:
                h_map_std = np.std(actual_h_map)
                max_possible_std = H / 2.0
                raw_flatness = max(1.0 - (h_map_std / max_possible_std), 0.0)
                filled_cells = np.sum(actual_h_map > self.v_pad)
                coverage = filled_cells / actual_h_map.size
                surface_flatness = raw_flatness * np.sqrt(coverage)
                
                filled_mask = actual_h_map > self.v_pad
                if np.any(filled_mask):
                    filled_std = np.std(actual_h_map[filled_mask])
                    stackability = 1.0 - min(filled_std / (H / 4.0), 1.0)
                else:
                    stackability = 1.0

            # 2. Center Proximity
            if max_h == 0:
                center_proximity = 0.0
            else:
                xs, zs = np.meshgrid(np.arange(W), np.arange(D))
                center_x, center_z = (W - 1) / 2.0, (D - 1) / 2.0
                dist_to_center = np.sqrt((xs - center_x)**2 + (zs - center_z)**2)
                max_dist = np.sqrt(center_x**2 + center_z**2)
                normalized_dist = dist_to_center / (max_dist + 1e-9)
                
                total_height = np.sum(actual_h_map)
                if total_height > 0:
                    center_proximity = 1.0 - np.sum(normalized_dist * actual_h_map) / total_height
                else:
                    center_proximity = 0.0

            # 3. Volume Efficiency
            if max_h == 0:
                volume_efficiency = 0.0
            else:
                volume_efficiency = volume / (W * max_h * D)

            # 4. Gap Fill Metric
            neighborhood_radius = max(w, d)
            nz_start = max(0, int(z) - neighborhood_radius)
            nz_end = min(D, int(z + d) + neighborhood_radius)
            nx_start = max(0, int(x) - neighborhood_radius)
            nx_end = min(W, int(x + w) + neighborhood_radius)
            
            neighborhood = old_a_map[nz_start:nz_end, nx_start:nx_end]
            
            if neighborhood.size > 0 and np.max(neighborhood) > 0:
                gap_score = (np.mean(neighborhood) - rest_y) / (np.max(old_a_map) + 1e-9)
                gap_fill = max(0.0, min(1.0, gap_score + 0.5))
            else:
                gap_fill = 0.5

            # 4b. Gap Creation Penalty: dead space newly trapped in pits too narrow for any visible item.
            # Only the rows and columns through the footprint can change, so only those are measured.
            footprint_dims = [min(itm[0], itm[2]) for itm in items if itm is not None]
            min_dim = min(footprint_dims) if footprint_dims else min(w, d)
            rows, cols = range(z, z + d), range(x, x + w)
            created = (dead_space(actual_h_map, rows, cols, min_dim, self.h_pad)
                       - dead_space(old_a_map, rows, cols, min_dim, self.h_pad))
            gap_creation_penalty = min(1.0, max(0.0, created) / (w * h * d))

            # 5. Same-Height Adjacency
            x_int, z_int = int(x), int(z)
            w_int, d_int = int(w), int(d)
            actual_placed_top = np.max(packer.actual_height_map[z_int:z_int + d_int, x_int:x_int + w_int])

            search_radius = 2
            n_x_start = max(0, x_int - search_radius)
            n_x_end = min(W, x_int + w_int + search_radius)
            n_z_start = max(0, z_int - search_radius)
            n_z_end = min(D, z_int + d_int + search_radius)

            neighborhood_adj = old_a_map[n_z_start:n_z_end, n_x_start:n_x_end]

            exclude_self = np.ones(neighborhood_adj.shape, dtype=bool)
            rel_z = z_int - n_z_start
            rel_x = x_int - n_x_start
            exclude_self[rel_z:rel_z + d_int, rel_x:rel_x + w_int] = False

            same_height = (np.abs(neighborhood_adj - actual_placed_top) <= 0.5) & (neighborhood_adj > 0) & exclude_self
            total_neighbor_cells = int(np.sum(exclude_self))

            if total_neighbor_cells > 0:
                adjacency = float(np.sum(same_height)) / total_neighbor_cells
            else:
                adjacency = 0.0

            # Note: Section 6 (Interweaving) is executed at the top of the function to capture old IDs.

            # 7. Support IoU
            if rest_y <= 0:
                support_iou = 1.0
            else:
                footprint = old_a_map[int(z):int(z + d), int(x):int(x + w)]
                total_cells = w * d
                if total_cells > 0:
                    supported_cells = np.sum(np.isclose(footprint, rest_y))
                    support_fraction = supported_cells / total_cells
                    overhang_score = support_fraction ** 2
                    
                    expand = 1
                    sx_start = max(0, int(x) - expand)
                    sx_end = min(W, int(x + w) + expand)
                    sz_start = max(0, int(z) - expand)
                    sz_end = min(D, int(z + d) + expand)
                    support_region_old = old_a_map[sz_start:sz_end, sx_start:sx_end]
                    platform_mask = np.isclose(support_region_old, rest_y)
                    platform_cells = np.sum(platform_mask)
                    
                    if platform_cells > 0:
                        support_region_new = actual_h_map[sz_start:sz_end, sx_start:sx_end]
                        covered = np.sum((support_region_new > rest_y + 0.1) & platform_mask)
                        surface_utilization = covered / platform_cells
                    else:
                        surface_utilization = 0.0
                    
                    support_iou = 0.7 * overhang_score + 0.3 * surface_utilization
                    height_decay = 1.0 - 0.5 * (rest_y / (H + 1e-9))
                    support_iou *= max(height_decay, 0.4)
                else:
                    support_iou = 0.0

            # === Footprint Coverage ===
            filled_cells = np.sum(actual_h_map > self.v_pad)
            linear_coverage = filled_cells / actual_h_map.size
            footprint_coverage = linear_coverage ** 1.5

            # === Big Bottom ===
            item_volume = w * h * d
            conveyor_volumes = [iw * ih * id_ for itm in items if itm is not None for iw, ih, id_ in [itm]]
            max_visible_volume = max(conveyor_volumes) if conveyor_volumes else item_volume
            max_visible_volume = max(max_visible_volume, item_volume)
            volume_fraction = item_volume / (max_visible_volume + 1e-9)
            height_fraction = rest_y / (H + 1e-9)
            big_bottom = (1.0 - height_fraction) * volume_fraction

            # === Layer Completion ===
            filled_mask = actual_h_map > self.v_pad
            if np.any(filled_mask):
                filled_heights = actual_h_map[filled_mask]
                max_filled = np.max(filled_heights)
                layer_completion = np.mean(filled_heights) / (max_filled + 1e-9) if max_filled > 0 else 0.0
            else:
                layer_completion = 0.0

            # Height Penalty
            height_penalty = (rest_y / (H + 1e-9)) ** 2

            # === DYNAMIC SCHEDULER ===
            progress = min(self.items_in_current_bin / 50.0, 1.0)
            
            flatnessScale = (1.5 * (1.0 - progress) + 0.5 * progress) * 2
            stackScale = 1
            centerScale = 1.5 * (1.0 - progress) + 0.3 * progress
            volumeScale = 0.3 * (1.0 - progress) + 1.5 * progress
            gapScale = 0.7 * (1.0 - progress) + 1.5 * progress
            gapCreationScale = 2.0 * (1.0 - progress) + 3.0 * progress
            adjacencyScale = 2
            interweaveScale = 0.0
            supportScale = 0.5 * (1.0 - progress) + 2.0 * progress
            coverageScale = 6.0 * (1.0 - progress) + 3.0 * progress
            bigBottomScale = 3.0 * (1.0 - progress) + 0.5 * progress
            layerScale = 3.0 * (1.0 - progress) + 2.0 * progress
            heightPenaltyScale = 3.0 * (1.0 - progress) + 12.0 * progress
            packScale = 100.0

            # Overhang Additive Penalty
            overhang_additive = 0.0
            if rest_y > 0:
                support_mask = np.isclose(old_a_map[int(z):int(z + d), int(x):int(x + w)], rest_y).astype(float)
                support_area = np.sum(support_mask)
                total_item_area = w * d
                
                if total_item_area > 0:
                    overhang_ratio = 1.0 - (support_area / total_item_area)
                    mid_x, mid_z = w / 2.0, d / 2.0
                    
                    q1 = np.sum(support_mask[:int(np.ceil(mid_z)), :int(np.ceil(mid_x))])
                    q2 = np.sum(support_mask[:int(np.ceil(mid_z)), int(np.floor(mid_x)):])
                    q3 = np.sum(support_mask[int(np.floor(mid_z)):, :int(np.ceil(mid_x))])
                    q4 = np.sum(support_mask[int(np.floor(mid_z)):, int(np.floor(mid_x)):])
                    
                    quadrants = np.array([q1, q2, q3, q4])
                    
                    if support_area > 0:
                        quadrant_fracs = quadrants / support_area
                        asymmetry = np.std(quadrant_fracs) / np.std([1.0, 0.0, 0.0, 0.0])
                    else:
                        asymmetry = 1.0
                    
                    sig = (1 / (1 + np.exp((-18 * overhang_ratio) + 5.5)))
                    asymmetry_amp = 1.0 + 0.5 * asymmetry * overhang_ratio
                    support_utilization = np.mean(u_maps[j][int(z):int(z + d), int(x):int(x + w)])
                    foundation_amp = 1.0 + 0.3 * (1.0 - support_utilization)
                    
                    overhang_scale = 3.0 * (1.0 - progress) + 6.0 * progress
                    overhang_additive = overhang_scale * sig * asymmetry_amp * foundation_amp

            # State-quality terms are paid as changes in a potential, so over an episode they telescope
            # and can't make ending it early worth more than packing another box
            potentials = {
                'flatness': flatnessScale * surface_flatness,
                'stackability': stackScale * stackability,
                'center': centerScale * center_proximity,
                'volume_eff': volumeScale * volume_efficiency,
                'coverage': coverageScale * footprint_coverage,
                'layer_completion': layerScale * layer_completion,
            }
            shaping = {key: value - self._potentials[j].get(key, 0.0) for key, value in potentials.items()}
            self._potentials[j] = potentials

            credits = {
                'packing': packScale * item_volume / (W * H * D),
                'gap_fill': gapScale * gap_fill,
                'adjacency': adjacencyScale * adjacency,
                'interweaving': interweaveScale * interweave_score,
                'support_iou': supportScale * support_iou,
                'big_bottom': bigBottomScale * big_bottom,
            }
            penalties = {
                'gap_penalty': gapCreationScale * gap_creation_penalty,
                'height_penalty': heightPenaltyScale * height_penalty,
                'overhang_penalty': overhang_additive,
            }
            # Penalties may cut a placement's credit by at most 90%, so every legal placement is worth packing
            credit_total = sum(credits.values())
            penalty_total = sum(penalties.values())
            penalty_cap = max(0.0, penalty_total - 0.9 * credit_total)

            reward = credit_total - penalty_total + penalty_cap + sum(shaping.values())

            self._reward_breakdown = {key: round(float(value), 4) for key, value in {**shaping, **credits}.items()}
            self._reward_breakdown.update({key: round(float(-value), 4) for key, value in penalties.items()})
            self._reward_breakdown['penalty_cap'] = round(float(penalty_cap), 4)
                    
            done = len(indices(actions)) == 0
            
            if done:
                total_volume = W * H * D
                reward += 10.0 * ((volume / total_volume) - 0.5)
                
                if self.max_bins != -1 and self.used_bins + 1 > self.max_bins:
                    for i, packer in enumerate(packer for packer in self.packers if packer.space_utilization() != 0):
                        self.used_packers.append(packer)
                    done = True
                else:
                    if self.replace == 'max':
                        loc = np.argmax([packer.space_utilization() for packer in self.packers])
                        packer = self.packers[loc]
                        self.used_packers.append(self.packers[loc])
                        self.packers[loc] = SpacePartitioner(self.size, floor_height=self.v_pad, v_pad_headroom=self.v_pad_headroom, h_pad_headroom=self.h_pad)
                        self.packers[loc].reset()
                        self._potentials[loc] = {}
                        
                        # FIX: Prevent the state leak by clearing the new bin's maps
                        self.boundary_maps[loc].fill(0.0)
                        self.id_maps[loc].fill(-1)
                        
                        self.items_in_current_bin = 0
                        added = 1
                        self.used_bins += 1
                    elif self.replace == 'all':
                        added = 0
                        while True:
                            loc = np.argmax([packer.space_utilization() for packer in self.packers])
                            packer = self.packers[loc]
                            
                            if packer.space_utilization() == 0:
                                break
                            if self.max_bins != -1 and self.used_bins + 1 > self.max_bins:
                                break

                            self.used_packers.append(self.packers[loc])
                            self.packers[loc] = SpacePartitioner(self.size, floor_height=self.v_pad, v_pad_headroom=self.v_pad_headroom, h_pad_headroom=self.h_pad)
                            self.packers[loc].reset()
                            self._potentials[loc] = {}
                            
                            # FIX: Prevent the state leak by clearing the new bin's maps
                            self.boundary_maps[loc].fill(0.0)
                            self.id_maps[loc].fill(-1)
                            
                            self.items_in_current_bin = 0
                            added += 1
                            self.used_bins += 1
                    else:
                        raise Exception('not implemented')
                    
                    next_state = self.state(step=True)
                    items, h_maps, u_maps, boundary_maps_state, actions = next_state
                    done = len(indices(actions)) == 0
                    if done:
                        for i, packer in enumerate(packer for packer in self.packers if packer.space_utilization() != 0):
                            self.used_packers.append(packer)
                        self.used_bins -= len([packer for packer in self.packers if packer.space_utilization() == 0])
                    
            return next_state, reward, done




    def p_map(self, i_bin, cuboid):
        x, y, z, w, h, d = cuboid
        
        W, H, D = self.packers[i_bin].size
        mask = np.zeros((D, W))
        mask[z:z + d, x:x + w] = h / H
        
        return mask
    
    def i_map(self, i_bin, items):
        W, H, D = self.packers[i_bin].size
        
        masks = np.zeros((self.k, 3))
        for i, item in enumerate(items):
            if item is None:
                continue
            w, h, d = item
            masks[i] = (w / W, h / H, d / D)

        return masks

