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

class MultiBinPackerEnv(Env):
    def __init__(self, n_bins, size, k=1, max_bins=None, max_items=None, replace='all', verbose=False, prealloc_bins=0, prealloc_items=0, shuffle=False, use_rotate=True, use_skip=True, h_pad=0.5, v_pad=0.5):
        super().__init__(verbose)
        self.n_bins = n_bins
        self.size = size
        self.k = k
        
        # Padding configuration
        # h_pad: horizontal padding (x, z) between boxes. realistic for real-world replication
        # v_pad: vertical padding (y) between stacked boxes and below first layer. minimal for physics sim
        # v_pad does NOT count against bin height: items "fall" into place in the real world
        self.h_pad = h_pad
        self.v_pad = v_pad
        
        # Scale factor for converting grid coords back to real-world units
        # Default 1.0, overridden when loading from file
        self.inv_scale = 1.0
        
        W, H, D = size
        self.v_pad_headroom = (H + 1) * v_pad
        
        self.packers = [SpacePartitioner(size, floor_height=v_pad, v_pad_headroom=self.v_pad_headroom) for _ in range(n_bins)]
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
        
    def reset(self):
        for packer in self.packers:
            packer.reset()
        self.conveyor.reset()
        
        self._state = None
        self.used_bins = self.n_bins
        self.used_packers = []
        self.items_in_current_bin = 0
        self._item_colors = []
        return self.state()
    
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
#             print(items)
#             print(self.actions(items, h_maps, rotate=True, skip=True))
            actions = list(self.actions(items, h_maps, rotate=self.use_rotate, skip=self.use_skip))
    
            self._state = (items, h_maps, u_maps, actions)
        
        return self._state
    
    def _height_maps(self):
        return [packer.height_map.copy() for packer in self.packers]
    
    def _actual_height_maps(self):
        return [packer.actual_height_map.copy() for packer in self.packers]
    
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
            
            # Using a tuple instead of a list avoids memory allocation overhead per loop
            split_candidates = (
                (sx, sz),                                      
                (sx + sw - w_pad, sz),                         
                (sx, sz + sd - d_pad),                         
                (sx + sw - w_pad, sz + sd - d_pad)             
            )
            
            for cx, cz in split_candidates:
                # math.ceil is significantly faster on scalars than np.ceil
                x, z = int(math.ceil(cx)), int(math.ceil(cz))
                
                if 0 <= x and 0 <= z and x + w_pad <= self.size[0] and z + d_pad <= self.size[2]:
                    if (x, z) not in splits:
                        splits[(x, z)] = split
        
        for placed in packer.actual_splits:
            for cx in (int(placed.x), int(placed.right)):
                for cz in (int(placed.z), int(placed.front)):
                    if cx + w_pad <= self.size[0] and cz + d_pad <= self.size[2]:
                        if (cx, cz) not in splits:
                            splits[(cx, cz)] = None

        xyz = []
        
        # Iterate directly over the deduplicated dictionary items
        for (x, z), split in splits.items():
            
            # 1. CHEAPEST CHECK: Does the unpadded height fit?
            actual_placement = packer.actual_height_map[z:z + d, x:x + w]
            actual_y = np.amax(actual_placement)
            if actual_y + h > self.size[1]:
                continue
                
            # 2. CHEAP CHECK: Does it meet the 50% stability threshold?
            placement = h_map[z:z + d, x:x + w]
            y = np.amax(placement)
            if np.count_nonzero(placement == y) / base_area <= 0.5:
                continue

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
        items, h_maps, u_maps, actions = self.state()
        
        # item, bin, rotation_placement
        i, j, k = action
        _, (x, y, z), (w, h, d), _ = actions[i][j][k]
        
        # Capture height map BEFORE placing the item for overhang calculation
        old_h_map = h_maps[j].copy()
        
        packer = self.packers[j]
        actual_cuboid = Cuboid(x, y, z, w, h, d)
        padded_cuboid = Cuboid(x, y, z, w + self.h_pad, h + self.v_pad, d + self.h_pad)
        item_color = self._item_colors.pop(i) if i < len(self._item_colors) else None
        if not packer.add(padded_cuboid, actual_cuboid=actual_cuboid, color=item_color):
            raise Exception(f'invalid space {actual_cuboid}')
        self.conveyor.grab(i)
        
        # Increment items placed in current bin
        self.items_in_current_bin += 1
        
        
        next_state = self.state(step=True)
        
        # reward shaping
        items, h_maps, u_maps, actions = next_state
        
        item = items[i]
        h_map = h_maps[j]
        
        volume = np.sum([split.volume for split in packer.actual_splits])

        # === REWARD COMPONENTS ===
        
        W, H, D = packer.size
        actual_h_map = packer.actual_height_map
        max_h = np.max(actual_h_map)

        # 1. Surface Variance: reward a flat/uniform top surface
        #    std = 0 means perfectly flat, so score = 1.0
        #    But flatness is only meaningful if the bin has good coverage.
        #    A tiny corner tower with std=0 should NOT score high.
        if max_h == 0:
            surface_flatness = 0.0  # empty bin is not "flat", it's empty
        else:
            # Measure flatness over the FULL height map (zeros included).
            # This penalizes sparse placement (a tall tower in one)
            # corner surrounded by zeros has HIGH std.
            h_map_std = np.std(actual_h_map)
            max_possible_std = H / 2.0
            raw_flatness = 1.0 - (h_map_std / max_possible_std)
            raw_flatness = max(raw_flatness, 0.0)
            
            # Coverage: fraction of bin footprint that is occupied
            filled_cells = np.sum(actual_h_map > self.v_pad)
            total_cells = actual_h_map.size
            coverage = filled_cells / total_cells
            
            # Flatness proportional to how much of the bin you've filled.
            # sqrt(coverage) so early placements aren't punished too harshly.
            surface_flatness = raw_flatness * np.sqrt(coverage)
            
            # Stackability: flatness of ONLY the filled cells
            # This directly measures whether items can be stacked on top
            # e.g. heights [7,7,8,9,10] = high std = low stackability
            #      heights [7,7,7,7,7] = zero std = perfect stackability
            filled_mask = actual_h_map > self.v_pad
            if np.any(filled_mask):
                filled_std = np.std(actual_h_map[filled_mask])
                # Max realistic std among filled cells (heights range 6-12 units)
                stackability = 1.0 - min(filled_std / (H / 4.0), 1.0)
            else:
                stackability = 1.0

        # 2. Center Proximity (Inside-Out): reward placing items near center first
        #    Uses Euclidean distance to center for a stronger, cleaner center-pull signal
        if max_h == 0:
            center_proximity = 0.0
        else:
            xs, zs = np.meshgrid(np.arange(W), np.arange(D))
            
            # Euclidean distance from center of bin
            center_x, center_z = (W - 1) / 2.0, (D - 1) / 2.0
            dist_to_center = np.sqrt((xs - center_x)**2 + (zs - center_z)**2)
            max_dist = np.sqrt(center_x**2 + center_z**2)
            normalized_dist = dist_to_center / (max_dist + 1e-9)
            
            # Weighted average: cells with more height have more influence
            # Score is 1.0 when mass is at center, 0.0 when at corners
            total_height = np.sum(actual_h_map)
            if total_height > 0:
                center_proximity = 1.0 - np.sum(normalized_dist * actual_h_map) / total_height
            else:
                center_proximity = 0.0

        # 3. Volume Efficiency: packed volume / bounding box volume
        #    Direct signal for dense packing
        if max_h == 0:
            volume_efficiency = 0.0
        else:
            bounding_box_volume = W * max_h * D
            volume_efficiency = volume / bounding_box_volume

        # 4. Gap Fill Metric: reward placing items into valleys/holes
        #    Compare placement height y to neighborhood average. lower y relative to neighbors = filling a gap
        neighborhood_radius = max(w, d)  # look at area around the placed item
        z_start = max(0, z - neighborhood_radius)
        z_end = min(D, z + d + neighborhood_radius)
        x_start = max(0, x - neighborhood_radius)
        x_end = min(W, x + w + neighborhood_radius)
        
        neighborhood = old_h_map[z_start:z_end, x_start:x_end]
        
        if neighborhood.size > 0 and np.max(neighborhood) > 0:
            neighborhood_mean = np.mean(neighborhood)
            # How far below the neighborhood average was the placement?
            # Positive = placed into a valley, Negative = placed on a peak
            gap_score = (neighborhood_mean - y) / (np.max(old_h_map) + 1e-9)
            # Clamp to [0, 1]: only reward filling gaps, don't penalize placing on top
            gap_fill = max(0.0, min(1.0, gap_score + 0.5))  # shift so neutral = 0.5
        else:
            gap_fill = 0.5  # neutral for first placement

        # 4b. Gap Creation Penalty: penalize placements that CREATE new gaps/valleys
        #     Compare the height map around the placed item BEFORE vs AFTER placement.
        #     If new deep valleys appeared (cells much lower than the new item top),
        #     the agent is building pillars/walls that create hard-to-fill voids.
        new_h_map = h_map  # h_map is already the post-placement height map
        
        # Measure gaps in the neighborhood BEFORE and AFTER placement
        # A "gap" is a cell significantly below its local maximum
        def measure_gap_severity(hmap_region):
            """Sum of (local_max - cell_height) for all cells, normalized."""
            if hmap_region.size == 0 or np.max(hmap_region) == 0:
                return 0.0
            local_max = np.max(hmap_region)
            # Total "air" between the surface and the highest point
            return np.sum(local_max - hmap_region) / (hmap_region.size * local_max + 1e-9)
        
        # Use a wider neighborhood to catch gaps created beside the placed item
        gap_radius = max(w, d) + 1
        gz_start = max(0, z - gap_radius)
        gz_end = min(D, z + d + gap_radius)
        gx_start = max(0, x - gap_radius)
        gx_end = min(W, x + w + gap_radius)
        
        old_gap_severity = measure_gap_severity(old_h_map[gz_start:gz_end, gx_start:gx_end])
        new_gap_severity = measure_gap_severity(new_h_map[gz_start:gz_end, gx_start:gx_end])
        
        # Positive = we made gaps worse, Negative = we reduced gaps
        gap_delta = new_gap_severity - old_gap_severity
        
        # gap_creation_penalty: 0 when no new gaps created, negative when gaps worsen
        # This is a PENALTY (subtracted from reward), so worse gaps = lower reward
        # Clamp: only penalize gap creation, don't double-reward gap filling (that's gap_fill's job)
        gap_creation_penalty = max(0.0, gap_delta)  # 0 to ~1 range

        # 5. Same-Height Adjacency: reward placing items next to items of the same height.
        #    Uses old_h_map so the placed item's own cells are not counted.
        #    Denominator = ALL cells in the neighborhood window (not just occupied),
        #    so isolated placements score near 0 and can't trivially saturate to 1.0.
        x_int, z_int = int(x), int(z)
        w_int, d_int = int(w), int(d)

        # actual_placed_top: the true top of the item we just placed
        actual_placed_top = np.max(packer.actual_height_map[z_int:z_int + d_int, x_int:x_int + w_int])

        # Neighborhood window (2 cells beyond the item's footprint on each side)
        search_radius = 2
        n_x_start = max(0, x_int - search_radius)
        n_x_end = min(W, x_int + w_int + search_radius)
        n_z_start = max(0, z_int - search_radius)
        n_z_end = min(D, z_int + d_int + search_radius)

        neighborhood = old_h_map[n_z_start:n_z_end, n_x_start:n_x_end]

        # Build mask that excludes the item's own footprint cells
        exclude_self = np.ones(neighborhood.shape, dtype=bool)
        rel_z = z_int - n_z_start
        rel_x = x_int - n_x_start
        exclude_self[rel_z:rel_z + d_int, rel_x:rel_x + w_int] = False

        # Same-height cells outside the item's own footprint
        same_height = (np.abs(neighborhood - actual_placed_top) <= 0.5) & (neighborhood > self.v_pad) & exclude_self
        total_neighbor_cells = int(np.sum(exclude_self))  # all non-self cells in window

        if total_neighbor_cells > 0:
            adjacency = float(np.sum(same_height)) / total_neighbor_cells
        else:
            adjacency = 0.0

        # 6. Support IoU: asymmetric support quality metric
        #    STRONGLY penalizes overhang (item bigger than support, or misaligned)
        #    Gently treats small-on-big (item fully supported by larger surface)
        #    Rewards completing coverage of a support platform (4 small boxes on 1 big)
        if y <= self.v_pad:
            # Ground level: floor is perfect support
            support_iou = 1.0
        else:
            footprint = old_h_map[z:z + d, x:x + w]
            total_cells = w * d
            if total_cells > 0:
                # Support fraction: what % of this item's footprint is supported?
                supported_cells = np.sum(np.isclose(footprint, y, atol=0.5))
                support_fraction = supported_cells / total_cells
                
                # Quadratic penalty for overhang: drops off sharply below 100%
                # 1.0=1.0, 0.9=0.81, 0.7=0.49, 0.5=0.25, 0.3=0.09
                # This STRONGLY pushes away from bigger-than-below and or hanging off edge
                overhang_score = support_fraction ** 2
                
                # Surface utilization: how well does this box (+ siblings) cover
                #     the support platform below?
                # Find the support platform: cells at height y in a small neighborhood
                # around the placement. This captures the big box surface we're building on.
                expand = 1
                sx_start = max(0, int(x) - expand)
                sx_end = min(W, int(x + w) + expand)
                sz_start = max(0, int(z) - expand)
                sz_end = min(D, int(z + d) + expand)
                support_region_old = old_h_map[sz_start:sz_end, sx_start:sx_end]
                platform_mask = np.isclose(support_region_old, y, atol=0.5)
                platform_cells = np.sum(platform_mask)
                
                if platform_cells > 0:
                    # How many platform cells now have something on top (after placement)?
                    support_region_new = h_map[sz_start:sz_end, sx_start:sx_end]
                    covered = np.sum((support_region_new > y + 0.1) & platform_mask)
                    surface_utilization = covered / platform_cells
                else:
                    surface_utilization = 0.0
                
                # Blend: overhang avoidance is dominant (70%), surface completion bonus (30%)
                # Same-size stacking: overhang=1.0, utilization=1.0 = 1.0  (ideal)
                # Small-on-big:       overhang=1.0, utilization=0.3 = 0.79 (decent)
                # 4 small completing:  overhang=1.0, utilization=1.0 = 1.0  (ideal)
                # Big-on-small:        overhang=0.25, utilization=1.0 = 0.475 (bad)
                support_iou = 0.7 * overhang_score + 0.3 * surface_utilization
                
                # Gentle height decay: well-supported items keep most credit
                height_decay = 1.0 - 0.5 * (y / (H + 1e-9))
                support_iou *= max(height_decay, 0.4)
            else:
                support_iou = 0.0

        # === Footprint Coverage ===
        # Directly reward using more of the bin's floor area.
        # This is the strongest anti-corner-stacking signal.
        # Exponential scaling: small coverage is severely punished, high coverage rewarded.
        filled_cells = np.sum(actual_h_map > self.v_pad)
        total_cells = actual_h_map.size
        linear_coverage = filled_cells / total_cells
        # Exponential: coverage^0.5 would be generous; coverage^2 is harsh.
        # Use coverage^1.5 so low coverage (0.1 = 0.03) is punished, high (0.8 = 0.72) rewarded.
        footprint_coverage = linear_coverage ** 1.5

        # === Big Bottom ===
        # Reward placing large items low, small items high.
        # Encourages a stable base. heavy/large boxes on the floor.
        item_volume = w * h * d
        # Use the largest item currently visible on the conveyor as reference,
        # not the entire bin volume (which makes every item's fraction ≈ 0).
        conveyor_volumes = []
        for itm in items:
            if itm is not None:
                iw, ih, id_ = itm
                conveyor_volumes.append(iw * ih * id_)
        # Fallback: use the placed item's own volume (score = 1.0 * height_factor)
        max_visible_volume = max(conveyor_volumes) if conveyor_volumes else item_volume
        # Use at least the current item volume to avoid division issues
        max_visible_volume = max(max_visible_volume, item_volume)
        volume_fraction = item_volume / (max_visible_volume + 1e-9)
        # height_fraction: 0 on floor, 1 at top of bin
        height_fraction = y / (H + 1e-9)
        # Score: large items placed low = high score; small items anywhere = modest score
        big_bottom = (1.0 - height_fraction) * volume_fraction

        # === Layer Completion ===
        # Reward building in uniform layers rather than towers.
        # Uses mean/max of FILLED cells. activates from item 2 onward.
        # A tower at y=80 next to items at y=10: mean=20, max=80 = score=0.25
        # All items at same height: mean=10, max=10 = score=1.0
        filled_mask = actual_h_map > self.v_pad
        if np.any(filled_mask):
            filled_heights = actual_h_map[filled_mask]
            max_filled = np.max(filled_heights)
            if max_filled > 0:
                layer_completion = np.mean(filled_heights) / (max_filled + 1e-9)
            else:
                layer_completion = 0.0
        else:
            layer_completion = 0.0  # empty bin has no layers

        # Height Penalty: quadratic so low stacking is nearly free but towers
        # are severely penalized. (y/H)^2: y=10->0.01, y=40->0.18, y=60->0.41, y=80->0.73
        height_penalty = (y / (H + 1e-9)) ** 2

        # === DYNAMIC SCHEDULER ===
        # Early: surface flatness + wall proximity (stable perimeter base)
        # Late: volume efficiency + gap filling (maximize items packed)
        progress = min(self.items_in_current_bin / 50.0, 1.0)
        
        # Surface flatness: starts high, decays
        flatnessScale = (1.5 * (1.0 - progress) + 0.5 * progress) * 2
        
        # Stackability: stays important throughout. can't stack on uneven surfaces ever
        stackScale = 1
        
        # Center proximity: starts high, decays
        centerScale = 1.5 * (1.0 - progress) + 0.3 * progress
        
        # Volume efficiency: starts low, grows
        volumeScale = 0.3 * (1.0 - progress) + 1.5 * progress
        
        # Gap fill: starts low, grows
        gapScale = 0.7 * (1.0 - progress) + 1.5 * progress

        # Gap creation penalty: important throughout. never want to create voids
        gapCreationScale = 2.0 * (1.0 - progress) + 3.0 * progress

        # Adjacency: important throughout. always want uniform-height neighbors
        adjacencyScale = 2

        # Support IoU: grows with progress. stacking quality matters more as bin fills
        supportScale = 0.5 * (1.0 - progress) + 2.0 * progress

        # Footprint coverage: boosted aggressively to prevent corner stacking
        coverageScale = 6.0 * (1.0 - progress) + 3.0 * progress

        # Big bottom: strongest early when base is being built, tapers off late
        bigBottomScale = 3.0 * (1.0 - progress) + 0.5 * progress

        # Layer completion: critical throughout. prevents towers
        layerScale = 3.0 * (1.0 - progress) + 2.0 * progress

        # Height penalty: starts at 3x scale (quadratic, so early low stacking tiny),
        # grows to 12x late. at y=60, H=94: (0.638)^2 * 12 = -4.9 vs floor -> ~0
        heightPenaltyScale = 3.0 * (1.0 - progress) + 12.0 * progress

        # === BASELINE ===
        # Subtract a baseline so that bad placements get negative reward but
        # decent placements get positive reward. Critical: if ALL rewards are
        # negative, the agent learns to end the episode FAST (towers exhaust
        # valid placements quickly) rather than packing efficiently.
        # 0.4 = nearly all rewards negative (observed: agent builds towers to die fast)
        # 0.2 = median placement ≈ 0, good packing = positive, towers = negative
        positive_scale_sum = (flatnessScale + stackScale + centerScale + volumeScale
                            + gapScale + adjacencyScale + supportScale
                            + coverageScale + bigBottomScale + layerScale)
        baseline = 0.2 * positive_scale_sum

        reward = (flatnessScale * surface_flatness
                + stackScale * stackability
                + centerScale * center_proximity 
                + volumeScale * volume_efficiency 
                + gapScale * gap_fill
                - gapCreationScale * gap_creation_penalty
                + adjacencyScale * adjacency
                + supportScale * support_iou
                + coverageScale * footprint_coverage
                + bigBottomScale * big_bottom
                + layerScale * layer_completion
                - heightPenaltyScale * height_penalty
                - baseline)

        # Store per-component breakdown for visualization / debugging
        self._reward_breakdown = {
            'flatness': round(float(flatnessScale * surface_flatness), 4),
            'stackability': round(float(stackScale * stackability), 4),
            'center': round(float(centerScale * center_proximity), 4),
            'volume_eff': round(float(volumeScale * volume_efficiency), 4),
            'gap_fill': round(float(gapScale * gap_fill), 4),
            'gap_penalty': round(float(-gapCreationScale * gap_creation_penalty), 4),
            'adjacency': round(float(adjacencyScale * adjacency), 4),
            'support_iou': round(float(supportScale * support_iou), 4),
            'coverage': round(float(coverageScale * footprint_coverage), 4),
            'big_bottom': round(float(bigBottomScale * big_bottom), 4),
            'layer_completion': round(float(layerScale * layer_completion), 4),
            'height_penalty': round(float(-heightPenaltyScale * height_penalty), 4),
            'baseline': round(float(-baseline), 4),
            'overhang_penalty': 0.0,  # updated below if applicable
        }

        # Overhang penalty: ADDITIVE penalty subtracted from reward.
        # Previously multiplicative (could never make reward negative); now directly subtracts.
        if y > self.v_pad:  # Only apply if item is not on ground level (above floor padding)
            # Calculate support: area where item sits on solid support (height == y before placement)
            support_mask = (old_h_map[z:z + d, x:x + w] == y).astype(float)
            support_area = np.sum(support_mask)
            total_item_area = w * d
            
            # Overhang ratio: how much of the item doesn't have proper support
            if total_item_area > 0:
                overhang_ratio = 1.0 - (support_area / total_item_area)
                
                # Calculate support symmetry across quadrants of the item's footprint
                mid_x = w / 2.0
                mid_z = d / 2.0
                
                # Sum of support in each quadrant (using floor/ceil to handle odd dimensions)
                q1 = np.sum(support_mask[:int(np.ceil(mid_z)), :int(np.ceil(mid_x))])   # top-left
                q2 = np.sum(support_mask[:int(np.ceil(mid_z)), int(np.floor(mid_x)):])   # top-right
                q3 = np.sum(support_mask[int(np.floor(mid_z)):, :int(np.ceil(mid_x))])   # bottom-left
                q4 = np.sum(support_mask[int(np.floor(mid_z)):, int(np.floor(mid_x)):])  # bottom-right
                
                quadrants = np.array([q1, q2, q3, q4])
                
                if support_area > 0:
                    quadrant_fracs = quadrants / support_area
                    max_std = np.std([1.0, 0.0, 0.0, 0.0])  # ~0.433
                    asymmetry = np.std(quadrant_fracs) / max_std
                else:
                    asymmetry = 1.0
                
                # Sigmoid-shaped base penalty from overhang ratio (0..1 range)
                sig = (1 / (1 + np.exp((-18 * overhang_ratio) + 5.5)))
                base_penalty = sig  # 0 when well-supported, ~1 when floating
                
                # Asymmetry amplifier: lopsided support is extra bad
                asymmetry_amp = 1.0 + 0.5 * asymmetry * overhang_ratio
                
                # Foundation: hollow columns underneath make it worse
                u_maps = self._utilization_maps()
                support_utilization = np.mean(u_maps[j][z:z + d, x:x + w])
                foundation_amp = 1.0 + 0.3 * (1.0 - support_utilization)
                
                # Final additive penalty (0 = no overhang, up to ~6+ for severe overhang)
                overhang_scale = 3.0 * (1.0 - progress) + 6.0 * progress
                overhang_additive = overhang_scale * base_penalty * asymmetry_amp * foundation_amp
                reward -= overhang_additive
                self._reward_breakdown['overhang_penalty'] = round(float(-overhang_additive), 4)
                
                if self.verbose:
                    print(f'Overhang ratio: {overhang_ratio:.2%}, Support asymmetry: {asymmetry:.2%}, Additive penalty: {overhang_additive:.3f}')
                    print(f'Progress: {progress:.2%}, Flatness: {flatnessScale:.3f}, Center: {centerScale:.3f}, Volume: {volumeScale:.3f}, Gap: {gapScale:.3f}, Adjacency: {adjacencyScale:.3f}, SupportIoU: {supportScale:.3f}')
                    print(f'Adjacency score: {adjacency:.3f}, Support IoU: {support_iou:.3f}')
        
        done = len(indices(actions)) == 0
        
        if done:
            # Terminal reward: bonus/penalty based on final bin utilization
            # High utilization = bonus, low utilization = penalty
            # This gives the agent a strong end-of-episode signal to maximize packing
            total_volume = W * H * D
            packing_ratio = volume / total_volume
            # Bonus scales from -1.0 (empty) to +1.0 (full)
            # e.g. 80% full = +0.6, 50% full = 0.0, 20% full = -0.6
            reward += 10.0 * (packing_ratio - 0.5)
            
            if self.max_bins != -1 and self.used_bins + 1 > self.max_bins:
                for i, packer in enumerate(packer for packer in self.packers if packer.space_utilization() != 0):
                    self.used_packers.append(packer)
                    loc = self.packers.index(packer)
                    if self.verbose:
                        print(f'bin {self.used_bins - self.n_bins + i}, loc: {loc}, space util: {packer.space_utilization() * 100:.2f}, packed items: {len(packer.actual_splits)}')
                done = True
            else:
                if self.replace == 'max':
                    loc = np.argmax([packer.space_utilization() for packer in self.packers])
                    packer = self.packers[loc]
                    if self.verbose:
                        print(f'bin {self.used_bins - self.n_bins}, loc: {loc}, space util: {packer.space_utilization() * 100:.2f}, packed items: {len(packer.actual_splits)}')

                    self.used_packers.append(self.packers[loc])
                    self.packers[loc] = SpacePartitioner(self.size, floor_height=self.v_pad, v_pad_headroom=self.v_pad_headroom)
                    self.packers[loc].reset()
                    self.items_in_current_bin = 0  # Reset counter for new bin
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
                        if self.verbose:
                            print(f'bin {self.used_bins - self.n_bins}, loc: {loc}, space util: {packer.space_utilization() * 100:.2f}, packed items: {len(packer.actual_splits)}')

                        self.used_packers.append(self.packers[loc])
                        self.packers[loc] = SpacePartitioner(self.size, floor_height=self.v_pad, v_pad_headroom=self.v_pad_headroom)
                        self.packers[loc].reset()
                        self.items_in_current_bin = 0  # Reset counter for new bin
                        added += 1
                        self.used_bins += 1
                else:
                    raise Exception('not implemented')
                
                next_state = self.state(step=True)
                items, h_maps, u_maps, actions = next_state
                done = len(indices(actions)) == 0
                if done:
                    for i, packer in enumerate(packer for packer in self.packers if packer.space_utilization() != 0):
                        self.used_packers.append(packer)
                        loc = self.packers.index(packer)
                        if self.verbose:
                            print(f'bin {self.used_bins - self.n_bins + i + 1}, loc: {loc}, space util: {packer.space_utilization() * 100:.2f}, packed items: {len(packer.actual_splits)}')
                    self.used_bins -= len([packer for packer in self.packers if packer.space_utilization() == 0])
#                 if not done:
#                     self.used_bins += added
#                     pass
            if self.verbose: print()
                
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

