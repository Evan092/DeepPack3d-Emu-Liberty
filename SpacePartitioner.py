import numpy as np
import matplotlib.pyplot as plt

from geometry import Cuboid

class Renderer:
    def __init__(self, figsize=(14, 12)):
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(projection='3d')
        plt.close()
        
    def clear(self):
        self.ax.clear()
        
    def draw(self, box, color=None, mode='fill', linewidth=1):
        if color is None:
            color = (*np.random.random((3,)) * 0.7 + 0.3, 0.6)
            
        ax = self.ax
        x, y, z = box.left, box.back, box.bottom
        dx, dy, dz = box.width, box.depth, box.height
        if mode == 'fill':
            xx = np.linspace(x, x + dx, 2)
            yy = np.linspace(y, y + dy, 2)
            zz = np.linspace(z, z + dz, 2)

            xx, yy = np.meshgrid(xx, yy)

            l1, l2 = xx.shape
            z0 = np.ones([l1, l2]) * z
            
            surf_kwargs = {'color': color, 'edgecolor': (0, 0, 0, 0.4), 'linewidth': 0.5}
            
            ax.plot_surface(xx, yy, z0, **surf_kwargs)
            ax.plot_surface(xx, yy, z0 + dz, **surf_kwargs)

            yy, zz = np.meshgrid(yy, zz)
            ax.plot_surface(x, yy, zz, **surf_kwargs)
            ax.plot_surface(x + dx, yy, zz, **surf_kwargs)

            xx, zz = np.meshgrid(xx, zz)
            ax.plot_surface(xx, y, zz, **surf_kwargs)
            ax.plot_surface(xx, y + dy, zz, **surf_kwargs)
        elif mode == 'stroke':
            xx = [x, x, x + dx, x + dx, x]
            yy = [y, y + dy, y + dy, y, y]
            kwargs = {'alpha': 1, 'color': color, 'linewidth': linewidth}
            ax.plot3D(xx, yy, [z] * 5, **kwargs)
            ax.plot3D(xx, yy, [z + dz] * 5, **kwargs)
            ax.plot3D([x, x], [y, y], [z, z + dz], **kwargs)
            ax.plot3D([x, x], [y + dy, y + dy], [z, z + dz], **kwargs)
            ax.plot3D([x + dx, x + dx], [y + dy, y + dy], [z, z + dz], **kwargs)
            ax.plot3D([x + dx, x + dx], [y, y], [z, z + dz], **kwargs)
            
        return color
    
    def show(self):
        return (self.fig)
        
def render(size, spaces, colors, conveyor_items=None, conveyor_colors=None):
    r = Renderer(figsize=(14, 12))

    # Painter's algorithm: draw boxes furthest from the default camera first.
    # Default matplotlib 3D view is roughly from front-right-above (azim≈-60, elev≈30).
    # Boxes with larger z (deeper) and larger x (right) are further; draw them first
    # so closer boxes paint over them correctly.
    sorted_spaces = sorted(spaces, key=lambda b: -(b.z + b.x))
    for box in sorted_spaces:
        colors[box] = r.draw(box, color=colors[box] if box in colors else None)

    r.draw(Cuboid(0, 0, 0, *size), color='red', mode='stroke', linewidth=2.5)

    max_x = size[0]
    max_height = size[1]  
    max_depth = size[2]   

    if conveyor_items:
        W = size[0]
        gap = 10
        x_offset = W + gap
        
        current_y = 0
        max_conv_width = 0
        max_conv_depth = 0
        
        for idx, item in enumerate(conveyor_items):
            if item is None:
                continue
            w_i, h_i, d_i = item
            
            conv_box = Cuboid(x_offset, current_y, size[2]-d_i, w_i, h_i, d_i)
            
            # Map the color to the index, default to green if missing
            c = conveyor_colors[idx] if conveyor_colors and idx < len(conveyor_colors) else (0.2, 0.8, 0.2, 0.5)
            r.draw(conv_box, color=c)
            
            current_y += h_i + 2
            
            if w_i > max_conv_width: max_conv_width = w_i
            if d_i > max_conv_depth: max_conv_depth = d_i
            
        max_x = x_offset + max_conv_width
        max_height = max(max_height, current_y)
        max_depth = max(max_depth, max_conv_depth)

    r.ax.set_box_aspect((max_x, max_depth, max_height))
    r.ax.set_xlim(0, max_x)
    r.ax.set_ylim(0, max_depth)
    r.ax.set_zlim(0, max_height)
    
    r.ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    r.ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    r.ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    r.ax.tick_params(axis='x', labelsize=12)
    r.ax.tick_params(axis='y', labelsize=12)
    r.ax.tick_params(axis='z', labelsize=12)
    
    r.fig.tight_layout(pad=1)

    return r.show()
    
class SpacePartitioner:
    def __init__(self, size, floor_height=0, v_pad_headroom=0):
        self.size = size 
        self.floor_height = floor_height
        w, h, d = size
        self.internal_size = (w, h + v_pad_headroom, d)
        self.reset()
        self._colors = {}
        
    def reset(self):
        w, h, d = self.internal_size 
        self.free_splits = [Cuboid(0, 0, 0, w, h, d)]
        self.splits = []
        self.actual_splits = []
        rw, _, rd = self.size 
        self.height_map = np.full((rd, rw), self.floor_height, dtype=float)
        self.actual_height_map = np.zeros((rd, rw), dtype=float)
        
    def fit(self, cuboid):
        outer = Cuboid(0, 0, 0, *self.internal_size)
        
        if not outer.contain(cuboid):
            return False
        
        if len(self.splits) < len(self.free_splits):
            for split in self.splits:
                if split.intersect(cuboid):
                    return False
            return True
        
        for split in self.free_splits:
            if split.contain(cuboid):
                return True
        return False
    
    def add(self, cuboid, actual_cuboid=None, color=None):
        if not self.fit(cuboid):
            return False
        
        if actual_cuboid is None:
            actual_cuboid = cuboid
            
        self.splits.append(cuboid)
        self.actual_splits.append(actual_cuboid)
        
        # Lock in the provided color
        if color is not None:
            self._colors[actual_cuboid] = color
        
        (left, bottom, back), (right, top, front) = cuboid.bounding_box()
        W, H, D = self.size
        clamp_right = int(min(right, W))
        clamp_front = int(min(front, D))
        left, back = int(left), int(back)
        cover = np.maximum(self.height_map[back:clamp_front, left:clamp_right], top)
        self.height_map[back:clamp_front, left:clamp_right] = cover
        
        (a_left, a_bottom, a_back), (a_right, a_top, a_front) = actual_cuboid.bounding_box()
        a_left, a_back = int(a_left), int(a_back)
        a_clamp_right = int(min(a_right, W))
        a_clamp_front = int(min(a_front, D))
        actual_y = np.max(self.actual_height_map[a_back:a_clamp_front, a_left:a_clamp_right])
        item_height = a_top - a_bottom
        actual_top = actual_y + item_height
        self.actual_height_map[a_back:a_clamp_front, a_left:a_clamp_right] = np.maximum(
            self.actual_height_map[a_back:a_clamp_front, a_left:a_clamp_right], actual_top)
        
        partitions = []
        new_partitions = []
        for partition in self.free_splits:
            if partition.intersect(cuboid):
                new_partitions.extend(partition.split(cuboid))
            else:
                partitions.append(partition)
                
        n_partitions = len(partitions)
        for i in range(len(new_partitions)):
            contained = False
            
            partition = new_partitions[i]
            
            for j in range(len(new_partitions)):
                if i != j and new_partitions[j].contain(partition):
                    contained = True
                    break
                    
            if not contained:
                for j in range(n_partitions):
                    if partitions[j].contain(partition):
                        contained = True
                        break
            
            if not contained:
                partitions.append(partition)
                
        self.free_splits = partitions
        
        return True
    
    def space_utilization(self):
        actual_used = np.sum([split.volume for split in self.actual_splits])
        return actual_used / np.prod(self.size)
        
    def render(self, free=False, conveyor_items=None, conveyor_colors=None):
        if free:
            splits = self.free_splits
        else:
            splits = self.actual_splits
        return render(self.size, splits, self._colors, conveyor_items=conveyor_items, conveyor_colors=conveyor_colors)