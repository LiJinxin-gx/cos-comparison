#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cos_comparison Multi‑Canvas Test Platform (Enhanced)
=====================================================

Features:
- Draw on canvas with pixel buffer (supports merging with loaded images)
- Adjustable canvas size and crop offset for training
- Synthesize drawn/imported content as training tensor
- Run tests on synthesized or loaded images
- Enhanced syntax highlighting in custom code editor
- Interactive Python shell for experimentation
- Compact left-panel layout
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser, scrolledtext, simpledialog
import time
import os
import sys
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr

# ---------- Import cos_comparison core ----------
try:
    from cos_comparison import core
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    core = None


# ============================================================
#  Utility: Heatmap conversion (pure tkinter)
# ============================================================
def jet_colormap(t):
    t = max(0.0, min(1.0, t))
    r = g = b = 0.0
    if t <= 0.125:
        r, g, b = 0.0, 0.0, 0.5 + 4.0 * t
    elif t <= 0.375:
        r, g, b = 0.0, 4.0 * (t - 0.125), 1.0
    elif t <= 0.625:
        r, g, b = 4.0 * (t - 0.375), 1.0, 1.0 - 4.0 * (t - 0.375)
    elif t <= 0.875:
        r, g, b = 1.0, 1.0 - 4.0 * (t - 0.625), 0.0
    else:
        r, g, b = 1.0 - 4.0 * (t - 0.875), 0.0, 0.0
    return (int(r * 255), int(g * 255), int(b * 255))


def array_to_photoimage(data_2d):
    if not data_2d or not data_2d[0]:
        return None
    h, w = len(data_2d), len(data_2d[0])
    vals = []
    for row in data_2d:
        for v in row:
            if v is not None and not (v != v):
                vals.append(v)
    if not vals:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = min(vals), max(vals)
        if vmax - vmin < 1e-9:
            vmax = vmin + 1.0

    img = tk.PhotoImage(width=w, height=h)
    rows = []
    for y in range(h):
        row_pixels = []
        for x in range(w):
            val = data_2d[y][x]
            t = 0.0
            if val is not None:
                try:
                    fval = float(val)
                    if not (fval != fval):
                        t = (fval - vmin) / (vmax - vmin)
                        t = max(0.0, min(1.0, t))
                except:
                    pass
            r, g, b = jet_colormap(t)
            row_pixels.append(f'#{r:02x}{g:02x}{b:02x}')
        rows.append('{' + ' '.join(row_pixels) + '}')
    img.put(' '.join(rows))
    return img


# ============================================================
#  Test Module Base and Built‑ins (with display_result)
# ============================================================
class TestModule:
    name = "Base Test"
    description = "Abstract test module"

    def get_param_widgets(self, parent):
        frame = ttk.Frame(parent)
        ttk.Label(frame, text="No parameters.").pack()
        return frame

    def run(self, image_data, **params):
        raise NotImplementedError

    def display_result(self, result, canvas, status_widget):
        raise NotImplementedError


class PassiveTest(TestModule):
    name = "Passive Mode"
    description = "Fixed-offset sliding window comparison."

    def get_param_widgets(self, parent):
        frame = ttk.Frame(parent)
        row = 0
        ttk.Label(frame, text="Window Size:").grid(row=row, column=0, sticky=tk.W, padx=2, pady=2)
        self.ws_var = tk.StringVar(value="3,3,1")
        ttk.Entry(frame, textvariable=self.ws_var, width=10).grid(row=row, column=1, padx=2, pady=2)
        row += 1

        ttk.Label(frame, text="Offset:").grid(row=row, column=0, sticky=tk.W, padx=2, pady=2)
        self.off_var = tk.StringVar(value="1,0,0")
        ttk.Entry(frame, textvariable=self.off_var, width=10).grid(row=row, column=1, padx=2, pady=2)
        row += 1

        ttk.Label(frame, text="Step:").grid(row=row, column=0, sticky=tk.W, padx=2, pady=2)
        self.step_var = tk.StringVar(value="1,1,1")
        ttk.Entry(frame, textvariable=self.step_var, width=10).grid(row=row, column=1, padx=2, pady=2)
        row += 1

        ttk.Label(frame, text="Algorithm:").grid(row=row, column=0, sticky=tk.W, padx=2, pady=2)
        self.algo_var = tk.StringVar(value="cosmod")
        algo_combo = ttk.Combobox(frame, textvariable=self.algo_var,
                                  values=("cos", "mod", "cosmod"), width=8)
        algo_combo.grid(row=row, column=1, padx=2, pady=2, sticky=tk.W)
        row += 1

        ttk.Label(frame, text="w1:").grid(row=row, column=0, sticky=tk.W, padx=2, pady=2)
        self.w1_var = tk.DoubleVar(value=1.0)
        ttk.Entry(frame, textvariable=self.w1_var, width=6).grid(row=row, column=1, padx=2, pady=2, sticky=tk.W)
        ttk.Label(frame, text="w2:").grid(row=row, column=2, sticky=tk.W, padx=2, pady=2)
        self.w2_var = tk.DoubleVar(value=1.0)
        ttk.Entry(frame, textvariable=self.w2_var, width=6).grid(row=row, column=3, padx=2, pady=2, sticky=tk.W)
        row += 1

        ttk.Label(frame, text="b1:").grid(row=row, column=0, sticky=tk.W, padx=2, pady=2)
        self.b1_var = tk.DoubleVar(value=0.0)
        ttk.Entry(frame, textvariable=self.b1_var, width=6).grid(row=row, column=1, padx=2, pady=2, sticky=tk.W)
        ttk.Label(frame, text="b2:").grid(row=row, column=2, sticky=tk.W, padx=2, pady=2)
        self.b2_var = tk.DoubleVar(value=0.0)
        ttk.Entry(frame, textvariable=self.b2_var, width=6).grid(row=row, column=3, padx=2, pady=2, sticky=tk.W)
        row += 1

        ttk.Label(frame, text="Channel:").grid(row=row, column=0, sticky=tk.W, padx=2, pady=2)
        self.channel_var = tk.IntVar(value=0)
        channel_spin = tk.Spinbox(frame, from_=0, to=10, width=5, textvariable=self.channel_var)
        channel_spin.grid(row=row, column=1, padx=2, pady=2, sticky=tk.W)
        row += 1
        return frame

    def run(self, image_data, **params):
        ws = tuple(map(int, self.ws_var.get().split(',')))
        off = tuple(map(int, self.off_var.get().split(',')))
        step = tuple(map(int, self.step_var.get().split(',')))
        w1 = self.w1_var.get()
        w2 = self.w2_var.get()
        b1 = self.b1_var.get()
        b2 = self.b2_var.get()
        algo_name = self.algo_var.get()
        algo_map = {'cos': core._cos, 'mod': core._mod, 'cosmod': core._cosmod}
        algo = algo_map.get(algo_name, core._cosmod)

        start = time.perf_counter()
        result = core.cos_comparison_passive(
            image_data,
            window_size=ws,
            d=off,
            step=step,
            w1=w1, w2=w2, b1=b1, b2=b2,
            algorithm=algo
        )
        elapsed = time.perf_counter() - start
        return result, {"time": elapsed}

    def display_result(self, result, canvas, status_widget):
        channel = self.channel_var.get()
        try:
            h = len(result)
            w = len(result[0]) if h else 0
            c = len(result[0][0]) if h and w else 0
            if c == 0:
                status_widget.config(text="Result has no channels")
                return
            if channel >= c:
                status_widget.config(text=f"Channel {channel} out of range (max {c-1})")
                return
            heatmap = [[result[y][x][channel] for x in range(w)] for y in range(h)]
            if canvas.display_heatmap(heatmap):
                status_widget.config(text=f"Displayed channel {channel}")
            else:
                status_widget.config(text="Failed to display heatmap")
        except Exception as e:
            status_widget.config(text=f"Display error: {e}")


class ActiveTest(TestModule):
    name = "Active Mode"
    description = "Template matching with kernel (3D kernel recommended)."

    def get_param_widgets(self, parent):
        frame = ttk.Frame(parent)
        row = 0
        ttk.Label(frame, text="Kernel (3D list):").grid(row=row, column=0, sticky=tk.W, padx=2, pady=2)
        self.kernel_var = tk.StringVar(value="[[[1,1],[1,1]]]")
        ttk.Entry(frame, textvariable=self.kernel_var, width=30).grid(row=row, column=1, padx=2, pady=2)
        ttk.Label(frame, text="(e.g. [[[1,1],[1,1]]] for 2x2x1)").grid(row=row+1, column=0, columnspan=2, sticky=tk.W, padx=2)
        row += 2

        ttk.Label(frame, text="Step:").grid(row=row, column=0, sticky=tk.W, padx=2, pady=2)
        self.step_var = tk.StringVar(value="1,1")
        ttk.Entry(frame, textvariable=self.step_var, width=10).grid(row=row, column=1, padx=2, pady=2)
        row += 1

        ttk.Label(frame, text="Algorithm:").grid(row=row, column=0, sticky=tk.W, padx=2, pady=2)
        self.algo_var = tk.StringVar(value="cosmod")
        algo_combo = ttk.Combobox(frame, textvariable=self.algo_var,
                                  values=("cos", "mod", "cosmod"), width=8)
        algo_combo.grid(row=row, column=1, padx=2, pady=2, sticky=tk.W)
        row += 1

        ttk.Label(frame, text="Channel:").grid(row=row, column=0, sticky=tk.W, padx=2, pady=2)
        self.channel_var = tk.IntVar(value=0)
        channel_spin = tk.Spinbox(frame, from_=0, to=10, width=5, textvariable=self.channel_var)
        channel_spin.grid(row=row, column=1, padx=2, pady=2, sticky=tk.W)
        row += 1
        return frame

    def run(self, image_data, **params):
        kernel = eval(self.kernel_var.get())
        if not isinstance(kernel, list) or len(kernel) == 0:
            raise ValueError("Kernel must be a non-empty nested list")
        # Ensure 3D: if kernel is 2D, wrap it
        if isinstance(kernel[0], list) and not isinstance(kernel[0][0], list):
            kernel = [kernel]  # make it 3D: [ [ [ ... ] ] ]
        step = tuple(map(int, self.step_var.get().split(',')))
        algo_name = self.algo_var.get()
        algo_map = {'cos': core._cos, 'mod': core._mod, 'cosmod': core._cosmod}
        algo = algo_map.get(algo_name, core._cosmod)

        start = time.perf_counter()
        result = core.cos_comparison_active(
            image_data,
            kernel=kernel,
            step=step,
            algorithm=algo
        )
        elapsed = time.perf_counter() - start
        return result, {"time": elapsed}

    def display_result(self, result, canvas, status_widget):
        channel = self.channel_var.get()
        try:
            h = len(result)
            w = len(result[0]) if h else 0
            c = len(result[0][0]) if h and w else 0
            if c == 0:
                status_widget.config(text="Result has no channels")
                return
            if channel >= c:
                status_widget.config(text=f"Channel {channel} out of range (max {c-1})")
                return
            heatmap = [[result[y][x][channel] for x in range(w)] for y in range(h)]
            if canvas.display_heatmap(heatmap):
                status_widget.config(text=f"Displayed channel {channel}")
            else:
                status_widget.config(text="Failed to display heatmap")
        except Exception as e:
            status_widget.config(text=f"Display error: {e}")


# ============================================================
#  Registry
# ============================================================
test_registry = {}

def register_test(module_class):
    instance = module_class()
    test_registry[instance.name] = instance

register_test(PassiveTest)
register_test(ActiveTest)


# ============================================================
#  Enhanced Canvas with Pixel Buffer, Drawing, and Image Merge
# ============================================================
class HeatmapCanvas(tk.Frame):
    def __init__(self, master, pixel_callback=None, width=512, height=512):
        super().__init__(master)
        self.pixel_callback = pixel_callback
        self.canvas_width = width
        self.canvas_height = height

        # ----- Scrollable canvas -----
        self.canvas = tk.Canvas(self, width=width, height=height, bg='white',
                                scrollregion=(0,0,width,height))
        h_scroll = tk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scroll = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.config(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        # Grid layout
        self.canvas.grid(row=0, column=0, sticky='nsew')
        h_scroll.grid(row=1, column=0, sticky='ew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ----- Pixel buffer (512x512) -----
        self.draw_buffer = tk.PhotoImage(width=width, height=height)
        # Fill with white
        self.draw_buffer.put('white', to=(0,0,width,height))

        # ----- State -----
        self.photo = None          # For loaded image or heatmap
        self.image_id = None
        self.heatmap_id = None
        self.pen_color = "black"
        self.pen_size = 2
        self.is_drawing = False
        self.last_x = self.last_y = None

        # Bind events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        self.canvas.bind("<Motion>", self.on_mouse_move)

    # ---------- Drawing ----------
    def start_draw(self, event):
        self.is_drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        if not self.is_drawing:
            return
        x, y = event.x, event.y
        # Draw on canvas
        self.canvas.create_line(self.last_x, self.last_y, x, y,
                                fill=self.pen_color, width=self.pen_size,
                                capstyle=tk.ROUND, smooth=False)
        # Draw on buffer (pixel-level)
        self._draw_line_on_buffer(self.last_x, self.last_y, x, y, self.pen_color, self.pen_size)
        self.last_x, self.last_y = x, y

    def stop_draw(self, event):
        self.is_drawing = False
        self.last_x = self.last_y = None

    def _draw_line_on_buffer(self, x0, y0, x1, y1, color, size):
        """Draw line on draw_buffer using DDA algorithm (pixel by pixel)."""
        if isinstance(color, tuple) and len(color) == 3:
            color_str = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
        else:
            color_str = color
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            self.draw_buffer.put(color_str, to=(x0, y0, x0+1, y0+1))
            return
        x_inc = dx / steps
        y_inc = dy / steps
        x = x0
        y = y0
        for _ in range(int(steps) + 1):
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= xi < self.canvas_width and 0 <= yi < self.canvas_height:
                half = size // 2
                for dy_off in range(-half, half + 1):
                    for dx_off in range(-half, half + 1):
                        px = xi + dx_off
                        py = yi + dy_off
                        if 0 <= px < self.canvas_width and 0 <= py < self.canvas_height:
                            if dx_off*dx_off + dy_off*dy_off <= (size/2)*(size/2):
                                self.draw_buffer.put(color_str, to=(px, py, px+1, py+1))
            x += x_inc
            y += y_inc

    def set_pen_color(self, color):
        self.pen_color = color

    def set_pen_size(self, size):
        self.pen_size = size

    def clear_drawings(self):
        items = self.canvas.find_all()
        for item in items:
            if item not in (self.image_id, self.heatmap_id):
                self.canvas.delete(item)
        # Reset draw_buffer to white (but if there is an image, we should re-merge it)
        self.draw_buffer.put('white', to=(0,0,self.canvas_width,self.canvas_height))
        if self.photo:
            # Re-merge the image
            self._merge_image_to_buffer(self.photo)

    # ---------- Image/Heatmap display ----------
    def _merge_image_to_buffer(self, img):
        """Copy the given PhotoImage into draw_buffer."""
        w = min(img.width(), self.canvas_width)
        h = min(img.height(), self.canvas_height)
        for y in range(h):
            row_pixels = []
            for x in range(w):
                pixel = img.get(x, y)
                if isinstance(pixel, str):
                    # already hex
                    pass
                else:
                    r, g, b = pixel
                    pixel = f'#{r:02x}{g:02x}{b:02x}'
                row_pixels.append(pixel)
            line = '{' + ' '.join(row_pixels) + '}'
            self.draw_buffer.put(line, to=(0, y, w, y+1))

    def load_image_from_photoimage(self, img):
        if self.image_id:
            self.canvas.delete(self.image_id)
        self.photo = img
        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        if self.heatmap_id:
            self.canvas.delete(self.heatmap_id)
            self.heatmap_id = None
        self._heatmap_data = None
        # Merge image into draw_buffer
        self._merge_image_to_buffer(img)

    def display_heatmap(self, data_2d):
        img = array_to_photoimage(data_2d)
        if img is None:
            return False
        if self.image_id:
            self.canvas.delete(self.image_id)
            self.image_id = None
        if self.heatmap_id:
            self.canvas.delete(self.heatmap_id)
        self.heatmap_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.photo = img
        self._heatmap_data = data_2d
        # Do NOT merge heatmap into draw_buffer (keep drawings separate)
        return True

    def clear_heatmap(self):
        if self.heatmap_id:
            self.canvas.delete(self.heatmap_id)
            self.heatmap_id = None
        self._heatmap_data = None

    def save_ps(self, filename):
        try:
            self.canvas.postscript(file=filename, colormode='color')
            return True
        except Exception:
            return False

    # ---------- Pixel inspection ----------
    def on_mouse_move(self, event):
        info = None
        x, y = event.x, event.y
        if self.photo:
            if x >= 0 and y >= 0 and x < self.photo.width() and y < self.photo.height():
                try:
                    pixel = self.photo.get(x, y)
                    if isinstance(pixel, str):
                        r = int(pixel[1:3], 16)
                        g = int(pixel[3:5], 16)
                        b = int(pixel[5:7], 16)
                    else:
                        r, g, b = pixel
                    val = None
                    if hasattr(self, '_heatmap_data') and self._heatmap_data is not None:
                        if y < len(self._heatmap_data) and x < len(self._heatmap_data[0]):
                            val = self._heatmap_data[y][x]
                    info = {'x': x, 'y': y, 'r': r, 'g': g, 'b': b, 'value': val}
                except Exception:
                    pass
        if self.pixel_callback and info:
            self.pixel_callback(info)

    # ---------- Data extraction with cropping ----------
    def get_data(self, start_x=0, start_y=0, width=None, height=None):
        """
        Extract pixel data from the draw_buffer as nested list:
        data[h][w] = [r, g, b]
        If width/height not specified, use full canvas size.
        """
        if width is None:
            width = self.canvas_width
        if height is None:
            height = self.canvas_height
        # Clamp to buffer boundaries
        end_x = min(start_x + width, self.canvas_width)
        end_y = min(start_y + height, self.canvas_height)
        data = []
        for y in range(start_y, end_y):
            row = []
            for x in range(start_x, end_x):
                pixel = self.draw_buffer.get(x, y)
                if isinstance(pixel, str):
                    r = int(pixel[1:3], 16)
                    g = int(pixel[3:5], 16)
                    b = int(pixel[5:7], 16)
                else:
                    r, g, b = pixel
                row.append([r, g, b])
            data.append(row)
        return data

    def get_photoimage(self):
        """Return a PhotoImage of the draw buffer."""
        return self.draw_buffer

    # ----- Pixel-level API for custom code -----
    def get_pixel_rgb(self, x, y):
        if not self.photo:
            return None
        try:
            pixel = self.photo.get(x, y)
            if isinstance(pixel, str):
                return (int(pixel[1:3], 16), int(pixel[3:5], 16), int(pixel[5:7], 16))
            else:
                return pixel
        except Exception:
            return None

    def set_pixel(self, x, y, color):
        if isinstance(color, tuple) and len(color) == 3:
            color = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
        self.canvas.create_line(x, y, x+1, y, fill=color, width=1)
        self.draw_buffer.put(color, to=(x, y, x+1, y+1))

    def get_heatmap_value(self, x, y):
        if hasattr(self, '_heatmap_data') and self._heatmap_data is not None:
            if y < len(self._heatmap_data) and x < len(self._heatmap_data[0]):
                return self._heatmap_data[y][x]
        return None


# ============================================================
#  Enhanced Syntax Highlighting
# ============================================================
def highlight_python_code(text_widget):
    # Clear old tags
    for tag in text_widget.tag_names():
        if tag.startswith('hl_'):
            text_widget.tag_delete(tag)

    keywords = [
        'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif',
        'else', 'except', 'exec', 'finally', 'for', 'from', 'global', 'if', 'import',
        'in', 'is', 'lambda', 'not', 'or', 'pass', 'print', 'raise', 'return', 'try',
        'while', 'with', 'yield', 'None', 'True', 'False'
    ]
    keyword_pattern = r'\b(' + '|'.join(keywords) + r')\b'
    string_pattern = r'(\".*?\"|\'.*?\')'
    comment_pattern = r'#.*$'
    # Numbers (integer, float)
    number_pattern = r'\b\d+\.?\d*\b'
    # Built-in functions
    builtins = [
        'print', 'len', 'range', 'type', 'int', 'float', 'str', 'list', 'dict',
        'tuple', 'set', 'sum', 'max', 'min', 'abs', 'sorted', 'enumerate', 'zip',
        'map', 'filter', 'any', 'all', 'isinstance', 'issubclass', 'hasattr',
        'getattr', 'setattr', 'delattr', 'dir', 'vars', 'locals', 'globals',
        'callable', 'chr', 'ord', 'bin', 'hex', 'oct', 'round', 'pow', 'divmod',
        'hash', 'id', 'repr', 'ascii', 'format', 'bytes', 'bytearray', 'memoryview',
        'object', 'property', 'staticmethod', 'classmethod'
    ]
    builtin_pattern = r'\b(' + '|'.join(builtins) + r')\b'

    def _highlight(pattern, tag, color):
        start = '1.0'
        while True:
            count_var = tk.IntVar()
            pos = text_widget.search(pattern, start, stopindex=tk.END,
                                     regexp=True, count=count_var)
            if not pos:
                break
            end = text_widget.index(f"{pos}+{count_var.get()}c")
            text_widget.tag_add(tag, pos, end)
            text_widget.tag_config(tag, foreground=color)
            start = end

    _highlight(keyword_pattern, 'hl_keyword', 'blue')
    _highlight(string_pattern, 'hl_string', 'green')
    _highlight(comment_pattern, 'hl_comment', 'red')
    _highlight(number_pattern, 'hl_number', 'orange')
    _highlight(builtin_pattern, 'hl_builtin', 'purple')


# ============================================================
#  Main Application (reorganized with Script+Shell tabs)
# ============================================================
class TestPlatform(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("cos_comparison Test Platform (Enhanced)")
        self.geometry("1400x900")

        if not CORE_AVAILABLE:
            messagebox.showwarning("Warning", "cos_comparison core not found.")

        # ----- Menu -----
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Image", command=self.open_image)
        filemenu.add_command(label="Save Current Canvas as PS", command=self.save_current_canvas)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        batchmenu = tk.Menu(menubar, tearoff=0)
        batchmenu.add_command(label="Batch Process Directory", command=self.batch_process_directory)
        batchmenu.add_command(label="Batch Parameter Sweep", command=self.batch_parameter_sweep)
        menubar.add_cascade(label="Batch", menu=batchmenu)

        self.config(menu=menubar)

        # ----- Main Paned Window -----
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # Left panel (width can be reduced)
        left_frame = ttk.Frame(main_pane, width=380)
        main_pane.add(left_frame, weight=0)

        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=1)

        # ==========================================================
        # Left Panel - Compact Layout
        # ==========================================================

        # ---- 1. Image info (single line) ----
        self.image_info = ttk.Label(left_frame, text="No image loaded", relief=tk.SUNKEN)
        self.image_info.pack(fill=tk.X, padx=5, pady=2)

        # ---- 2. Canvas settings + Synthesis (horizontal) ----
        settings_frame = ttk.LabelFrame(left_frame, text="Canvas Crop")
        settings_frame.pack(fill=tk.X, padx=5, pady=2)
        # Use grid for compactness
        ttk.Label(settings_frame, text="W:").grid(row=0, column=0, padx=1)
        self.canvas_width_var = tk.IntVar(value=512)
        ttk.Entry(settings_frame, textvariable=self.canvas_width_var, width=4).grid(row=0, column=1, padx=1)
        ttk.Label(settings_frame, text="H:").grid(row=0, column=2, padx=1)
        self.canvas_height_var = tk.IntVar(value=512)
        ttk.Entry(settings_frame, textvariable=self.canvas_height_var, width=4).grid(row=0, column=3, padx=1)
        ttk.Label(settings_frame, text="X:").grid(row=0, column=4, padx=1)
        self.start_x_var = tk.IntVar(value=0)
        ttk.Entry(settings_frame, textvariable=self.start_x_var, width=4).grid(row=0, column=5, padx=1)
        ttk.Label(settings_frame, text="Y:").grid(row=0, column=6, padx=1)
        self.start_y_var = tk.IntVar(value=0)
        ttk.Entry(settings_frame, textvariable=self.start_y_var, width=4).grid(row=0, column=7, padx=1)
        ttk.Button(settings_frame, text="Synthesize", command=self.synthesize_from_current_canvas).grid(row=1, column=0, columnspan=8, sticky=tk.EW, pady=2)

        # ---- 3. Test selection and parameters ----
        test_sel_frame = ttk.Frame(left_frame)
        test_sel_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(test_sel_frame, text="Test:").pack(side=tk.LEFT)
        self.test_var = tk.StringVar()
        self.test_combo = ttk.Combobox(test_sel_frame, textvariable=self.test_var,
                                       values=list(test_registry.keys()), state="readonly")
        self.test_combo.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.test_combo.bind('<<ComboboxSelected>>', self.on_test_selected)

        # Parameter frame (expandable)
        self.param_frame = ttk.LabelFrame(left_frame, text="Parameters")
        self.param_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        self.param_container = ttk.Frame(self.param_frame)
        self.param_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.current_param_widgets = None

        # ---- 4. Run button and time (horizontal) ----
        run_frame = ttk.Frame(left_frame)
        run_frame.pack(fill=tk.X, padx=5, pady=2)
        self.run_btn = ttk.Button(run_frame, text="Run Test (New Canvas)", command=self.run_test_new_canvas)
        self.run_btn.pack(side=tk.LEFT, padx=2)
        self.time_label = ttk.Label(run_frame, text="Time: --", relief=tk.SUNKEN)
        self.time_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # ---- 5. Drawing tools (compact horizontal) ----
        draw_frame = ttk.LabelFrame(left_frame, text="Drawing")
        draw_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(draw_frame, text="Colour:").grid(row=0, column=0, padx=1)
        self.color_btn = tk.Button(draw_frame, bg="black", width=3, command=self.choose_color)
        self.color_btn.grid(row=0, column=1, padx=1)
        ttk.Label(draw_frame, text="Size:").grid(row=0, column=2, padx=1)
        self.size_var = tk.IntVar(value=2)
        size_spin = tk.Spinbox(draw_frame, from_=1, to=50, width=3,
                               textvariable=self.size_var, command=self.update_pen_size)
        size_spin.grid(row=0, column=3, padx=1)
        ttk.Button(draw_frame, text="Clear", command=self.clear_all_drawings).grid(row=0, column=4, padx=1)

        # ---- 6. Notebook: Script | Shell (replaces old code editor) ----
        code_notebook = ttk.Notebook(left_frame)
        code_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ---- Tab 1: Script Editor ----
        script_tab = ttk.Frame(code_notebook)
        code_notebook.add(script_tab, text="Script")

        self.code_editor = scrolledtext.ScrolledText(script_tab, height=8, font=("Courier", 10))
        self.code_editor.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.code_editor.bind('<KeyRelease>', lambda e: highlight_python_code(self.code_editor))

        script_btn_frame = ttk.Frame(script_tab)
        script_btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(script_btn_frame, text="Execute Code", command=self.execute_user_code).pack(side=tk.LEFT, padx=2)
        ttk.Button(script_btn_frame, text="Clear", command=lambda: self.code_editor.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=2)

        # Insert example code (same as before)
        example = '''
# ==========================================================
# Public API for custom code
# Available: app, core, image_data, current_canvas, canvases,
# run_passive(), run_active(), display_result(), create_canvas(),
# get_canvas(), set_pixel(), get_pixel_rgb(), get_heatmap_value(),
# load_image(), synthesize_from_canvas(), set_image_data(),
# get_canvas_data(), canvas_width_var, etc.
# ==========================================================

def my_custom_analysis():
    if image_data is None:
        print("No image loaded or synthesized")
        return
    result = core.cos_comparison_passive(
        image_data,
        window_size=(5,5,1),
        d=(1,1,0),
        step=(1,1,1),
        algorithm=core._cosmod
    )
    canvas = app.create_canvas_with_name("MyAnalysis")
    if canvas:
        h = len(result); w = len(result[0]) if h else 0
        if h and w:
            heatmap = [[result[y][x][0] for x in range(w)] for y in range(h)]
            canvas.display_heatmap(heatmap)
            canvas.set_pixel(10, 10, (255,0,0))
    return result

# result = my_custom_analysis()
'''
        self.code_editor.insert(tk.END, example)
        highlight_python_code(self.code_editor)

        # ---- Tab 2: Interactive Shell ----
        shell_tab = ttk.Frame(code_notebook)
        code_notebook.add(shell_tab, text="Shell")

        # Shell output area (read-only)
        self.shell_output = scrolledtext.ScrolledText(shell_tab, height=8, font=("Courier", 10), state='disabled')
        self.shell_output.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        # Configure tags for colors (optional)
        self.shell_output.tag_config("stdout", foreground="black")
        self.shell_output.tag_config("stderr", foreground="red")
        self.shell_output.tag_config("prompt", foreground="blue")

        # Shell input line
        shell_input_frame = ttk.Frame(shell_tab)
        shell_input_frame.pack(fill=tk.X, pady=2)
        ttk.Label(shell_input_frame, text=">>> ").pack(side=tk.LEFT)
        self.shell_entry = tk.Entry(shell_input_frame, font=("Courier", 10))
        self.shell_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.shell_entry.bind('<Return>', self.shell_execute)

        # Shell namespace (persistent variables)
        self.shell_namespace = {
            'app': self,
            'core': core,
            'tk': tk,
            'ttk': ttk,
            'messagebox': messagebox,
            '__builtins__': __builtins__,
        }
        # Add a welcome message
        self.shell_append("Python shell ready. Type commands and press Enter.\n", "stdout")

        # ---- 7. Pixel info (bottom) ----
        self.pixel_info = ttk.Label(left_frame, text="Pixel: --", relief=tk.SUNKEN)
        self.pixel_info.pack(fill=tk.X, padx=5, pady=2, side=tk.BOTTOM)

        # ---- 8. Status bar (main bottom) ----
        self.status = ttk.Label(self, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # ----- Right Panel: Notebook with Canvas Tabs (unchanged) -----
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        canvas_control_frame = ttk.Frame(right_frame)
        canvas_control_frame.pack(fill=tk.X, pady=2)
        ttk.Button(canvas_control_frame, text="Add Canvas", command=self.add_canvas).pack(side=tk.LEFT, padx=2)
        ttk.Button(canvas_control_frame, text="Delete Canvas", command=self.delete_canvas).pack(side=tk.LEFT, padx=2)
        ttk.Button(canvas_control_frame, text="Rename Canvas", command=self.rename_canvas).pack(side=tk.LEFT, padx=2)

        self.canvas_dict = {}
        self.canvas_counter = 0
        self.add_canvas("Original")
        self.add_canvas("Result")

        self.current_image_data = None
        self.current_photoimage = None
        self.current_module = None
        self._last_result = None

        if test_registry:
            self.test_combo.current(0)
            self.on_test_selected()

        self.update_pixel_callback()

    # ---------- Shell methods ----------
    def shell_append(self, text, tag="stdout"):
        """Append text to the shell output widget with a given tag."""
        self.shell_output.config(state='normal')
        self.shell_output.insert(tk.END, text, tag)
        self.shell_output.see(tk.END)
        self.shell_output.config(state='disabled')

    def shell_execute(self, event=None):
        """Execute the command in the shell entry and display result."""
        cmd = self.shell_entry.get().strip()
        self.shell_entry.delete(0, tk.END)
        if not cmd:
            return
        # Display the command line with prompt
        self.shell_append(f">>> {cmd}\n", "prompt")
        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute the command in the shell namespace
                exec(cmd, self.shell_namespace)
            stdout_text = stdout_capture.getvalue()
            stderr_text = stderr_capture.getvalue()
            if stdout_text:
                self.shell_append(stdout_text, "stdout")
            if stderr_text:
                self.shell_append(stderr_text, "stderr")
        except Exception as e:
            self.shell_append(f"{type(e).__name__}: {e}\n", "stderr")
        # Also update the namespace with any new variables (already updated in exec)

    # ---------- Canvas management ----------
    def add_canvas(self, name=None):
        if name is None:
            self.canvas_counter += 1
            name = f"Canvas {self.canvas_counter}"
        orig_name = name
        i = 1
        while name in self.canvas_dict:
            name = f"{orig_name} {i}"
            i += 1
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=name)
        canvas = HeatmapCanvas(tab, pixel_callback=self.on_pixel_info, width=512, height=512)
        canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas_dict[name] = canvas
        return canvas

    def delete_canvas(self):
        if len(self.canvas_dict) <= 1:
            messagebox.showwarning("Cannot delete", "At least one canvas must remain.")
            return
        current_tab = self.notebook.select()
        if not current_tab:
            return
        tab_text = self.notebook.tab(current_tab, "text")
        if messagebox.askyesno("Delete Canvas", f"Delete canvas '{tab_text}'?"):
            self.notebook.forget(current_tab)
            del self.canvas_dict[tab_text]

    def rename_canvas(self):
        current_tab = self.notebook.select()
        if not current_tab:
            return
        old_name = self.notebook.tab(current_tab, "text")
        new_name = simpledialog.askstring("Rename Canvas", "New name:", initialvalue=old_name)
        if new_name and new_name != old_name:
            if new_name in self.canvas_dict:
                messagebox.showerror("Error", "Name already exists.")
                return
            self.canvas_dict[new_name] = self.canvas_dict.pop(old_name)
            self.notebook.tab(current_tab, text=new_name)

    def get_current_canvas(self):
        current_tab = self.notebook.select()
        if not current_tab:
            return None
        tab_text = self.notebook.tab(current_tab, "text")
        return self.canvas_dict.get(tab_text)

    def get_canvas(self, name):
        return self.canvas_dict.get(name)

    def get_current_canvas_name(self):
        current_tab = self.notebook.select()
        if not current_tab:
            return None
        return self.notebook.tab(current_tab, "text")

    def create_canvas_with_name(self, name):
        if name in self.canvas_dict:
            return self.canvas_dict[name]
        return self.add_canvas(name)

    def create_new_canvas_for_result(self, base_name="Result"):
        i = 1
        while True:
            name = f"{base_name} {i}"
            if name not in self.canvas_dict:
                break
            i += 1
        canvas = self.add_canvas(name)
        for tab_id in self.notebook.tabs():
            if self.notebook.tab(tab_id, "text") == name:
                self.notebook.select(tab_id)
                break
        return canvas

    def update_pixel_callback(self):
        for canvas in self.canvas_dict.values():
            canvas.pixel_callback = self.on_pixel_info

    def on_pixel_info(self, info):
        if info:
            val_str = f" value={info['value']:.3f}" if info['value'] is not None else ""
            self.pixel_info.config(text=f"Pixel: ({info['x']},{info['y']}) RGB=({info['r']},{info['g']},{info['b']}){val_str}")
        else:
            self.pixel_info.config(text="Pixel: --")

    # ---------- Drawing tools ----------
    def choose_color(self):
        color = colorchooser.askcolor(title="Pen Colour", initialcolor="black")
        if color[1]:
            self.color_btn.config(bg=color[1])
            for canvas in self.canvas_dict.values():
                canvas.set_pen_color(color[1])

    def update_pen_size(self):
        size = self.size_var.get()
        for canvas in self.canvas_dict.values():
            canvas.set_pen_size(size)

    def clear_all_drawings(self):
        for canvas in self.canvas_dict.values():
            canvas.clear_drawings()
        self.status.config(text="Drawings cleared")

    # ---------- Image loading (external) ----------
    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.gif *.pgm *.ppm *.png")]
        )
        if not path:
            return
        try:
            img = tk.PhotoImage(file=path)
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            return

        self.current_photoimage = img
        self.current_image_data = self._photoimage_to_data(img)
        canvas = self.get_current_canvas()
        if canvas:
            canvas.load_image_from_photoimage(img)
        w, h = img.width(), img.height()
        self.image_info.config(text=f"Image: {os.path.basename(path)} ({w}x{h})")
        self.status.config(text=f"Loaded {path}")

    def _photoimage_to_data(self, img):
        w, h = img.width(), img.height()
        data = []
        for y in range(h):
            row = []
            for x in range(w):
                pixel = img.get(x, y)
                if isinstance(pixel, str):
                    r = int(pixel[1:3], 16)
                    g = int(pixel[3:5], 16)
                    b = int(pixel[5:7], 16)
                else:
                    r, g, b = pixel
                row.append([r, g, b])
            data.append(row)
        return data

    # ---------- Synthesis from current canvas (with crop) ----------
    def synthesize_from_current_canvas(self):
        canvas = self.get_current_canvas()
        if canvas is None:
            messagebox.showwarning("Warning", "No canvas selected.")
            return
        width = self.canvas_width_var.get()
        height = self.canvas_height_var.get()
        start_x = self.start_x_var.get()
        start_y = self.start_y_var.get()
        data = canvas.get_data(start_x, start_y, width, height)
        # Create a PhotoImage from data for display
        img = tk.PhotoImage(width=width, height=height)
        for y in range(height):
            row_pixels = []
            for x in range(width):
                r, g, b = data[y][x]
                row_pixels.append(f'#{r:02x}{g:02x}{b:02x}')
            img.put('{' + ' '.join(row_pixels) + '}', to=(0, y, width, y+1))
        self.current_photoimage = img
        self.current_image_data = data
        self.image_info.config(text=f"Synthesized from canvas ({width}x{height})")
        self.status.config(text="Synthesized image data from canvas.")

    # ---------- Test selection ----------
    def on_test_selected(self, event=None):
        name = self.test_var.get()
        if name not in test_registry:
            return
        self.current_module = test_registry[name]
        if self.current_param_widgets:
            self.current_param_widgets.destroy()
        self.current_param_widgets = self.current_module.get_param_widgets(self.param_container)
        self.current_param_widgets.pack(fill=tk.BOTH, expand=True)
        self.status.config(text=f"Selected: {name}")

    # ---------- Run test (creates new canvas) ----------
    def run_test_new_canvas(self):
        if not CORE_AVAILABLE:
            messagebox.showerror("Error", "cos_comparison core not available")
            return
        if self.current_module is None:
            messagebox.showwarning("Warning", "Select a test module")
            return
        if self.current_image_data is None:
            messagebox.showwarning("Warning", "Load an image or synthesize from canvas first")
            return

        self.status.config(text="Running...")
        self.run_btn.config(state=tk.DISABLED)
        self.update()

        try:
            result, metadata = self.current_module.run(self.current_image_data)
        except Exception as e:
            self.status.config(text=f"Error: {e}")
            self.run_btn.config(state=tk.NORMAL)
            messagebox.showerror("Test Error", str(e))
            return

        elapsed = metadata.get('time', 0.0)
        self.time_label.config(text=f"Time: {elapsed:.3f}s")
        self.status.config(text="Test complete")

        canvas = self.create_new_canvas_for_result("Result")
        self.current_module.display_result(result, canvas, self.status)
        self._last_result = result
        self.run_btn.config(state=tk.NORMAL)

    # ---------- Batch processing ----------
    def batch_process_directory(self):
        if not CORE_AVAILABLE:
            messagebox.showerror("Error", "Core not available")
            return
        if self.current_module is None:
            messagebox.showwarning("Warning", "Select a test module")
            return

        dir_path = filedialog.askdirectory(title="Select image directory")
        if not dir_path:
            return

        extensions = ('.gif', '.pgm', '.ppm', '.png')
        files = [f for f in os.listdir(dir_path) if f.lower().endswith(extensions)]
        if not files:
            messagebox.showinfo("No images", "No supported images found in directory.")
            return

        self.status.config(text=f"Batch processing {len(files)} images...")
        self.update()

        total_time = 0
        for i, filename in enumerate(files):
            path = os.path.join(dir_path, filename)
            try:
                img = tk.PhotoImage(file=path)
                data = self._photoimage_to_data(img)
                result, meta = self.current_module.run(data)
                total_time += meta.get('time', 0)
                canvas = self.create_new_canvas_for_result(f"Batch {i+1}")
                self.current_module.display_result(result, canvas, self.status)
                self.update()
            except Exception as e:
                print(f"Error processing {filename}: {e}")

        self.status.config(text=f"Batch done: {len(files)} images, total time {total_time:.3f}s")

    def batch_parameter_sweep(self):
        if not CORE_AVAILABLE:
            messagebox.showerror("Error", "Core not available")
            return
        if self.current_module is None:
            messagebox.showwarning("Warning", "Select a test module")
            return
        if self.current_image_data is None:
            messagebox.showwarning("Warning", "Load an image or synthesize first")
            return

        dialog = tk.Toplevel(self)
        dialog.title("Parameter Sweep")
        dialog.geometry("600x400")
        ttk.Label(dialog, text="Enter each parameter set on a new line.\n"
                               "Format: window_size,offset,step (e.g. 3,3,1;1,0,0;1,1,1)").pack(pady=5)
        text = scrolledtext.ScrolledText(dialog, height=15)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text.insert(tk.END, "3,3,1;1,0,0;1,1,1\n5,5,1;1,0,0;1,1,1\n7,7,1;1,0,0;1,1,1")
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=5)
        def run_sweep():
            lines = text.get(1.0, tk.END).strip().split('\n')
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split(';')
                if len(parts) != 3:
                    messagebox.showerror("Invalid format", f"Line: {line}\nExpected 3 parts separated by ';'")
                    continue
                try:
                    ws = tuple(map(int, parts[0].split(',')))
                    off = tuple(map(int, parts[1].split(',')))
                    step = tuple(map(int, parts[2].split(',')))
                except:
                    messagebox.showerror("Invalid numbers", f"Line: {line}")
                    continue
                if hasattr(self.current_module, 'ws_var'):
                    self.current_module.ws_var.set(','.join(map(str, ws)))
                    self.current_module.off_var.set(','.join(map(str, off)))
                    self.current_module.step_var.set(','.join(map(str, step)))
                try:
                    result, meta = self.current_module.run(self.current_image_data)
                    canvas = self.create_new_canvas_for_result("Sweep")
                    self.current_module.display_result(result, canvas, self.status)
                    self.update()
                except Exception as e:
                    print(f"Sweep error: {e}")
            dialog.destroy()
            self.status.config(text="Parameter sweep completed")
        ttk.Button(btn_frame, text="Run Sweep", command=run_sweep).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT)

    # ---------- User code execution (from Script tab) ----------
    def execute_user_code(self):
        code = self.code_editor.get(1.0, tk.END)
        if not code.strip():
            return
        ns = {
            'app': self,
            'core': core,
            'image_data': self.current_image_data,
            'current_canvas': self.get_current_canvas(),
            'canvases': self.canvas_dict,
            'register_test': register_test,
            'TestModule': TestModule,
            'tk': tk,
            'ttk': ttk,
            'messagebox': messagebox,
            '__builtins__': __builtins__,
            'create_canvas': self.create_canvas_with_name,
            'get_canvas': self.get_canvas,
            'get_current_canvas_name': self.get_current_canvas_name,
            'load_image': self._load_image_from_path,
            'run_passive': lambda: self._run_test_with_module('Passive Mode'),
            'run_active': lambda: self._run_test_with_module('Active Mode'),
            'display_result': self._display_result_on_canvas,
            'set_pixel': self._api_set_pixel,
            'get_pixel_rgb': self._api_get_pixel_rgb,
            'get_heatmap_value': self._api_get_heatmap_value,
            'synthesize_from_canvas': self._api_synthesize_from_canvas,
            'set_image_data': self._api_set_image_data,
            'get_canvas_data': self._api_get_canvas_data,
            'canvas_width_var': self.canvas_width_var,
            'canvas_height_var': self.canvas_height_var,
            'start_x_var': self.start_x_var,
            'start_y_var': self.start_y_var,
        }
        try:
            exec(code, ns)
            self.test_combo['values'] = list(test_registry.keys())
            self.status.config(text="Code executed successfully.")
        except Exception as e:
            messagebox.showerror("Code Execution Error", f"{e}\n{traceback.format_exc()}")
            self.status.config(text="Code execution failed")

    # ---------- API helper functions ----------
    def _load_image_from_path(self, path):
        try:
            img = tk.PhotoImage(file=path)
            self.current_photoimage = img
            self.current_image_data = self._photoimage_to_data(img)
            canvas = self.get_current_canvas()
            if canvas:
                canvas.load_image_from_photoimage(img)
            w, h = img.width(), img.height()
            self.image_info.config(text=f"Image: {os.path.basename(path)} ({w}x{h})")
            self.status.config(text=f"Loaded {path}")
            return True
        except Exception as e:
            print(f"Load error: {e}")
            return False

    def _run_test_with_module(self, module_name):
        if module_name not in test_registry:
            raise ValueError(f"Module {module_name} not found")
        module = test_registry[module_name]
        if self.current_image_data is None:
            raise ValueError("No image loaded")
        result, meta = module.run(self.current_image_data)
        return result

    def _display_result_on_canvas(self, result, canvas=None):
        if canvas is None:
            canvas = self.get_current_canvas()
        if canvas is None:
            raise ValueError("No canvas available")
        if self.current_module:
            self.current_module.display_result(result, canvas, self.status)
        else:
            try:
                h = len(result)
                w = len(result[0]) if h else 0
                c = len(result[0][0]) if h and w else 0
                if c > 0:
                    heatmap = [[result[y][x][0] for x in range(w)] for y in range(h)]
                    canvas.display_heatmap(heatmap)
                    self.status.config(text="Generic display (channel 0)")
            except Exception as e:
                self.status.config(text=f"Display error: {e}")

    def _api_set_pixel(self, canvas, x, y, color):
        if canvas is None:
            canvas = self.get_current_canvas()
        if canvas is None:
            raise ValueError("No canvas specified and no current canvas")
        canvas.set_pixel(x, y, color)

    def _api_get_pixel_rgb(self, canvas, x, y):
        if canvas is None:
            canvas = self.get_current_canvas()
        if canvas is None:
            raise ValueError("No canvas specified and no current canvas")
        return canvas.get_pixel_rgb(x, y)

    def _api_get_heatmap_value(self, canvas, x, y):
        if canvas is None:
            canvas = self.get_current_canvas()
        if canvas is None:
            raise ValueError("No canvas specified and no current canvas")
        return canvas.get_heatmap_value(x, y)

    def _api_synthesize_from_canvas(self, canvas=None):
        if canvas is None:
            canvas = self.get_current_canvas()
        if canvas is None:
            raise ValueError("No canvas specified and no current canvas")
        width = self.canvas_width_var.get()
        height = self.canvas_height_var.get()
        start_x = self.start_x_var.get()
        start_y = self.start_y_var.get()
        data = canvas.get_data(start_x, start_y, width, height)
        self.current_image_data = data
        w, h = width, height
        img = tk.PhotoImage(width=w, height=h)
        for y in range(h):
            row_pixels = []
            for x in range(w):
                r, g, b = data[y][x]
                row_pixels.append(f'#{r:02x}{g:02x}{b:02x}')
            img.put('{' + ' '.join(row_pixels) + '}', to=(0, y, w, y+1))
        self.current_photoimage = img
        self.image_info.config(text=f"Synthesized from canvas ({w}x{h})")
        self.status.config(text="Synthesized image data.")
        return data

    def _api_set_image_data(self, data):
        if not data or not data[0]:
            raise ValueError("Invalid data")
        self.current_image_data = data
        h = len(data)
        w = len(data[0])
        img = tk.PhotoImage(width=w, height=h)
        for y in range(h):
            row_pixels = []
            for x in range(w):
                r, g, b = data[y][x]
                row_pixels.append(f'#{r:02x}{g:02x}{b:02x}')
            img.put('{' + ' '.join(row_pixels) + '}', to=(0, y, w, y+1))
        self.current_photoimage = img
        self.image_info.config(text=f"Set custom data ({w}x{h})")
        self.status.config(text="Image data set.")

    def _api_get_canvas_data(self, canvas=None):
        if canvas is None:
            canvas = self.get_current_canvas()
        if canvas is None:
            raise ValueError("No canvas specified and no current canvas")
        return canvas.get_data()

    # ---------- Save canvas ----------
    def save_current_canvas(self):
        canvas = self.get_current_canvas()
        if canvas is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".ps",
                                            filetypes=[("PostScript", "*.ps")])
        if path:
            if canvas.save_ps(path):
                self.status.config(text=f"Saved to {path}")
            else:
                self.status.config(text="Save failed")


# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    app = TestPlatform()
    app.mainloop()
