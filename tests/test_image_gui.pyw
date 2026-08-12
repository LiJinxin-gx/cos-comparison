
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_image_gui.pyw - complete self-contained image testing GUI.

A standalone Tkinter application with zero third-party dependencies
(tkinter / ttk / array / io):

* Unbounded distributed canvas - pixels live in fixed-size tiles of
  array('B') blocks, so drawing never clips at an edge and negative
  coordinates work (TiledPixelBuffer).
* Pen tools (colour / size / eraser / clear), heatmap overlay, and
  pluggable image loaders (.pbm / .pgm / .ppm / .png / .gif / .rrgb).
* Script tab: persistent editor with syntax highlighting;
  execution lands in a shared namespace.
* Shell tab: IDLE-style interactive Python shell.  Code can read
  keyboard input straight from the input line - ``input()`` and
  ``sys.stdin.readline()`` are bridged to the widget, so no console
  is needed.  Ctrl+D inside a waiting input() = end of input.

Double-click to open the GUI, or run from the console::

    pythonw test_image_gui.pyw --script run.py --image photo.ppm

Options
-------
--api FILE      python file executed before the GUI starts; may call
                app.add_api('name', value) and app.set_api_hook(...).
--script FILE   python file injected into the Script tab (replaces the
                demo script) and executed immediately.
--image FILE    image loaded at startup.
--handlers FILE python file executed in the handler namespace before
                the GUI starts (loaders / savers / remote hooks).
--remote [HOST:PORT] start the optional TCP control server (off by
                default, binds 127.0.0.1 with an OS-assigned port).
                IPv4 and IPv6 addresses supported, e.g. --remote
                ::1:9000 or --remote 0.0.0.0:8000.
--smoke         headless self-test: build the window invisibly and exit.

Design
------
The API namespace can be extended by the host::

    app.add_api('name', value)
    app.set_api_hook(callable_or_dict)   # merged before every exec

The module is importable; external code can drive the platform directly
without subprocesses::

    import test_image_gui as tp
    p = tp.create_platform()       # hidden window + event pump
    p.show()                       # deiconify
    p.set_pixel(10, 10, 'red')
    p.export_canvas('Original', 'out.ppm')
    snap = p.snapshot()            # in-memory canvas persistence
    p.hide()
    p.close()

Pre-bound names inside injected code include: app, api (ApiFacade),
current_canvas, canvases, tk, ttk, messagebox, get_canvas,
create_canvas, delete_canvas, rename_canvas, canvas_names,
hide_canvas, show_canvas, export_canvas, import_canvas, load_image,
load_image_rows, get_canvas_data, get_pixel, set_pixel,
display_heatmap, get_tensor, set_tensor, tensor_shape,
platform_state, run_tests, add_test, register_external,
call_external, start_remote, stop_remote, help_api,
canvas_width_var, canvas_height_var, start_x_var, start_y_var,
__builtins__.

Remote control is opt-in: ``app.start_remote(port)``, the ``--remote``
switch or handler code opens a newline-delimited JSON TCP server bound
to 127.0.0.1 (see the Help > API Reference window for the protocol).
"""
import io
import os
import sys
import time
import json
import base64
import socket
import inspect
import queue
import array
import threading
import traceback
from contextlib import redirect_stdout, redirect_stderr, contextmanager

__all__ = [
    'PlatformApp', 'CompanionPlatform', 'create_platform',
    'create_companion', 'create_companion_platform',
    'ApiFacade', 'RemoteServer', 'HeatmapCanvas', 'InteractiveShell',
    'PixelBuffer', 'TiledPixelBuffer', 'BufferView',
    'register_image_loader', 'register_image_saver',
    'load_image_path', 'save_image_path', 'save_image_write',
    'rows_to_data', 'load_pnm', 'load_rrgb', 'load_photoimage',
    'parse_color', 'hex_rgb', 'jet_colormap', 'rows_to_photoimage',
    'heatmap_to_rows', 'highlight_code', 'api_reference_text',
    'start_remote', 'launch', 'main',
]


@contextmanager
def _redirect_stdin(stream):
    """Temporarily replace sys.stdin (stdlib has no redirect_stdin)."""
    old = sys.stdin
    sys.stdin = stream
    try:
        yield
    finally:
        sys.stdin = old
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from tkinter import scrolledtext, simpledialog
__version__ = "0.4.0"
# ============================================================

#  Colour helpers

# ============================================================

def parse_color(color):
    """Accept '#rrggbb' string, (r,g,b) tuple, [r,g,b] list (incl. from
    JSON), or common name -> (r,g,b)."""
    if isinstance(color, (tuple, list)) and len(color) == 3:
        return (int(color[0]) & 0xFF, int(color[1]) & 0xFF, int(color[2]) & 0xFF)
    if isinstance(color, str):
        s = color.strip()
        if s.startswith('#'):
            s = s[1:]
        if len(s) == 6:
            try:
                return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
            except ValueError:
                raise ValueError("invalid colour: %r" % (color,))
        named = {
            'white': (255, 255, 255), 'black': (0, 0, 0), 'red': (255, 0, 0),
            'green': (0, 255, 0), 'blue': (0, 0, 255),
            'yellow': (255, 255, 0), 'cyan': (0, 255, 255),
            'magenta': (255, 0, 255), 'gray': (128, 128, 128),
            'grey': (128, 128, 128), 'orange': (255, 165, 0),
            'pink': (255, 192, 203),
        }
        low = s.lower()
        if low in named:
            return named[low]
    raise ValueError("invalid colour: %r" % (color,))
def hex_rgb(rgb):
    r, g, b = int(rgb[0]) & 0xFF, int(rgb[1]) & 0xFF, int(rgb[2]) & 0xFF
    return '#%02x%02x%02x' % (r, g, b)
def jet_colormap(t):
    """Jet-like colormap: t in [0,1] -> (r,g,b)."""
    t = max(0.0, min(1.0, t))
    if t <= 0.25:
        return (0, int(4.0 * t * 255), 255)
    if t <= 0.5:
        return (0, 255, int((1.0 - 4.0 * (t - 0.25)) * 255))
    if t <= 0.75:
        return (int(4.0 * (t - 0.5) * 255), 255, 0)
    return (255, int((1.0 - 4.0 * (t - 0.75)) * 255), 0)
def rows_to_photoimage(rows, master=None):
    """Convert data_2d (h x w numeric) into a tk.PhotoImage heatmap.
    Auto-normalised to the data range; NaN/None treated as minimum.
    Returns None when the grid is empty.
    """
    if not rows or not rows[0]:
        return None
    h, w = len(rows), len(rows[0])
    vals = []
    for row in rows:
        for v in row:
            try:
                f = float(v)
                if not (f != f):
                    vals.append(f)
            except Exception:
                pass
    if not vals:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = min(vals), max(vals)
        if vmax - vmin < 1e-9:
            vmax = vmin + 1.0
    img = tk.PhotoImage(master=master, width=w, height=h)
    text_rows = []
    for y in range(h):
        line = '{'
        for x in range(w):
            t = 0.0
            v = rows[y][x]
            try:
                f = float(v)
                if not (f != f):
                    t = (f - vmin) / (vmax - vmin)
                    t = max(0.0, min(1.0, t))
            except Exception:
                pass
            r, g, b = jet_colormap(t)
            line += '#%02x%02x%02x ' % (r, g, b)
        text_rows.append(line[:-1] + '}')
    img.put(' '.join(text_rows))
    return img
# ============================================================

#  PixelBuffer: flat RGB array('B') storage with BufferView

# ============================================================

class BufferView(object):
    """Non-copying rectangular window over a PixelBuffer.
    get(x,y)/set(x,y,color) use view-local coordinates (0..w-1, 0..h-1).
    Similarly named methods follow parent accessors.  to_rows() makes a
    materialised copy as nested lists of (r,g,b).
    """
    __slots__ = ('buf', 'x', 'y', 'w', 'h')
    def __init__(self, buf, x, y, w, h):
        self.buf = buf
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
    @property
    def width(self):
        return self.w
    @property
    def height(self):
        return self.h
    @property
    def x0(self):
        return self.x
    @property
    def y0(self):
        return self.y
    def contains(self, x, y):
        return (0 <= x < self.w and 0 <= y < self.h)
    def get(self, x, y):
        if not self.contains(x, y):
            return None
        return self.buf.get(self.x + x, self.y + y)
    def set(self, x, y, rgb):
        if not self.contains(x, y):
            return False
        self.buf.set(self.x + x, self.y + y, rgb)
        return True
    def get_rows(self):
        rows = []
        for y in range(self.h):
            row = []
            for x in range(self.w):
                row.append(self.buf.get(self.x + x, self.y + y))
            rows.append(row)
        return rows
    def set_rows(self, rows, x=0, y=0):
        for yy, row in enumerate(rows):
            for xx, rgb in enumerate(row):
                self.set(x + xx, y + yy, rgb)
    def to_rows(self):
        return self.get_rows()
    def fill(self, rgb):
        for y in range(self.h):
            for x in range(self.w):
                self.buf.set(self.x + x, self.y + y, rgb)
    def __repr__(self):
        return '<BufferView x=%d y=%d w=%d h=%d>' % (self.x, self.y, self.w, self.h)
class PixelBuffer(object):
    """RGB image buffer backed by a single flat array('B').
    storage:   self.data   array of bytes, w*h*3 (R,G,B interleaved)
    api:       get(x,y) -> (r,g,b);  set(x,y,(r,g,b));
               get_view(x,y,w,h) -> BufferView;
               get_rows(x,y,w,h) -> list of rows of (r,g,b) tuples
    """
    __slots__ = ('width', 'height', 'data')
    def __init__(self, width, height, fill=(255, 255, 255)):
        self.width = int(width)
        self.height = int(height)
        self.data = array.array('B', bytes(fill)) * (self.width * self.height)
    @classmethod
    def from_rows(cls, rows):
        h = len(rows)
        w = len(rows[0]) if h else 0
        buf = cls(w, h)
        buf.set_rows(rows, 0, 0)
        return buf
    # ---------- access ----------

    def _ix(self, x, y):
        return (y * self.width + x) * 3
    def get(self, x, y):
        if not (0 <= x < self.width and 0 <= y < self.height):
            return None
        i = self._ix(x, y)
        return (self.data[i], self.data[i + 1], self.data[i + 2])
    def set(self, x, y, rgb):
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        i = self._ix(x, y)
        r, g, b = parse_color(rgb)
        self.data[i] = r
        self.data[i + 1] = g
        self.data[i + 2] = b
        return True
    def get_view(self, x=0, y=0, w=None, h=None):
        """Return a BufferView (no copy), clamped into bounds."""
        if w is None:
            w = self.width - x
        if h is None:
            h = self.height - y
        x2 = min(x + w, self.width)
        y2 = min(y + h, self.height)
        return BufferView(self, x, y, x2 - x, y2 - y)
    def fill(self, rgb):
        r, g, b = parse_color(rgb)
        n = self.width * self.height
        self.data[:] = array.array('B', bytes((r, g, b))) * n
    def fill_region(self, rgb, x, y, w, h):
        r, g, b = parse_color(rgb)
        for yy in range(y, min(y + h, self.height)):
            for xx in range(x, min(x + w, self.width)):
                i = self._ix(xx, yy)
                self.data[i] = r
                self.data[i + 1] = g
                self.data[i + 2] = b
    def get_rows(self, x=0, y=0, w=None, h=None):
        if w is None:
            w = self.width - x
        if h is None:
            h = self.height - y
        x2 = min(x + w, self.width)
        y2 = min(y + h, self.height)
        rows = []
        for yy in range(y, y2):
            row = []
            for xx in range(x, x2):
                row.append(self.get(xx, yy))
            rows.append(row)
        return rows
    def set_rows(self, rows, x=0, y=0):
        """Blit rows (h x w of (r,g,b)) at (x,y); clipped at edges."""
        for yy, row in enumerate(rows):
            for xx, rgb in enumerate(row):
                self.set(x + xx, y + yy, rgb)
    def get_row_slice(self, y, x0, x1):
        """Return flat list of (r,g,b) tuples for one row range."""
        out = []
        for x in range(x0, x1):
            out.append(self.get(x, y))
        return out
    # ---------- drawing ----------

    def draw_line(self, x0, y0, x1, y1, rgb, size=1):
        """DDA line with round brush of given size (legacy algorithm)."""
        r, g, b = parse_color(rgb)
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            self._brush(x0, y0, r, g, b, size)
            return
        xi = x0
        yi = y0
        xi_inc = dx / steps
        yi_inc = dy / steps
        for _ in range(steps + 1):
            self._brush(int(round(xi)), int(round(yi)), r, g, b, size)
            xi += xi_inc
            yi += yi_inc
    def _brush(self, x, y, r, g, b, size):
        """Round brush: pixels inside the radius (size/2) are painted,
        matching the legacy HeatmapCanvas implementation."""
        size = max(1, int(size))
        half = size // 2
        limit = (size / 2.0) ** 2
        for py in range(y - half, y + half + 1):
            for px in range(x - half, x + half + 1):
                if (px - x) * (px - x) + (py - y) * (py - y) <= limit:
                    if 0 <= px < self.width and 0 <= py < self.height:
                        i = self._ix(px, py)
                        self.data[i] = r
                        self.data[i + 1] = g
                        self.data[i + 2] = b
    # ---------- exports ----------

    def get_photoimage(self, master=None):
        img = tk.PhotoImage(master=master, width=self.width, height=self.height)
        text_rows = []
        for y in range(self.height):
            row = self.get_rows(0, y, self.width, 1)[0]
            text_rows.append('{' + ' '.join('#%02x%02x%02x' % tuple(p) for p in row) + '}')
        img.put(' '.join(text_rows))
        return img
    def write_ppm(self, path):
        """Save buffer as binary P6 PPM (no third-party dependency)."""
        with open(path, 'wb') as f:
            f.write(('P6\n%d %d\n255\n' % (self.width, self.height)).encode('ascii'))
            f.write(bytes(self.data))
    def __repr__(self):
        return '<PixelBuffer %dx%d>' % (self.width, self.height)
class TiledPixelBuffer(object):
    """Unbounded RGB canvas backed by fixed-size distributed tiles.
    Storage: ``self.tiles`` maps (tx, ty) tile keys to independent
    ``array('B')`` blocks of TILE_SIZE x TILE_SIZE pixels, so the drawing
    surface has no width/height and strokes are never clipped at an edge
    - any integer coordinate (including negative ones) can be written,
    and tiles are allocated lazily on first contact.
    API mirrors PixelBuffer where meaningful:
      get(x, y) / set(x, y, rgb)   unbounded; untouched tiles read as fill
      get_view(x, y, w, h)         BufferView (non-copying, unbounded)
      get_rows(x, y, w, h)         nested rows; missing pixels = fill
      set_rows(rows, x, y)         blit at any origin
      fill(rgb) / clear(w, h)
      draw_line / brush            no edge clipping
      bounds() -> (min_x, min_y, max_x, max_y) or None when empty
      tile_count() / tiles()       distributed-state introspection
      get_photoimage(x, y, w, h)   PhotoImage of a world rectangle
      write_ppm(path, x, y, w, h)  explicit-region binary P6 export
    """
    TILE = 64
    __slots__ = ('tile_map', 'fill_rgb', '_minx', '_miny', '_maxx', '_maxy')
    def __init__(self, width=0, height=0, fill=(255, 255, 255)):
        self.tile_map = {}
        self.fill_rgb = tuple(parse_color(fill))
        self._minx = self._miny = self._maxx = self._maxy = None
        if width and height:
            self._touch(0, 0)
            self._touch(width - 1, height - 1)
    # ---------- world bounds ----------

    def bounds(self):
        """Used world rectangle (min_x, min_y, max_x, max_y) or None."""
        if self._minx is None:
            return None
        return (self._minx, self._miny, self._maxx, self._maxy)
    def _touch(self, x, y):
        if self._minx is None:
            self._minx = self._maxx = x
            self._miny = self._maxy = y
        else:
            if x < self._minx:
                self._minx = x
            if x > self._maxx:
                self._maxx = x
            if y < self._miny:
                self._miny = y
            if y > self._maxy:
                self._maxy = y
    def tile_count(self):
        return len(self.tile_map)
    def tiles(self):
        return list(self.tile_map.items())
    # ---------- access ----------

    def _setpx(self, x, y, r, g, b):
        tx, ty = x // self.TILE, y // self.TILE
        tile = self.tile_map.get((tx, ty))
        if tile is None:
            tile = array.array('B', bytes(self.fill_rgb)) * (
                self.TILE * self.TILE)
            self.tile_map[(tx, ty)] = tile
        i = ((y - ty * self.TILE) * self.TILE + (x - tx * self.TILE)) * 3
        tile[i] = r
        tile[i + 1] = g
        tile[i + 2] = b
        self._touch(x, y)
    def get(self, x, y):
        x, y = int(x), int(y)
        tile = self.tile_map.get((x // self.TILE, y // self.TILE))
        if tile is None:
            return tuple(self.fill_rgb)
        i = ((y % self.TILE) * self.TILE + (x % self.TILE)) * 3
        return (tile[i], tile[i + 1], tile[i + 2])
    def set(self, x, y, rgb):
        r, g, b = parse_color(rgb)
        self._setpx(int(x), int(y), r, g, b)
        return True
    def get_view(self, x=0, y=0, w=None, h=None):
        if w is None or h is None:
            b = self.bounds()
            if b is None:
                x = y = w = h = 0
            else:
                if w is None:
                    w = b[2] - x + 1
                if h is None:
                    h = b[3] - y + 1
        return BufferView(self, x, y, w, h)
    def get_rows(self, x=0, y=0, w=None, h=None):
        x, y = int(x), int(y)
        if w is None or h is None:
            b = self.bounds()
            if b is None:
                return []
            if w is None:
                w = b[2] - int(x) + 1
            if h is None:
                h = b[3] - int(y) + 1
        w, h = int(w), int(h)
        rows = []
        for yy in range(y, y + h):
            rows.append([self.get(xx, yy) for xx in range(x, x + w)])
        return rows
    def set_rows(self, rows, x=0, y=0):
        for yy, row in enumerate(rows):
            for xx, rgb in enumerate(row):
                self.set(x + xx, y + yy, rgb)
    def fill(self, rgb):
        self.fill_rgb = tuple(parse_color(rgb))
        fill_block = array.array('B', bytes(self.fill_rgb)) * (
            self.TILE * self.TILE)
        for tile in self.tile_map.values():
            tile[:] = fill_block
    def clear(self, width=0, height=0):
        self.tile_map.clear()
        self.fill_rgb = (255, 255, 255)
        self._minx = self._miny = self._maxx = self._maxy = None
        if width and height:
            self._touch(0, 0)
            self._touch(width - 1, height - 1)
    def fill_region(self, rgb, x, y, w, h):
        r, g, b = parse_color(rgb)
        x, y = int(x), int(y)
        for yy in range(y, y + int(h)):
            for xx in range(x, x + int(w)):
                self._setpx(xx, yy, r, g, b)
    # ---------- drawing (never clipped) ----------

    def draw_line(self, x0, y0, x1, y1, rgb, size=1):
        """DDA line with round brush; crosses tiles and negative coords."""
        x0, y0, x1, y1 = (int(round(x0)), int(round(y0)),
                          int(round(x1)), int(round(y1)))
        r, g, b = parse_color(rgb)
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            self._brush(x0, y0, r, g, b, size)
            return
        xi, yi = x0, y0
        xi_inc = dx / steps
        yi_inc = dy / steps
        for _ in range(steps + 1):
            self._brush(int(round(xi)), int(round(yi)), r, g, b, size)
            xi += xi_inc
            yi += yi_inc
    def _brush(self, x, y, r, g, b, size):
        size = max(1, int(size))
        half = size // 2
        limit = (size / 2.0) ** 2
        for py in range(y - half, y + half + 1):
            for px in range(x - half, x + half + 1):
                if (px - x) * (px - x) + (py - y) * (py - y) <= limit:
                    self._setpx(px, py, r, g, b)
    # ---------- exports ----------

    def get_photoimage(self, x=0, y=0, w=None, h=None, master=None):
        if w is None or h is None:
            b = self.bounds()
            if b is None:
                x = y = 0
                w = h = 1
            else:
                x, y = b[0], b[1]
                w = b[2] - b[0] + 1
                h = b[3] - b[1] + 1
        img = tk.PhotoImage(master=master, width=w, height=h)
        text_rows = []
        for yy in range(y, y + h):
            text_rows.append('{' + ' '.join(
                '#%02x%02x%02x' % tuple(self.get(xx, yy))
                for xx in range(x, x + w)) + '}')
        img.put(' '.join(text_rows))
        return img
    def write_ppm(self, path, x=0, y=0, w=None, h=None):
        """Binary P6 export of a world rectangle (default: whole bounds)."""
        if w is None or h is None:
            b = self.bounds()
            if b is None:
                x = y = w = h = 0
            else:
                x, y = b[0], b[1]
                w = b[2] - b[0] + 1
                h = b[3] - b[1] + 1
        with open(path, 'wb') as f:
            f.write(('P6\n%d %d\n255\n' % (w, h)).encode('ascii'))
            for yy in range(y, y + h):
                row = bytes(bytearray(
                    ch for xx in range(x, x + w) for ch in self.get(xx, yy)))
                f.write(row)
    def __repr__(self):
        return '<TiledPixelBuffer tiles=%d bounds=%r>' % (
            len(self.tile_map), self.bounds())
# ============================================================

#  Image loader registry (pluggable; stdlib only)

# ============================================================

IMAGE_LOADERS = {}
def register_image_loader(ext, loader):
    """Register a loader for the given extension (leading dot optional).
    loader(path) -> (width, height, rows)
    rows: list of rows, each row a list of (r, g, b) tuples.
    """
    IMAGE_LOADERS[ext.lower().lstrip('.')] = loader
def loader_for(path):
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    return IMAGE_LOADERS.get(ext)
# ---------- pure-python PNM family (P1-P6) ----------

def _pnm_header_tokens(stream):
    """Yield integer tokens from a PNM header, skipping comments."""
    while True:
        line = stream.readline()
        if not line:
            return
        if line.startswith(b'#'):
            continue
        for tok in line.split():
            if tok.startswith(b'#'):
                break
            try:
                yield int(tok)
            except ValueError:
                pass
def _pnm_scale(v, maxv):
    if maxv <= 0 or maxv == 255:
        return v if maxv else 0
    return int(v * 255 / maxv)
def heatmap_to_rows(values):
    """Normalised heatmap grid -> list of rows of (r,g,b) tuples.
    Same auto-normalisation as rows_to_photoimage, rendered through
    jet_colormap.  Returns None for empty input.
    """
    if not values or not values[0]:
        return None
    h, w = len(values), len(values[0])
    vals = []
    for row in values:
        for v in row:
            try:
                f = float(v)
                if not (f != f):
                    vals.append(f)
            except Exception:
                pass
    if not vals:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = min(vals), max(vals)
        if vmax - vmin < 1e-9:
            vmax = vmin + 1.0
    out = []
    for row in values:
        outrow = []
        for v in row:
            t = 0.0
            try:
                f = float(v)
                if not (f != f):
                    t = (f - vmin) / (vmax - vmin)
                    t = max(0.0, min(1.0, t))
            except Exception:
                pass
            outrow.append(jet_colormap(t))
        out.append(outrow)
    return out
def load_pnm(path):
    """Load PNM (P1-P6, ASCII or binary). Returns (w, h, rows)."""
    with open(path, 'rb') as f:
        magic = f.readline().strip()
        if len(magic) < 2 or magic[:1] != b'P':
            raise ValueError('not a PNM file: %r' % path)
        kind = chr(magic[1])
        if kind not in '123456':
            raise ValueError('unsupported PNM kind: %r' % path)
        toks = _pnm_header_tokens(f)
        try:
            w = int(next(toks))
            h = int(next(toks))
            maxv = 1 if kind == '4' else int(next(toks))
        except StopIteration:
            raise ValueError('corrupt PNM header: %r' % path)
        raw = f.read()
        return _pnm_decode(kind, w, h, maxv, raw)
def _pnm_decode(kind, w, h, maxv, raw):
    tokens = None
    if kind in '123':
        toks = []
        for line in raw.decode('latin-1').splitlines():
            for tok in line.split():
                if not tok.startswith('#'):
                    toks.append(int(tok))
        idx = 0
        rows = []
        for y in range(h):
            row = []
            for x in range(w):
                if kind == '1':  # PBM ascii: black if 1
                    v = 0 if toks[idx] else 255
                    idx += 1
                    row.append((v, v, v))
                elif kind == '2':  # PGM ascii
                    v = _pnm_scale(toks[idx], maxv)
                    idx += 1
                    row.append((v, v, v))
                else:  # P3 ascii RGB
                    r, g, b = toks[idx:idx + 3]
                    idx += 3
                    row.append((_pnm_scale(r, maxv), _pnm_scale(g, maxv),
                                _pnm_scale(b, maxv)))
                if idx >= len(toks) and x == w - 1 and y == h - 1:
                    pass
                if idx >= len(toks):
                    break
            rows.append(row)
            if idx >= len(toks):
                break
        # pad rows that are cut short due to odd input

        while len(rows) < h:
            rows.append([(0, 0, 0)] * w)
        return w, h, rows
    # binary kinds P4/P5/P6

    rows = []
    for y in range(h):
        row = []
        for x in range(w):
            if kind == '4':  # PBM bitmap, MSB first
                i = y * ((w + 7) // 8) + x // 8
                byte = raw[i] if i < len(raw) else 0
                bit = (byte >> (7 - (x % 8))) & 1
                v = 0 if bit else 255
                row.append((v, v, v))
            elif kind == '5':  # PGM grayscale
                i = y * w + x
                v = _pnm_scale(raw[i] if i < len(raw) else 0, maxv)
                row.append((v, v, v))
            else:  # P6 RGB
                i = (y * w + x) * 3
                if i + 2 < len(raw):
                    r, g, b = raw[i], raw[i + 1], raw[i + 2]
                    row.append((_pnm_scale(r, maxv), _pnm_scale(g, maxv),
                                _pnm_scale(b, maxv)))
                else:
                    row.append((0, 0, 0))
        rows.append(row)
    return w, h, rows
# ---------- PhotoImage fallback: PNG / GIF ----------

def load_photoimage(path):
    """(w, h, rows) via tk.PhotoImage; supports png/gif/ppm/pgm."""
    img = tk.PhotoImage(file=path)
    w, h = img.width(), img.height()
    rows = []
    for y in range(h):
        row = []
        for x in range(w):
            pix = img.get(x, y)
            if isinstance(pix, str):
                row.append((int(pix[1:3], 16), int(pix[3:5], 16),
                            int(pix[5:7], 16)))
            else:
                row.append(tuple(pix))
        rows.append(row)
    return w, h, rows
# ---------- raw RGB (.rrgb) ----------

def load_rrgb(path):
    """Raw interleaved RGB bytes; autodetect square size from file size."""
    data = open(path, 'rb').read()
    n = len(data) // 3
    w = int(n ** 0.5)  # square by default
    h = n // w if w else 0
    rows = []
    i = 0
    for y in range(h):
        row = []
        for x in range(w):
            row.append((data[i], data[i + 1], data[i + 2]))
            i += 3
        rows.append(row)
    return w, h, rows
register_image_loader('pbm', load_pnm)
register_image_loader('pgm', load_pnm)
register_image_loader('ppm', load_pnm)
register_image_loader('png', load_photoimage)
register_image_loader('gif', load_photoimage)
register_image_loader('rrgb', load_rrgb)
def load_image_path(path, app=None, canvas=None):
    """Load an image file through the registry; PhotoImage as fallback.
    Loaders may accept a context: loader(path, **ctx) with ctx.file_path,
    ctx.app and ctx.canvas.  Single-argument loaders keep working.
    """
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    loader = IMAGE_LOADERS.get(ext) or load_photoimage
    ctx = {'file_path': path, 'app': app, 'canvas': canvas}
    try:
        return loader(path, **ctx)
    except TypeError:
        return loader(path)
# ============================================================

#  Canvas widget

# ============================================================

class HeatmapCanvas(tk.Frame):
    """Scrollable RGB canvas backed by an unbounded TiledPixelBuffer.
    Design (performance + merging):
      * ``self.buffer`` is the single source of truth: when an image is
        loaded its pixels are blitted into the buffer, so later pen
        strokes are *merged on top of the image*; get_data() returns the
        merged result - exactly like the legacy implementation.
      * ``self.photo`` mirrors the buffer for display; pen strokes update
        only the bounding rectangle that changed (dirty-region push), not
        the whole image, which keeps freehand drawing smooth even on
        512x512 canvases.
      * Heatmaps are drawn as a separate overlay item and do not touch
        the pixel buffer.
    Public API (used by injected code and external drivers):
      get_data(x, y, w, h)        -> rows [ (r,g,b) ... ]
      get_rows(...)               alias of get_data
      get_pixel_rgb(x, y) / get_pixel(x, y)
      set_pixel(x, y, color)      -> bool
      get_heatmap_value(x, y)
      display_heatmap(values)     -> bool (heatmap overlay)
      load_image(rows) / load_rows(rows)  -> (w, h)
      clear_drawings() / clear_all()
      get_buffer()                -> PixelBuffer (live reference)
      get_view(x,y,w,h)           -> BufferView
      save_ps(filename)           -> bool
      get_photoimage()            -> PhotoImage of merged buffer
      pixel_dict()                -> dict for small regions
    """
    def __init__(self, master, width=512, height=512, pixel_callback=None):
        super().__init__(master)
        self.canvas_width = width
        self.canvas_height = height
        self.pixel_callback = pixel_callback
        self.canvas = tk.Canvas(self, width=width, height=height, bg='white',
                                scrollregion=(0, 0, width, height),
                                highlightthickness=0)
        h_scroll = tk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scroll = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.config(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        h_scroll.grid(row=1, column=0, sticky='ew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.buffer = TiledPixelBuffer(width, height, (255, 255, 255))
        self.photo = None            # PhotoImage mirroring the buffer
        self._photo_ox = 0           # world coord of photo top-left corner
        self._photo_oy = 0
        self.image_id = None
        self.heatmap_id = None
        self.heatmap_photo = None
        self.heatmap_data = None     # last heatmap grid (for value lookup)
        self.pen_color = (0, 0, 0)
        self.pen_size = 2
        self.pen_erasing = False
        self.is_drawing = False
        self.last_x = self.last_y = None
        self._vectors = []           # stroke polylines for PS export
        self.metrics = {'draw_calls': 0, 'sync_calls': 0, 'sync_ms': 0.0,
                        'tiles': 0}
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)
        self.canvas.bind('<Motion>', self.on_mouse_move)
        self._sync_photo(full=True)
    def get_geometry(self):
        """Live geometry of this canvas' Tk widgets (monitoring support).
        Returns dict with root coords / size of the inner tk.Canvas,
        the world bounds of the pixel buffer and the tile count.
        """
        try:
            return {
                'x': self.winfo_rootx(), 'y': self.winfo_rooty(),
                'width': self.winfo_width(), 'height': self.winfo_height(),
                'canvas_width': self.canvas.winfo_width(),
                'canvas_height': self.canvas.winfo_height(),
                'mapped': bool(self.winfo_ismapped()),
                'buffer_bounds': self.buffer.bounds(),
                'tiles': self.buffer.tile_count(),
            }
        except tk.TclError:
            return {}
    # ---------- internal display sync (dirty-region aware) ----------

    def _sync_photo(self, full=False, x0=0, y0=0, x1=0, y1=0):
        """Push buffer -> PhotoImage.
        The photo only ever covers the used world rectangle; its top-left
        corner sits at (self._photo_ox, self._photo_oy) so any world
        coordinate (including negative ones) maps 1:1 onto canvas
        coordinates.  With full=False only the intersecting part of the
        rect (x0,y0)-(x1,y1) inclusive is updated (fast path)."""
        t0 = time.perf_counter()
        self.metrics['sync_calls'] += 1
        if self.photo is None or full:
            b = self.buffer.bounds()
            if b is None:
                ox, oy, w, h = 0, 0, self.canvas_width, self.canvas_height
            else:
                ox, oy, mx, my = b
                w, h = mx - ox + 1, my - oy + 1
            self._photo_ox, self._photo_oy = ox, oy
            self.photo = self.buffer.get_photoimage(ox, oy, w, h, master=self)
            if self.image_id is None:
                self.image_id = self.canvas.create_image(ox, oy, anchor='nw',
                                                         image=self.photo)
            else:
                self.canvas.coords(self.image_id, ox, oy)
                self.canvas.itemconfig(self.image_id, image=self.photo)
            self.canvas.config(scrollregion=(ox, oy, ox + w, oy + h))
            self.metrics['sync_ms'] += (time.perf_counter() - t0) * 1000.0
            self.metrics['tiles'] = self.buffer.tile_count()
            return
        ox, oy = self._photo_ox, self._photo_oy
        ax0 = max(ox, min(x0, x1))
        ay0 = max(oy, min(y0, y1))
        ax1 = min(ox + self.photo.width() - 1, max(x0, x1))
        ay1 = min(oy + self.photo.height() - 1, max(y0, y1))
        if ax1 < ax0 or ay1 < ay0:
            self.metrics['sync_ms'] += (time.perf_counter() - t0) * 1000.0
            return
        rows = self.buffer.get_rows(int(ax0), int(ay0),
                                    int(ax1) - int(ax0) + 1,
                                    int(ay1) - int(ay0) + 1)
        text_rows = ['{' + ' '.join(
            '#%02x%02x%02x' % tuple(p) for p in row) + '}' for row in rows]
        try:
            self.photo.put(' '.join(text_rows),
                           to=(int(ax0 - ox), int(ay0 - oy),
                               int(ax1 - ox + 1), int(ay1 - oy + 1)))
        except tk.TclError:
            pass
        self.metrics['sync_ms'] += (time.perf_counter() - t0) * 1000.0
    # ---------- drawing events ----------

    def start_draw(self, event):
        self.is_drawing = True
        self.last_x = self.canvas.canvasx(event.x)
        self.last_y = self.canvas.canvasy(event.y)
        self._vectors.append([(self.last_x, self.last_y)])
    def draw(self, event):
        if not self.is_drawing:
            return
        self.metrics['draw_calls'] += 1
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        x0, y0 = self.last_x, self.last_y
        if self.pen_erasing:
            self.buffer.draw_line(x0, y0, x, y, (255, 255, 255),
                                  max(self.pen_size * 3, 6))
        else:
            self.buffer.draw_line(x0, y0, x, y, self.pen_color, self.pen_size)
        # if the stroke escaped the current photo, rebuild it around the

        # grown world bounds (unbounded canvas); otherwise dirty-rect push

        b = max(self.pen_size, 1)
        pad = max(2, b if self.pen_erasing else b // 2 + 1)
        bb = self.buffer.bounds()
        if (bb is None or bb[0] < self._photo_ox or bb[1] < self._photo_oy or
                bb[2] > self._photo_ox + self.photo.width() - 1 or
                bb[3] > self._photo_oy + self.photo.height() - 1):
            self._sync_photo(full=True)
        else:
            self._sync_photo(full=False,
                             x0=min(x0, x) - pad, y0=min(y0, y) - pad,
                             x1=max(x0, x) + pad, y1=max(y0, y) + pad)
        self.last_x, self.last_y = x, y
    def stop_draw(self, event=None):
        if self._vectors and self._vectors[-1]:
            pts = self._vectors[-1]
            if len(pts) > 1:
                flat = [coord for pt in pts for coord in pt]
                self.canvas.create_line(*flat,
                                        fill='#%02x%02x%02x' % tuple(self.pen_color),
                                        width=max(self.pen_size, 1),
                                        capstyle=tk.ROUND, joinstyle=tk.ROUND,
                                        tags=('stroke',))
        self.is_drawing = False
        self.last_x = self.last_y = None
    def set_pen_erasing(self, on):
        self.pen_erasing = bool(on)
    def set_pen_color(self, color):
        self.pen_color = parse_color(color)
    def set_pen_size(self, size):
        self.pen_size = max(1, int(size))
    # ---------- buffer inspection API ----------

    def get_data(self, start_x=0, start_y=0, width=None, height=None):
        if width is None:
            width = self.canvas_width - start_x
        if height is None:
            height = self.canvas_height - start_y
        return self.buffer.get_rows(start_x, start_y, width, height)
    def get_rows(self, start_x=0, start_y=0, width=None, height=None):
        return self.get_data(start_x, start_y, width, height)
    def get_pixel_rgb(self, x, y):
        p = self.buffer.get(x, y)
        if p is None:
            return None
        return tuple(p)
    def get_pixel(self, x, y):
        return self.get_pixel_rgb(x, y)
    def set_pixel(self, x, y, color):
        ok = self.buffer.set(x, y, color)
        if ok:
            bb = self.buffer.bounds()
            if (bb is None or bb[0] < self._photo_ox or
                    bb[1] < self._photo_oy or
                    bb[2] > self._photo_ox + self.photo.width() - 1 or
                    bb[3] > self._photo_oy + self.photo.height() - 1):
                self._sync_photo(full=True)
            else:
                self._sync_photo(full=False, x0=x, y0=y, x1=x, y1=y)
        return ok
    def get_view(self, x=0, y=0, w=None, h=None):
        return self.buffer.get_view(x, y, w, h)
    def get_region_rows(self, x, y, w, h):
        return self.buffer.get_view(x, y, w, h).to_rows()
    def get_world_bounds(self):
        """Used world rectangle (min_x, min_y, max_x, max_y) or None."""
        return self.buffer.bounds()
    def get_data_full(self):
        """Rows covering the whole used world bounds (no clipping)."""
        b = self.buffer.bounds()
        if b is None:
            return []
        return self.buffer.get_rows(b[0], b[1], b[2] - b[0] + 1,
                                    b[3] - b[1] + 1)
    def get_heatmap_value(self, x, y):
        if self.heatmap_data is not None:
            try:
                return self.heatmap_data[y][x]
            except Exception:
                return None
        return None
    # ---------- image display & merging ----------

    def load_image(self, rows):
        """Merge an image into the buffer and display it.
        The image pixels become the base layer; subsequent pen strokes
        are merged on top, exactly like the legacy implementation.
        Returns (w, h).
        """
        h = len(rows)
        w = len(rows[0]) if h else 0
        if w == 0 or h == 0:
            return (0, 0)
        self.buffer = TiledPixelBuffer(w, h)
        self.buffer.set_rows(rows, 0, 0)
        self.canvas_width = w
        self.canvas_height = h
        self.canvas.config(width=w, height=h)
        self._vectors = []
        self.canvas.delete('stroke')
        self._sync_photo(full=True)
        return (w, h)
    def load_rows(self, rows):
        return self.load_image(rows)
    def get_buffer(self):
        return self.buffer
    # ---------- heatmap (overlay, does not touch buffer) ----------

    def display_heatmap(self, values):
        """Merge a heatmap into the drawing buffer (legacy single-layer).
        The heatmap pixels become the buffer base - pen strokes draw
        directly on top of them and stay visible, exactly like the old
        HeatmapCanvas._merge_image_to_buffer behaviour.  The numeric
        grid stays available via get_heatmap_value().  Returns bool.
        """
        if not values or not values[0]:
            return False
        rows = heatmap_to_rows(values)
        if rows is None:
            return False
        self.buffer = TiledPixelBuffer(len(rows[0]), len(rows))
        self.buffer.set_rows(rows, 0, 0)
        self.canvas_width = len(rows[0])
        self.canvas_height = len(rows)
        self.canvas.config(width=self.canvas_width,
                           height=self.canvas_height)
        self._vectors = []
        self.canvas.delete('stroke')
        if self.heatmap_id:
            self.canvas.delete(self.heatmap_id)
            self.heatmap_id = None
        self.heatmap_photo = None
        self.heatmap_data = values
        self._sync_photo(full=True)
        return True
    def clear_heatmap(self):
        """Dismiss heatmap state.  Merged pixels stay in the buffer
        (single-layer model); use load_image() to restore a base."""
        self.heatmap_data = None
        self.heatmap_photo = None
    # ---------- erase / clear ----------

    def clear_drawings(self):
        self.buffer.clear(self.canvas_width, self.canvas_height)
        self._vectors = []
        self.canvas.delete('stroke')
        self._sync_photo(full=True)
    def clear_all(self):
        self.clear_drawings()
        self.clear_heatmap()
    # ---------- misc ----------

    def save_ps(self, filename, should_delete_strokes=True):
        try:
            self.canvas.postscript(file=filename, colormode='color')
            return True
        except Exception:
            return False
    def on_mouse_move(self, event):
        x = int(round(self.canvas.canvasx(event.x)))
        y = int(round(self.canvas.canvasy(event.y)))
        rgb = self.buffer.get(x, y)
        hv = self.get_heatmap_value(x, y)
        info = None
        if rgb:
            info = {'x': x, 'y': y, 'r': rgb[0], 'g': rgb[1], 'b': rgb[2],
                    'value': hv}
        if self.pixel_callback and info:
            self.pixel_callback(info)
# ============================================================

#  Interactive shell (IDLE-like)

# ============================================================

import codeop
class ShellStream(io.TextIOBase):
    """Writable text stream rendering straight into a Shell widget.

    Used as stdout/stderr while injected code runs, so print() output
    and input() prompts appear immediately instead of being buffered
    until the block finishes (``captured()`` still collects the text
    for programmatic callers).
    """

    def __init__(self, shell, tag='out'):
        self._shell = shell
        self._tag = tag
        self._capture = []

    def writable(self):
        return True

    def readable(self):
        return False

    def seekable(self):
        return False

    def write(self, s):
        text = '' if s is None else str(s)
        if text:
            self._capture.append(text)
            self._shell.write(text, self._tag)
        return len(text)

    def flush(self):
        try:
            self._shell.text.see(tk.END)
        except Exception:
            pass

    def captured(self):
        return ''.join(self._capture)


class ShellStdin(io.TextIOBase):
    """Text input stream fed by the Shell widget, IDLE StdInputFile style.

    ``readline()`` simply asks the shell for the next typed line; the
    shell runs a nested Tk mainloop while waiting, so all normal
    window events (including keystrokes) keep working - exactly like
    IDLE's Console class.
    """

    def __init__(self, shell):
        self._shell = shell
        self._line_buffer = ''
        self._closed = False

    @property
    def closed(self):
        return self._closed

    def readable(self):
        return True

    def writable(self):
        return False

    def seekable(self):
        return False

    def read(self, size=-1):
        if self.closed:
            raise ValueError("read from closed file")
        if size is None:
            size = -1
        elif not isinstance(size, int):
            raise TypeError('must be int, not ' + type(size).__name__)
        result = self._line_buffer
        self._line_buffer = ''
        if size < 0:
            while line := self.readline():
                result += line
        else:
            while len(result) < size:
                line = self.readline()
                if not line:
                    break
                result += line
            self._line_buffer = result[size:]
            result = result[:size]
        return result

    def readline(self, size=-1):
        if self.closed:
            raise ValueError("read from closed file")
        if size is None:
            size = -1
        elif not isinstance(size, int):
            raise TypeError('must be int, not ' + type(size).__name__)
        line = self._line_buffer or self._shell.readline()
        if size < 0:
            size = len(line)
        eol = line.find('\n', 0, size)
        if eol >= 0:
            size = eol + 1
        self._line_buffer = line[size:]
        return line[:size]

    def close(self):
        if not self._closed:
            self._closed = True
            self._shell._unblock_stdin(True)


class InteractiveShell(tk.Frame):
    """IDLE-style single-widget interactive shell.
    One white Text widget hosts the whole transcript:
        >>> user code line 1
        ... user code line 2
        stdout / stderr / error text ...
    Only the area after the last prompt (the input region) is editable;
    earlier lines stay visible and selectable for copying - exactly
    like IDLE.  Backspace / Delete / typing are blocked inside the
    read-only history; Ctrl+C copies anywhere, Ctrl+X / Ctrl+V work on
    the input region only.
    """
    PROMPT1 = '>>> '
    PROMPT2 = '... '
    def __init__(self, master, namespace=None):
        super().__init__(master)
        self.namespace = namespace if namespace is not None else {}
        if '__builtins__' not in self.namespace:
            self.namespace['__builtins__'] = __builtins__
        self.history = []
        self._hist_pos = -1          # -1 = live edit position
        self._saved_draft = ''
        self.apply_hook = None       # set by PlatformApp: fn(namespace)
        self.exec_done_cb = None     # set by PlatformApp: fn(result dict)
        self.exec_ms = 0.0           # last execution wall time (ms)
        # ---- IDLE-style console state ----
        self.stdin = ShellStdin(self)
        self.reading = False         # blocked in readline() (input waiting)
        self.executing = False       # user code running
        self._endoffile = False      # Ctrl+D while reading
        self._running = False
        # ---- single transcript widget (white background, IDLE look) ----

        self.text = scrolledtext.ScrolledText(
            self, font=('Consolas', 11), wrap='word',
            bg='white', fg='black', insertbackground='black',
            undo=True, maxundo=-1, height=8)
        self.text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.text.tag_config('prompt', foreground='#1f5fa8')
        self.text.tag_config('in', foreground='#1f5fa8')
        self.text.tag_config('out', foreground='#202020')
        self.text.tag_config('err', foreground='#c00000')
        self.text.tag_config('sys', foreground='#008000')
        # iomark = boundary between locked history/output (before it)
        # and the editable input region (iomark .. end), IDLE-style.

        self.text.mark_set('iomark', '1.0')
        self.text.mark_gravity('iomark', tk.LEFT)
        self._bind_keys()
        self.write('Python interpreter ready - type code after the '
                   'prompt; Enter submits a complete block, '
                   'Shift+Enter/Ctrl+Enter forces submit; Ctrl+C/V/X '
                   'copy/paste/cut; Alt+P/N browse history.\n', 'sys')
        self.showprompt()
        schedule_highlight(self.text, delay=0, fn=highlight_shell)
    # ---------- key bindings ----------

    def _bind_keys(self):
        t = self.text
        t.bind('<Return>', self._on_return)
        t.bind('<Shift-Return>', self._force_submit)
        t.bind('<Control-Return>', self._force_submit)
        t.bind('<Alt-Return>', self._force_submit)
        t.bind('<Alt-t>', self._force_submit)
        t.bind('<Tab>', self._on_tab)
        t.bind('<Alt-p>', self._history_prev)
        t.bind('<Alt-n>', self._history_next)
        t.bind('<BackSpace>', self._guard_backspace)
        t.bind('<Delete>', self._guard_delete)
        t.bind('<Key>', self._guard_key)
        t.bind('<Control-v>', self._paste)
        t.bind('<Control-x>', self._cut)
        t.bind('<Control-d>', self._on_eof)
        t.bind('<KeyRelease>',
               lambda e: schedule_highlight(self.text, fn=highlight_shell))
    # ---------- prompt management ----------

    def resetoutput(self):
        """Lock the input region into history (IDLE resetoutput)."""
        try:
            if self.text.get('end-2c') != '\n':
                self.text.insert('end-1c', '\n')
            self.text.mark_set('iomark', 'end-1c')
        except tk.TclError:
            pass

    def showprompt(self):
        """Write a fresh '>>> ' prompt and open a new input region."""
        self.resetoutput()
        self.write(self.PROMPT1, 'prompt')
        try:
            self.text.mark_set('insert', 'end-1c')
        except tk.TclError:
            pass
        self.text.see(tk.END)
    def _cur_input(self):
        """Input text after the current prompt, prompt prefixes stripped."""
        try:
            txt = self.text.get('iomark', 'end-1c')
        except tk.TclError:
            return ''
        out = []
        first = True
        for ln in txt.split('\n'):
            if first:
                if ln.startswith(self.PROMPT1):
                    ln = ln[len(self.PROMPT1):]
                first = False
            elif ln.startswith(self.PROMPT2):
                ln = ln[len(self.PROMPT2):]
            out.append(ln)
        return '\n'.join(out)
    def _set_input(self, code):
        """Replace the editable input region with *code* (history recall)."""
        lines = code.split('\n')
        self.text.delete('iomark', 'end-1c')
        self.text.insert(tk.END, lines[0])
        for ln in lines[1:]:
            self.text.insert(tk.END, '\n' + self.PROMPT2 + ln)
        self.text.mark_set('insert', tk.END)
        self.text.see(tk.END)
        schedule_highlight(self.text, delay=0, fn=highlight_shell)
    # ---------- output ----------

    def write(self, text, tag='out'):
        try:
            self.text.mark_gravity('iomark', tk.RIGHT)
            self.text.insert('iomark', text, (tag,))
            self.text.mark_gravity('iomark', tk.LEFT)
        except tk.TclError:
            return
        self.text.see(tk.END)
    # ---------- core execution ----------

    def exec_code(self, code, tag='out'):
        """Execute *code* in the shared namespace.

        stdout/stderr stream into the transcript in real time and
        stdin is bridged to the keyboard input line (input() waits for
        a typed line + Enter).  Returns dict with keys: ok, value,
        stdout, stderr, ms (execution wall time in milliseconds).
        """
        if self._running:
            msg = 'execution already in progress\n'
            self.write(msg, 'err')
            return {'ok': False, 'value': None, 'stdout': '', 'stderr': msg,
                    'ms': 0.0}
        if self.apply_hook:
            try:
                self.apply_hook(self.namespace)
            except Exception as e:
                self.write('hook error: %s\n' % e, 'err')
        self._running = True
        self.executing = True
        out = ShellStream(self, 'out')
        err = ShellStream(self, 'err')
        ok = True
        value = None
        t0 = time.perf_counter()
        try:
            with redirect_stdout(out), redirect_stderr(err), \
                    _redirect_stdin(self.stdin):
                code_obj = compile(code, '<injected>', 'exec')
                exec(code_obj, self.namespace)
        except Exception:
            ok = False
            err.write(traceback.format_exc())
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            self.exec_ms = ms
            self.stdin._line_buffer = ''
            self.reading = False
            self.executing = False
            self._running = False
        return {'ok': ok, 'value': value, 'stdout': out.captured(),
                'stderr': err.captured(), 'ms': ms}
    # ---------- block completeness (IDLE: codeop) ----------

    @staticmethod
    def _incomplete(code):
        """True when *code* is an unfinished Python block (IDLE logic)."""
        if not code.strip():
            return False
        try:
            result = codeop.compile_command(code, '<stdin>', 'exec')
        except (SyntaxError, OverflowError, ValueError):
            return False
        return result is None
    # ---------- document editing protection (IDLE-like) ----------

    def _at_history_point(self):
        try:
            return bool(self.text.compare('insert', '<', 'iomark'))
        except tk.TclError:
            return False
    def _selection_crosses_history(self):
        try:
            sel = self.text.tag_ranges('sel')
        except Exception:
            return False
        if not sel:
            return False
        try:
            return bool(self.text.compare(sel[0], '<', 'iomark'))
        except tk.TclError:
            return False
    def _guard_key(self, event=None):
        if event and event.state & (0x0004 | 0x20000):
            return None          # allow Ctrl / Alt chords (copy, select-all)
        if self._selection_crosses_history():
            self.text.tag_remove('sel', '1.0', tk.END)
            self.text.mark_set('insert', 'iomark')
            return 'break'
        if self._at_history_point():
            self.text.mark_set('insert', 'iomark')
            return 'break'
        return None
    def _guard_backspace(self, event=None):
        if self._at_history_point():
            self.text.mark_set('insert', 'iomark')
            return 'break'
        return None
    def _guard_delete(self, event=None):
        if self._selection_crosses_history():
            self.text.tag_remove('sel', '1.0', tk.END)
            self.text.mark_set('insert', 'iomark')
            return 'break'
        if self._at_history_point():
            self.text.mark_set('insert', 'iomark')
            return 'break'
        return None
    def _paste(self, event=None):
        try:
            txt = self.clipboard_get()
        except tk.TclError:
            return None
        self.text.mark_set('insert', 'iomark')
        self.text.insert(tk.INSERT, txt)
        return 'break'
    def _cut(self, event=None):
        if self._selection_crosses_history():
            self.text.tag_remove('sel', '1.0', tk.END)
            self.text.mark_set('insert', 'iomark')
            return 'break'
        return None
    # ---------- enter handling (IDLE enter_callback semantics) ----------

    def _on_return(self, event=None):
        if self.executing and not self.reading:
            return None          # default binding: insert a newline
        if self.reading:
            return self._submit_input_line()
        try:
            if self.text.compare('insert', '<', 'iomark'):
                # cursor in the locked history: jump to the input line
                self.text.mark_set('insert', 'end-1c')
                return 'break'
        except tk.TclError:
            pass
        block = self._cur_input()
        if not block.strip():
            self._new_blank_out()
            return 'break'
        if self._incomplete(block):
            self._continue_line()
            return 'break'
        self._submit(block)
        return 'break'
    def _force_submit(self, event=None):
        if self.executing and not self.reading:
            return None
        if self.reading:
            return self._submit_input_line()
        try:
            if self.text.compare('insert', '<', 'iomark'):
                self.text.mark_set('insert', 'end-1c')
                return 'break'
        except tk.TclError:
            pass
        block = self._cur_input()
        if not block.strip():
            self._new_blank_out()
            return 'break'
        self._submit(block)
        return 'break'
    def _new_blank_out(self):
        try:
            self.text.tag_add('in', 'iomark', 'end-1c')
        except tk.TclError:
            pass
        self.showprompt()
    def _continue_line(self):
        lines = self._cur_input().split('\n')
        last = lines[-1].rstrip()
        indent = len(last) - len(last.lstrip(' '))
        if last.endswith(':'):
            indent += 4
        self.text.insert(tk.END, '\n' + self.PROMPT2 + ' ' * indent,
                         'prompt')
        self.text.mark_set('insert', 'end-1c')
        self.text.see(tk.END)
    def _submit(self, block):
        try:
            self.text.tag_add('in', 'iomark', 'end-1c')
        except tk.TclError:
            pass
        self.history.append(block)
        self._hist_pos = len(self.history)
        self.resetoutput()          # lock the input region into history
        res = self.exec_code(block, 'out')
        if self.exec_done_cb:
            try:
                self.exec_done_cb(res)
            except Exception:
                pass
        self.showprompt()
        schedule_highlight(self.text, delay=0, fn=highlight_shell)

    # ---------- stdin (IDLE Console.readline) ----------

    def _unblock_stdin(self, eof=False):
        """Signal a blocked readline(): EOF or a typed line is ready."""
        if eof:
            self._endoffile = True
        self.reading = False
        try:
            self.winfo_toplevel().quit()   # exit the nested mainloop
        except tk.TclError:
            pass

    def _submit_input_line(self):
        """Enter while input() waits: newline + end the nested loop."""
        try:
            self.text.tag_add('in', 'iomark', 'end-1c')
        except tk.TclError:
            pass
        self.text.insert('end-1c', '\n')
        self.text.see(tk.END)
        self._unblock_stdin()
        return 'break'

    def readline(self):
        """Read one typed line; run a nested Tk mainloop while waiting.

        All normal window events (keystrokes, focus, ...) keep working
        because the nested mainloop() processes them - the IDLE way.
        Returns '' (with Ctrl+D) or the typed line with a trailing
        newline.
        """
        self.reading = True
        self._endoffile = False
        self.write('[ waiting for keyboard input - type a line and '
                   'press Enter; Ctrl+D = end of input ]\n', 'sys')
        top = self.winfo_toplevel()
        try:
            while self.reading:
                top.mainloop()
        except tk.TclError:
            self.reading = False
        if self._endoffile:
            self._endoffile = False
            self.resetoutput()
            self.showprompt()
            return ''
        line = self._cur_input()
        self.resetoutput()
        return line.rstrip('\n') + '\n'

    def _on_eof(self, event=None):
        if self.reading:
            try:
                if self.text.compare('insert', '!=', 'end-1c'):
                    return None
            except tk.TclError:
                return None
            self._unblock_stdin(True)
            return 'break'
        if not self.executing:
            return None
        return None
    # ---------- tab ----------

    def _on_tab(self, event=None):
        try:
            sel = self.text.tag_ranges('sel')
        except Exception:
            sel = None
        if sel and (not self._selection_crosses_history()):
            indented = []
            for i, ln in enumerate(
                    self.text.get('sel.first', 'sel.last').split('\n')):
                indented.append(('    ' if i == 0 else '\n    ') + ln)
            self.text.replace('sel.first', 'sel.last', ''.join(indented))
        else:
            self.text.insert(tk.INSERT, '    ')
        return 'break'
    # ---------- history ----------

    def _history_prev(self, event=None):
        if not self.history:
            return 'break'
        if self._hist_pos == len(self.history):
            self._saved_draft = self._cur_input()
        if self._hist_pos > 0:
            self._hist_pos -= 1
        if 0 <= self._hist_pos < len(self.history):
            self._set_input(self.history[self._hist_pos])
        return 'break'
    def _history_next(self, event=None):
        if not self.history or self._hist_pos >= len(self.history):
            return 'break'
        self._hist_pos += 1
        if self._hist_pos < len(self.history):
            self._set_input(self.history[self._hist_pos])
        else:
            self._set_input(getattr(self, '_saved_draft', ''))
        return 'break'
    # ---------- programmatic api ----------

    def run_complete_source(self, code):
        """Programmatic call: run full source through the shell."""
        return self.exec_code(code)
    def input_has_content(self):
        return bool(self._cur_input().strip())
    def clear_transcript(self):
        """Wipe the transcript and start a fresh prompt."""
        try:
            self.text.delete('1.0', tk.END)
        except tk.TclError:
            return
        self.text.mark_set('iomark', '1.0')
        self.showprompt()
    # compatibility alias for older external callers

    @property
    def entry(self):
        return self.text
PY_KEYWORDS = (
    'and as assert async await break class continue def del elif else '
    'except finally for from global if import in is lambda nonlocal not or '
    'pass raise return try while with yield').split()
PY_BUILTINS = (
    'print len range type int float str list dict tuple set sum max min abs '
    'sorted enumerate zip map filter any all isinstance getattr hasattr '
    'None True False self super object exec eval compile open bytes bytearray'
    ).split()
PY_CONS = ('None', 'True', 'False')
# ============================================================

#  Syntax highlighting (stdlib re only)

# ============================================================

import re as _re
COLOR_MAP = {'keyword': 'blue', 'string': 'brown', 'comment': 'red',
             'number': 'darkorange', 'builtin': 'purple',
             'constant': 'orange', 'decorator': 'darkgreen'}
_HL_TAGGED = set()
def _ensure_tags(node):
    key = id(node)
    if key in _HL_TAGGED:
        return
    for kind, colour in COLOR_MAP.items():
        node.tag_configure('hl_' + kind, foreground=colour)
    _HL_TAGGED.add(key)
KEYWORDS_SET = set(PY_KEYWORDS)
CONSTANTS_SET = set(PY_CONS)
BUILTINS_SET = set(PY_BUILTINS)
TOKEN_RE = _re.compile(
    r'(?P<string3>"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')'
    r'|(?P<string>(?:[rRbBuUfF]{0,2})'
    r'(?:"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'))'
    r'|(?P<comment>#[^\n]*)'
    r'|(?P<number>\b(?:0[xX][0-9a-fA-F_]+|'
    r'\d+(?:\.\d+)?(?:[eE][+-]?\d+)?[lL]?)\b)'
    r'|(?P<decorator>@(?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]*)'
    r'|(?P<name>\b[A-Za-z_][A-Za-z0-9_]*\b)')
def _hl(node, src, m, kind):
    val = m.group()
    prefix = src[:m.start()]
    line_no = prefix.count('\n') + 1
    last_nl = prefix.rfind('\n')
    col = m.start() if last_nl < 0 else m.start() - last_nl - 1
    pos = '%d.%d' % (line_no, col)
    end_pos = pos + ' +%dc' % len(val)
    node.tag_add('hl_' + kind, pos, end_pos)
def highlight_code(node):
    """Full re-highlight of a tk.Text widget (moderate size is fine)."""
    try:
        _ensure_tags(node)
        total = node.index('end-1c')
    except Exception:
        return
    for kind in COLOR_MAP:
        try:
            node.tag_remove('hl_' + kind, '1.0', total)
        except Exception:
            pass
    try:
        src = node.get('1.0', total)
    except Exception:
        return
    if not src:
        return
    if len(src) > 262144:
        return
    try:
        for m in TOKEN_RE.finditer(src):
            kind = m.lastgroup
            if kind is None:
                continue
            if kind == 'string3':
                kind = 'string'
            val = m.group()
            if kind == 'name':
                if val in KEYWORDS_SET:
                    kind = 'keyword'
                elif val in CONSTANTS_SET:
                    kind = 'constant'
                elif val in BUILTINS_SET:
                    kind = 'builtin'
                else:
                    continue
            _hl(node, src, m, kind)
    except Exception:
        pass
def highlight_shell(shell):
    """Highlight code in an IDLE-style interactive shell: only lines
    that carry code (prompt lines and the editable input region after
    the iomark) get colours; plain stdout/stderr lines stay plain."""
    if not hasattr(shell, 'text'):
        shell = _shell_owner_of(shell)
    t = shell.text
    try:
        _ensure_tags(t)
        total = t.index('end-1c')
        iomark = t.index('iomark')
    except Exception:
        return
    try:
        iomark_line, iomark_col = (int(x) for x in iomark.split('.'))
    except Exception:
        iomark_line, iomark_col = 1, 0
    for kind in COLOR_MAP:
        try:
            t.tag_remove('hl_' + kind, '1.0', total)
        except Exception:
            pass
    try:
        src = t.get('1.0', total)
    except Exception:
        return
    if not src:
        return
    if len(src) > 262144:
        return
    lines = src.split('\n')
    in_triple = None
    try:
        for i, ln in enumerate(lines):
            line_no = i + 1
            code = None
            col = 0
            if line_no > iomark_line:
                if ln.startswith(shell.PROMPT2):
                    code, col = ln[len(shell.PROMPT2):], len(shell.PROMPT2)
                else:
                    code, col = ln, 0
            elif line_no == iomark_line:
                if ln.startswith(shell.PROMPT1) or \
                        ln.startswith(shell.PROMPT2):
                    pre = shell.PROMPT1 if ln.startswith(shell.PROMPT1) \
                        else shell.PROMPT2
                    code, col = ln[len(pre):], len(pre)
                else:
                    code, col = ln[iomark_col:], iomark_col
            else:
                if ln.startswith(shell.PROMPT1):
                    code, col = ln[len(shell.PROMPT1):], len(shell.PROMPT1)
                elif ln.startswith(shell.PROMPT2):
                    code, col = ln[len(shell.PROMPT2):], len(shell.PROMPT2)
                else:
                    code = None
            if not code:
                if in_triple:
                    _hl_line(t, line_no, 0, len(ln), 'string')
                continue
            if in_triple:
                close = code.find(in_triple)
                if close < 0:
                    _hl_line(t, line_no, col, len(code), 'string')
                    continue
                _hl_line(t, line_no, col, close + 3, 'string')
                in_triple = None
                code, col = code[close + 3:], col + close + 3
            if not code.strip():
                continue
            for m in TOKEN_RE.finditer(code):
                kind = m.lastgroup
                if kind is None:
                    continue
                if kind == 'string3':
                    kind = 'string'
                val = m.group()
                if kind == 'name':
                    if val in KEYWORDS_SET:
                        kind = 'keyword'
                    elif val in CONSTANTS_SET:
                        kind = 'constant'
                    elif val in BUILTINS_SET:
                        kind = 'builtin'
                    else:
                        continue
                if val == '':
                    continue
                if kind == 'string' and val in ('"""', "'''"):
                    in_triple = val
                    continue
                pos = '%d.%d' % (line_no, col + m.start())
                end_pos = pos + ' +%dc' % len(val)
                t.tag_add('hl_' + kind, pos, end_pos)
    except Exception:
        pass
def _hl_line(t, line_no, col, length, kind):
    """Colour a plain slice of one line (cross-line string helper)."""
    if length <= 0:
        return
    pos = '%d.%d' % (line_no, col)
    t.tag_add('hl_' + kind, pos, pos + ' +%dc' % length)
def _shell_owner_of(text_widget):
    """Map a ScrolledText back to its InteractiveShell owner via the
    Tk parent chain (schedule_highlight may hand us the raw text widget)."""
    w = text_widget.master
    while w is not None:
        if isinstance(w, InteractiveShell):
            return w
        w = getattr(w, 'master', None)
    return text_widget
_HL_JOBS = {}
def schedule_highlight(node, delay=180, fn=None):
    """Debounced highlight: rapid typing only re-highlights after the
    editor settles for *delay* ms, avoiding flicker on big documents.
    *fn* defaults to highlight_code; pass highlight_shell for shells."""
    if fn is None:
        fn = highlight_code
    key = id(node)
    job = _HL_JOBS.pop(key, None)
    if job is not None:
        try:
            node.after_cancel(job)
        except Exception:
            pass
    try:
        _HL_JOBS[key] = node.after(
            delay, lambda: _run_highlight(node, key, fn))
        node.bind('<Destroy>', lambda e: _cancel_highlight(node, key),
                  add='+')
    except Exception:
        _HL_JOBS.pop(key, None)
def _cancel_highlight(node, key):
    job = _HL_JOBS.pop(key, None)
    if job is not None:
        try:
            node.after_cancel(job)
        except Exception:
            pass
def _run_highlight(node, key, fn):
    _HL_JOBS.pop(key, None)
    try:
        fn(node)
    except tk.TclError:
        pass
if __name__ == '__main__':
    pass
# ============================================================

#  Main platform application

# ============================================================

DEFAULT_SCRIPT = '''
# ==========================================================

# Injected-code sandbox (caller is trusted).

# Pre-bound names: app, api, canvases, current_canvas, tk,

# ttk, messagebox, get_canvas, create_canvas, delete_canvas,

# rename_canvas, load_image, save_image, get_canvas_data,

# get_pixel, set_pixel, display_heatmap, run_tests, add_test,

# canvas_width_var, canvas_height_var, start_x_var, start_y_var

# ==========================================================

# --- Example: draw three coloured dots next to each other.

def sample_test(canvas=None):
    c = canvas or current_canvas
    c.set_pixel(50, 50, (255, 0, 0))
    c.set_pixel(51, 50, (0, 255, 0))
    c.set_pixel(52, 50, (0, 0, 255))
    return c.get_pixel_rgb(52, 50)
# --- Example: 16x16 random heatmap overlay.

def show_heat():
    import random
    heat = [[random.random() for _ in range(16)] for _ in range(16)]
    return current_canvas.display_heatmap(heat)
# --- Example: register a named test, then run it.

def add_demo_test():
    def t(data):
        n = max(1, len(data) * len(data[0]))
        return sum(p[0] for row in data for p in row) / n
    add_test('mean red', t)
    return 'test registered'
# Run Script once to define helpers, then in the Shell tab call:

#   sample_test(); show_heat(); add_demo_test(); run_tests()

'''
# ============================================================

#  Image save registry (pluggable, stdlib only)

# ============================================================

IMAGE_SAVERS = {}
def register_image_saver(ext, saver):
    """Register a saver for the given extension (leading dot optional).
    saver(path, rows, app=None) or saver(path, rows, **ctx) where rows
    is a list of (r,g,b) rows and ctx carries the extended context
    (file_path, pixel_rows, tensor, widths, canvas, canvas_name, app).
    """
    IMAGE_SAVERS[ext.lower().lstrip('.')] = saver
def _save_ppm(path, rows, app=None):
    with open(path, 'wb') as f:
        h = len(rows)
        w = len(rows[0]) if h else 0
        f.write(('P6\n%d %d\n255\n' % (w, h)).encode('ascii'))
        for row in rows:
            f.write(bytes([v for px in row for v in px]))
def _save_pgm(path, rows, app=None):
    with open(path, 'wb') as f:
        h = len(rows)
        w = len(rows[0]) if h else 0
        f.write(('P5\n%d %d\n255\n' % (w, h)).encode('ascii'))
        for row in rows:
            for px in row:
                g = int(0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2])
                f.write(bytes((g,)))
def _save_ps(path, rows, app=None):
    """Save via canvas postscript when a canvas is available in app."""
    if app is None:
        raise ValueError('PS save needs an app (canvas) context')
    c = app.get_current_canvas()
    if c is None:
        raise ValueError('no canvas to export as PS')
    c.save_ps(path)
register_image_saver('ppm', _save_ppm)
register_image_saver('pgm', _save_pgm)
register_image_saver('ps', _save_ps)
# ============================================================

#  Optional remote control server (TCP + JSON lines, localhost only)

# ============================================================

class RemoteServer(object):
    """Minimal TCP control server for external / remote operation.

    Off by default.  Started via ``app.start_remote(...)``, the
    ``--remote`` CLI switch or handler code.  Binds to 127.0.0.1 by
    default; pass a different host for LAN / IPv6 use ('' = all
    interfaces with dual-stack where available).

    Protocol: newline-delimited JSON requests::

        {"cmd": "ping"}
        {"cmd": "state"}
        {"cmd": "exec", "code": "set_pixel(0, 0, 'red')"}
        {"cmd": "call", "name": "set_pixel",
         "args": [1, 2, "green"], "kwargs": {}}
        {"cmd": "hooks"}

    Replies are single-line JSON: {"ok": true, "result": ...} or
    {"ok": false, "error": "..."}.  Requests are dispatched on the Tk
    thread (a queue polled with ``after``), so Tk calls stay safe.
    """
    def __init__(self, app, port=0, host='127.0.0.1'):
        self.app = app
        self.host = host
        self._stop = False
        self._q = queue.Queue()
        self._thread = None
        self._after_id = None
        self._sock = None
        self.port = self._bind(host, port)
    def _bind(self, host, port):
        """Bind with automatic IPv4 / IPv6 / dual-stack selection.
        ``host`` may be a literal (incl. '::1'), 'localhost', an empty
        string (all interfaces) or '*'."""
        if host in ('', '*'):
            host = None
        last = None
        try:
            infos = socket.getaddrinfo(
                host, port, type=socket.SOCK_STREAM,
                proto=socket.IPPROTO_TCP,
                flags=getattr(socket, 'AI_PASSIVE', 0))
        except socket.gaierror as e:
            raise OSError('cannot resolve host %r: %s' % (host, e))
        for fam, _stype, _proto, _cname, sa in infos:
            try:
                s = socket.socket(fam, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if fam == socket.AF_INET6:
                    try:
                        s.setsockopt(socket.IPPROTO_IPV6,
                                     socket.IPV6_V6ONLY, 0)
                    except (OSError, AttributeError):
                        pass
                s.bind(sa)
                s.listen(5)
                self._sock = s
                self.host = host if host is not None else (
                    '::' if fam == socket.AF_INET6 else '0.0.0.0')
                self.addr_family = fam
                return s.getsockname()[1]
            except OSError as e:
                last = e
                try:
                    s.close()
                except OSError:
                    pass
        raise OSError('cannot bind %s:%s: %s' % (host or '*', port, last))
    # ---------- lifecycle ----------

    def start(self):
        self._thread = threading.Thread(target=self._accept_loop,
                                        daemon=True)
        self._thread.start()
        self._after_id = self.app.after(80, self._poll_queue)
    def stop(self):
        self._stop = True
        try:
            self._sock.close()
        except OSError:
            pass
        if self._after_id is not None:
            try:
                self.app.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
    # ---------- server thread ----------

    def _accept_loop(self):
        while not self._stop:
            try:
                conn, _addr = self._sock.accept()
            except OSError:
                return
            try:
                conn.settimeout(10.0)
                data = conn.recv(1 << 20)
                req = json.loads(data.decode('utf-8'))
            except Exception:
                try:
                    conn.close()
                except OSError:
                    pass
                continue
            self._q.put((conn, req))
    # ---------- Tk-thread dispatch (queue polled via after) ----------

    def _poll_queue(self):
        try:
            while True:
                conn, req = self._q.get_nowait()
                reply = self._dispatch(req)
                try:
                    conn.sendall((json.dumps(reply) + '\n').encode('utf-8'))
                except OSError:
                    pass
                finally:
                    try:
                        conn.close()
                    except OSError:
                        pass
        except queue.Empty:
            pass
        if not self._stop:
            try:
                self._after_id = self.app.after(80, self._poll_queue)
            except tk.TclError:
                return
    # ---------- request handling ----------

    def _dispatch(self, req):
        if not isinstance(req, dict):
            return {'ok': False, 'error': 'request must be a JSON object'}
        cmd = req.get('cmd')
        if cmd == 'ping':
            return {'ok': True, 'result': {'pong': True,
                                           'version': __version__}}
        if cmd == 'state':
            try:
                return {'ok': True, 'result': self.app.get_platform_state()}
            except Exception as e:
                return {'ok': False, 'error': '%s: %s'
                        % (type(e).__name__, e)}
        if cmd == 'exec':
            code = req.get('code') or ''
            try:
                res = self.app.shell.exec_code(code)
                return {'ok': bool(res.get('ok')), 'result': {
                    'value': res.get('value'),
                    'stdout': res.get('stdout', ''),
                    'stderr': res.get('stderr', ''),
                    'ms': res.get('ms', 0.0)},
                    'error': res.get('stderr') or None}
            except Exception as e:
                return {'ok': False, 'error': '%s: %s'
                        % (type(e).__name__, e)}
        if cmd in ('call', 'cat'):
            name = req.get('name')
            args = req.get('args') or []
            kwargs = req.get('kwargs') or {}
            if not isinstance(kwargs, dict):
                return {'ok': False, 'error': 'kwargs must be an object'}
            user = self.app.external_hooks.get(name)
            if user is not None:
                fn = user
            else:
                facade = self.app.api
                if name in ApiFacade.ALLOWED:
                    fn = getattr(facade, name)
                elif name in facade.registry and \
                        callable(facade.registry[name]):
                    fn = facade.registry[name]
                else:
                    return {'ok': False,
                            'error': 'unknown operation: %r' % name}
            try:
                return {'ok': True, 'result': fn(*args, **kwargs)}
            except Exception as e:
                return {'ok': False, 'error': '%s: %s'
                        % (type(e).__name__, e)}
        if cmd == 'hooks':
            return {'ok': True, 'result': {
                'external': sorted(self.app.external_hooks),
                'api': sorted(ApiFacade.ALLOWED),
                'registry': sorted(self.app.api.registry)}}
        if cmd == 'shutdown':
            try:
                self.app.stop_remote()
            except Exception:
                pass
            return {'ok': True, 'result': 'stopped'}
        return {'ok': False, 'error': 'unknown cmd: %r' % cmd}
def start_remote(app, port=0, host='127.0.0.1'):
    """Convenience: start the optional control server on an app.
    Accepts either app.start_remote's flexible signature or the legacy
    positional form (port first)."""
    if isinstance(port, str):
        host, port = port, host
    return app.start_remote(host=host, port=port)
# ============================================================

#  Help windows: API reference + live diagnostics

# ============================================================

class ApiDocsWindow(tk.Toplevel):
    """Read-only API reference browser (Help > API Reference)."""
    def __init__(self, master=None):
        super().__init__(master)
        self.title('API Reference - Image Test Platform %s' % __version__)
        self.geometry('760x620')
        txt = scrolledtext.ScrolledText(self, font=('Consolas', 10),
                                        wrap='word')
        txt.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        txt.insert('1.0', api_reference_text())
        txt.config(state='disabled')
        ttk.Button(self, text='Close', command=self.destroy
                   ).pack(pady=(0, 8), side=tk.RIGHT, padx=8)
class DiagnosticsWindow(tk.Toplevel):
    """Live monitor: platform state + component geometry + metrics.
    Auto-refreshes while open (useful for external-driver debugging).
    """
    def __init__(self, master=None):
        super().__init__(master)
        self.app = master if (master is not None and
                              hasattr(master, 'get_platform_state')) else None
        self.title('Diagnostics - Image Test Platform')
        self.geometry('660x580')
        self.txt = scrolledtext.ScrolledText(self, font=('Consolas', 9),
                                             wrap='none')
        self.txt.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        bar = ttk.Frame(self)
        bar.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text='Auto-refresh (1 s)',
                        variable=self.auto_var).pack(side=tk.LEFT,
                                                     padx=(0, 8))
        ttk.Button(bar, text='Refresh', command=self._refresh
                   ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(bar, text='Close', command=self.destroy
                   ).pack(side=tk.RIGHT)
        self._refresh()
        self._schedule()
    def _schedule(self):
        try:
            alive = self.winfo_exists()
        except tk.TclError:
            return
        if alive:
            if self.auto_var.get():
                self._refresh()
            self.after(1000, self._schedule)
    def _refresh(self):
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        try:
            if self.app is not None:
                state = self.app.get_platform_state()
            else:
                state = {'note': 'no platform app attached'}
        except Exception as e:
            state = {'error': '%s: %s' % (type(e).__name__, e)}
        text = json.dumps(state, indent=1, ensure_ascii=False)
        self.txt.config(state='normal')
        self.txt.delete('1.0', tk.END)
        self.txt.insert('1.0', text)
        self.txt.config(state='disabled')
def save_image_path(path, rows, app=None, **extra):
    """Save rows to a file via the saver registry. Suffix defines format.
    Saver dispatch is backwards compatible: savers are tried with the
    extended context (saver(path, rows, app, **ctx)), then with app and
    finally with (path, rows) only.  ctx carries the file path being
    written and the full pixel tensor (pixel_rows / tensor), so custom
    logic can adapt to canvas name, dimensions and the app instance.
    """
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    saver = IMAGE_SAVERS.get(ext) or IMAGE_SAVERS['ppm']
    c = None
    name = None
    if app is not None:
        try:
            c = app.get_current_canvas()
        except Exception:
            c = None
        try:
            name = app.current_canvas_name()
        except Exception:
            name = None
    ctx = {'file_path': path, 'pixel_rows': rows, 'app': app,
           'canvas': c, 'canvas_name': name,
           'width': len(rows[0]) if rows and rows[0] else 0,
           'height': len(rows) if rows else 0,
           'tensor': rows}
    dispatch = {k: v for k, v in ctx.items() if k != 'app'}
    try:
        saver(path, rows, app, **dispatch)
    except TypeError:
        try:
            saver(path, rows, **ctx)     # documented ``**ctx`` form
        except TypeError:
            try:
                saver(path, rows, app)   # legacy positional app
            except TypeError:
                saver(path, rows)
# ============================================================

#  Main platform application

# ============================================================

class FlowFrame(ttk.Frame):
    """Adaptive toolbar: children flow left-to-right and wrap onto
    extra rows when the available width shrinks, so buttons never get
    clipped outside the window.  Use ``flow.add(button)`` instead of
    ``button.pack(...)``.  Ideal for toolbars on resizable layouts.
    """
    MAX_ROWS = 8
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self._items = []
        self._rows = []
        self._pending = False
        self._after_job = None
        # Row frames are created eagerly (before the window is mapped)
        # and never destroyed afterwards: only the buttons move between
        # rows.  Creating new pack slaves while the root window's first
        # layout is still in flight can silently drop them, leaving
        # every button invisible until the next resize.
        for _ in range(self.MAX_ROWS):
            rf = ttk.Frame(self)
            rf.pack(side=tk.TOP, fill=tk.X, anchor=tk.W)
            self._rows.append(rf)
        self.bind('<Configure>', lambda e: self._relayout())
        self.bind('<Destroy>', lambda e: self._cancel_pending())
    def add(self, widget, padx=4, pady=2):
        self._items.append((widget, padx, pady))
        self._relayout()
        return widget
    def clear(self):
        for w, _px, _py in self._items:
            w.destroy()
        self._items = []
        for rf in self._rows:
            for ch in list(rf.winfo_children()):
                try:
                    ch.pack_forget()
                except tk.TclError:
                    pass
    def _relayout(self, event=None):
        if not self._items:
            return
        width = self.winfo_width()
        if width < 10:
            if not self._pending:
                self._pending = True
                try:
                    self._after_job = self.after(50, self._do_relayout)
                except tk.TclError:
                    self._pending = False
            return
        self._do_relayout()
    def _cancel_pending(self):
        if self._pending and self._after_job is not None:
            self._pending = False
            try:
                self.after_cancel(self._after_job)
            except Exception:
                pass
        self._after_job = None
    def _do_relayout(self):
        self._pending = False
        try:
            width = self.winfo_width()
        except tk.TclError:
            return
        if width < 10:
            return
        # split items into rows that fit the current width
        rows = []
        cur, cur_w = [], 0
        for w, px, py in self._items:
            try:
                bw = w.winfo_reqwidth()
            except tk.TclError:
                continue
            if cur and cur_w + bw > width:
                rows.append(cur)
                cur, cur_w = [], 0
            cur.append((w, px, py))
            cur_w += bw + px
        if cur:
            rows.append(cur)
        # move every child into its row; surplus rows stay empty
        for i, row in enumerate(rows[:self.MAX_ROWS]):
            rf = self._rows[i]
            for ch in list(rf.winfo_children()):
                try:
                    ch.pack_forget()
                except tk.TclError:
                    pass
            for w, px, py in row:
                w.pack(in_=rf, side=tk.LEFT, padx=px, pady=py)

class PlatformApp(tk.Tk):
    """The main test platform window.
    Tests are injected strictly through code execution (script tab and
    shell tab share one namespace).  No core library is imported;
    external callers may drive this class directly (see module docstring
    and the companion test_platform_driver.py).
    Save support: File > Export Image As... uses the IMAGE_SAVERS
    registry which the host may extend (register_image_saver).
    """
    def __init__(self, master=None, width=512, height=512):
        super().__init__(master)
        self.title('Image Test Platform - inject any algorithm via code')
        self.geometry('1320x860')
        self.minsize(900, 560)
        self.api_chains = []
        self.shell = None
        self.script_editor = None
        self.canvas_dict = {}
        self.current_image_data = None
        self._last_result = None
        self.test_registry = {}
        self.extra_test_menu = []
        # ----- modern extensibility state -----

        self.api = None                # ApiFacade instance (see _init_api)
        self.external_hooks = {}       # named callables, remote/injected
        self._hidden_canvases = set()  # canvas names with hidden tabs
        self._remote = None            # RemoteServer or None
        self._remote_poll_id = None
        self._metrics = {'started': time.time(), 'exec_count': 0,
                         'exec_ms_total': 0.0, 'last_exec_ms': 0.0}
        self._remote_started_at = None
        # ----- shared variables (crop / size params, kept compatible) -----

        self.width_var = tk.IntVar(value=width)
        self.height_var = tk.IntVar(value=height)
        self.start_x_var = tk.IntVar(value=0)
        self.start_y_var = tk.IntVar(value=0)
        self.pen_color_var = tk.StringVar(value='black')
        self.pen_size_var = tk.IntVar(value=2)
        self.test_var = tk.StringVar()
        self.theme_var = tk.StringVar(value='default')
        self.pixel_info = None   # (kept name for compatibility)
        self.status = None
        self._apply_theme()
        self._build_menu()
        self._build_layout()
        self.api = ApiFacade(self)
        self._init_api()
        # Size the window by its content's request instead of a fixed
        # size: Tk's pack layout silently drops widgets that get packed
        # while the very first layout runs in 'not enough space' mode
        # (they only reappear on resize).  Letting the window follow the
        # requested size avoids that compressed first layout entirely.
        try:
            self.tk.call('wm', 'geometry', self._w, '')
        except tk.TclError:
            pass
    # ----------------------------------------------------------

    # theme (modern look with graceful fallbacks)

    # ----------------------------------------------------------

    def _apply_theme(self):
        style = ttk.Style(self)
        theme_name = None
        candidates = ['vista', 'clam', 'default', 'classic']
        if sys.platform == 'win32':
            candidates = ['vista', 'clam', 'default']
        for cand in candidates:
            try:
                style.theme_use(cand)
                theme_name = cand
                break
            except tk.TclError:
                continue
        self._theme_name = theme_name or 'default'
        try:
            style.configure('TButton', padding=(10, 3))
            style.configure('Toolbar.TButton', padding=(12, 6),
                            font=('Segoe UI', 9))
            style.configure('TNotebook', padding=(2, 2))
            style.configure('TNotebook.Tab', padding=(14, 5))
            style.configure('TLabelframe', padding=(2, 2))
            style.configure('TLabelframe.Label', font=('Segoe UI', 9, 'bold'))
            style.configure('TLabel', padding=(1, 1))
            style.configure('TEntry', padding=(3, 2))
            style.configure('TCheckbutton', padding=2)
            style.configure('TRadiobutton', padding=(2, 2))
        except Exception:
            pass
    def _build_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label='Open Image(s)...',
                             command=self.open_images)
        filemenu.add_command(label='Import Data Grid(s)...',
                             command=self.import_data_dialog)
        filemenu.add_command(label='Export Canvas...',
                             command=self.save_canvas_dialog)
        filemenu.add_command(label='Export All Canvases...',
                             command=self.save_all_dialog)
        filemenu.add_separator()
        filemenu.add_command(label='Exit', command=self.destroy)
        menubar.add_cascade(label='File', menu=filemenu)
        canvasmenu = tk.Menu(menubar, tearoff=0)
        canvasmenu.add_command(label='Add Canvas', command=self.add_canvas)
        canvasmenu.add_command(label='Delete Canvas', command=self.delete_canvas)
        canvasmenu.add_command(label='Rename Canvas', command=self.rename_canvas)
        menubar.add_cascade(label='Canvas', menu=canvasmenu)
        pixelmenu = tk.Menu(menubar, tearoff=0)
        pixelmenu.add_command(label='Synthesize Crop Region',
                              command=self.synthesize_from_current_canvas)
        menubar.add_cascade(label='Pixel', menu=pixelmenu)
        runmenu = tk.Menu(menubar, tearoff=0)
        runmenu.add_command(label='Run Script', command=self.run_script)
        runmenu.add_command(label='Run Registered Tests',
                            command=self.run_tests)
        menubar.add_cascade(label='Run', menu=runmenu)
        viewmenu = tk.Menu(menubar, tearoff=0)
        viewmenu.add_command(label='Canvas Visibility...',
                             command=self.canvas_visibility_dialog)
        viewmenu.add_command(label='Diagnostics...',
                             command=self.open_diagnostics)
        menubar.add_cascade(label='View', menu=viewmenu)
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label='API Reference...',
                             command=self.open_api_docs)
        helpmenu.add_command(label='About', command=self.about_dialog)
        menubar.add_cascade(label='Help', menu=helpmenu)
        self.config(menu=menubar)
    def _build_layout(self):
        # ---------- root status bar (docked bottom, full width) ----------

        self.status = ttk.Label(self, text='Ready', relief=tk.SUNKEN,
                                anchor=tk.W, padding=(8, 3))
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        # ---------- main toolbar (button-first, wraps on narrow widths) ----

        self.toolbar = FlowFrame(self)
        self.toolbar.pack(fill=tk.X, padx=8, pady=(6, 2))
        for txt, cmd in (
                ('Open Image(s)', self.open_images),
                ('Import Matrix', self.import_data_dialog),
                ('Save Image', self.save_canvas_dialog),
                ('Export All', self.save_all_dialog),
                ('Handlers', self.open_handler_settings),
                ('Run Script', self.run_script),
                ('Run Tests', self.run_tests),
                ('API Docs', self.open_api_docs),
                ('About', self.about_dialog)):
            self.toolbar.add(ttk.Button(self.toolbar, text=txt, command=cmd,
                                        style='Toolbar.TButton'))
        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.paned = main
        # ---------- left column ----------

        left = ttk.Frame(main, width=420)
        main.add(left, weight=0)
        try:
            main.pane(left, minsize=280)
        except tk.TclError:
            pass
        # image info bar (same visual language as the pixel bar below)

        self.image_info = ttk.Label(left, text='No image loaded',
                                    relief=tk.SUNKEN, padding=(6, 3))
        self.image_info.pack(fill=tk.X, padx=8, pady=(0, 4))
        # ---- card: crop / synthesize ----

        sf = ttk.LabelFrame(left, text=' Crop / Synthesize ')
        sf.pack(fill=tk.X, padx=8, pady=(0, 6))
        grid = ttk.Frame(sf)
        grid.pack(fill=tk.X, padx=8, pady=(8, 4))
        params = [('W:', self.width_var), ('H:', self.height_var),
                  ('X:', self.start_x_var), ('Y:', self.start_y_var)]
        for col, (lab, var) in enumerate(params):
            ttk.Label(grid, text=lab).grid(row=0, column=col * 2,
                                           sticky='e', padx=(0, 2))
            ttk.Entry(grid, textvariable=var, width=6, justify='center'
                      ).grid(row=0, column=col * 2 + 1, sticky='w',
                             padx=(0, 12))
        actions = FlowFrame(sf)
        actions.pack(fill=tk.X, padx=8, pady=(2, 8))
        actions.add(ttk.Button(actions, text='Synthesize',
                               command=self.synthesize_from_current_canvas))
        # ---- card: drawing tools ----

        df = ttk.LabelFrame(left, text=' Drawing ')
        df.pack(fill=tk.X, padx=8, pady=(0, 6))
        tools = FlowFrame(df)
        tools.pack(fill=tk.X, padx=8, pady=(8, 8))
        tools.add(ttk.Label(tools, text='Colour:'))
        color_btn = tk.Button(tools, bg='black', width=2, relief=tk.RAISED,
                              command=self.choose_color)
        self.color_btn = color_btn
        tools.add(color_btn)
        tools.add(ttk.Label(tools, text='Size:'))
        self.size_spin = ttk.Spinbox(tools, from_=1, to=50, width=4,
                                     textvariable=self.pen_size_var,
                                     command=self.apply_pen)
        tools.add(self.size_spin)
        tools.add(ttk.Button(tools, text='Eraser',
                             command=self.toggle_eraser))
        tools.add(ttk.Button(tools, text='Clear',
                             command=self.clear_drawings))
        # ---- code notebook (script + shell) ----

        code_nb = ttk.Notebook(left)
        code_nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))
        script_tab = ttk.Frame(code_nb)
        code_nb.add(script_tab, text='  Script  ')
        self.script_editor = scrolledtext.ScrolledText(
            script_tab, font=('Consolas', 10), wrap='word', undo=True,
            height=8)
        self.script_editor.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.script_editor.insert('1.0', DEFAULT_SCRIPT)
        self.script_editor.bind('<KeyRelease>',
                                lambda e: self._schedule_highlight(
                                    self.script_editor))
        self._schedule_highlight(self.script_editor, delay=0)
        sbtns = FlowFrame(script_tab)
        sbtns.pack(fill=tk.X, padx=4, pady=(0, 4))
        sbtns.add(ttk.Button(sbtns, text='Run',
                             command=self.run_script))
        sbtns.add(ttk.Button(sbtns, text='Clear',
                             command=self.clear_script))
        sbtns.add(ttk.Button(sbtns, text='Reset',
                             command=self.reset_script))
        shell_tab = ttk.Frame(code_nb)
        code_nb.add(shell_tab, text='  Shell  ')
        shell_bar = FlowFrame(shell_tab)
        shell_bar.pack(fill=tk.X, padx=4, pady=(4, 0))
        shell_bar.add(ttk.Button(shell_bar, text='Clear Transcript',
                                 command=lambda: self.shell.clear_transcript()
                                 ))
        shell_bar.add(ttk.Label(shell_bar, text='Ctrl+P/N: history  '
                                               'Shift/Ctrl+Enter: force submit'
                               ))
        self.shell = InteractiveShell(shell_tab)
        self.shell.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.shell.exec_done_cb = self._on_shell_exec_done


        def _focus_shell_on_tab(event=None):
            if code_nb.select() == str(shell_tab):
                self.shell.text.focus_set()
                self.shell.text.focus_force()


        code_nb.bind('<<NotebookTabChanged>>', _focus_shell_on_tab)
        # pixel bar (bottom of the left column, matches image info bar)

        self.pixel_label = ttk.Label(left, text='Pixel: --', relief=tk.SUNKEN)
        self.pixel_label.pack(fill=tk.X, padx=8, pady=(2, 2), side=tk.BOTTOM)
        # ---------- right column: canvas workspace ----------

        right = ttk.Frame(main)
        main.add(right, weight=1)
        ws = ttk.LabelFrame(right, text=' Canvas Workspace ')
        ws.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        ctrl = FlowFrame(ws)
        ctrl.pack(fill=tk.X, padx=8, pady=(6, 4))
        for txt, cmd in (
                ('+ Canvas', self.add_canvas),
                ('Delete', self.delete_canvas),
                ('Rename', self.rename_canvas),
                ('Hide', self.hide_canvas),
                ('Show', self._show_recent_canvas)):
            ctrl.add(ttk.Button(ctrl, text=txt, command=cmd))
        self.canvas_nb = ttk.Notebook(ws)
        self.canvas_nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 4))
        # canvas name bar: tabs shrink when many canvases exist, so the
        # active canvas is always identifiable from this docked label
        self.canvas_name_bar = ttk.Label(ws, text='Canvas: --',
                                         relief=tk.SUNKEN, anchor=tk.W,
                                         padding=(8, 3))
        self.canvas_name_bar.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.canvas_nb.bind('<<NotebookTabChanged>>',
                            lambda e: self._update_canvas_name_bar())
        self.add_canvas('Original')
        self.add_canvas('Result')
        self.update_idletasks()
    # ----------------------------------------------------------

    # API namespace - fully injectable

    # ----------------------------------------------------------

    def add_api(self, name, value):
        """Register an extra name injected into every exec namespace.
        Applied immediately to the live namespace and kept for later
        re-injection (new shells / resets).
        """
        self.api_chains.append((name, value))
        if self.shell is not None:
            self.shell.namespace[name] = value
    def set_api_hook(self, hook):
        """hook: callable(app, ns) applied before each exec, or a dict."""
        self.api_chains.append(('__hook__', hook))
        if self.shell is not None:
            self._apply_hooks(self.shell.namespace)
    def _build_namespace(self):
        return {
            'app': self,
            'api': self.api,
            'canvases': self.canvas_dict,
            'current_canvas': self.get_current_canvas(),
            'tk': tk,
            'ttk': ttk,
            'messagebox': messagebox,
            'get_canvas': self.get_canvas,
            'create_canvas': self.create_canvas,
            'select_canvas': self.api.select_canvas,
            'delete_canvas': self.delete_canvas,
            'rename_canvas': self.rename_canvas,
            'canvas_names': self.canvas_names,
            'hide_canvas': self.hide_canvas,
            'show_canvas': self.show_canvas,
            'export_canvas': self.api.export_canvas,
            'import_canvas': self.api.import_canvas,
            'load_image': self.load_image,
            'save_image': save_image_path,
            'get_canvas_data': self.get_canvas_data,
            'get_pixel': self.api.get_pixel,
            'set_pixel': self.api.set_pixel,
            'display_heatmap': self.api.display_heatmap,
            'get_tensor': self.api.get_tensor,
            'set_tensor': self.api.set_tensor,
            'tensor_shape': self.api.tensor_shape,
            'platform_state': self.get_platform_state,
            'run_tests': self.run_tests,
            'add_test': self._add_test_stub,
            'register_external': self.register_external,
            'call_external': self.call_external,
            'start_remote': self.start_remote,
            'stop_remote': self.stop_remote,
            'help_api': self.api.help,
            # short C-API-style names (see ApiFacade aliases)
            'px': self.api.px,
            'set_px': self.api.set_px,
            'blit': self.api.blit,
            'rows': self.api.rows,
            'heatmap': self.api.heatmap,
            'canvas': self.api.canvas,
            'new_canvas': self.api.new_canvas,
            'select': self.api.select,
            'current_name': self.api.current_name,
            'drop_canvas': self.api.drop_canvas,
            'rename': self.api.rename,
            'hide': self.api.hide,
            'show': self.api.show,
            'names': self.api.names,
            'load': self.api.load,
            'export': self.api.export,
            'tensor': self.api.tensor,
            'put_tensor': self.api.put_tensor,
            'shape': self.api.shape,
            'listen': self.api.listen,
            'unlisten': self.api.unlisten,
            'rinfo': self.api.rinfo,
            'version': self.api.version,
            'canvas_width_var': self.width_var,
            'canvas_height_var': self.height_var,
            'start_x_var': self.start_x_var,
            'start_y_var': self.start_y_var,
            '__builtins__': __builtins__,
        }
    def _init_api(self):
        ns = self._build_namespace()
        self.shell.apply_hook = self._apply_hooks
        self.shell.namespace.update(ns)
        self._apply_hooks(ns)
        self.shell.namespace.update(ns)
    def _apply_hooks(self, ns):
        for name, value in self.api_chains[:]:
            if name == '__hook__':
                try:
                    if callable(value):
                        value(self, ns)
                    elif isinstance(value, dict):
                        ns.update(value)
                except Exception as e:
                    print('api hook error:', e)
            else:
                ns[name] = value
    # ----------------------------------------------------------

    # scripting

    # ----------------------------------------------------------

    def run_script(self):
        code = self.script_editor.get('1.0', 'end-1c')
        if not code.strip():
            return
        self.status.config(text='Running script...')
        res = self.shell.exec_code(code, 'out')
        self._on_shell_exec_done(res)
    def _on_shell_exec_done(self, res):
        """Shared post-exec hook: update timing metrics + status bar."""
        ms = res.get('ms', self.shell.exec_ms) if isinstance(res, dict) else 0.0
        self._metrics['exec_count'] += 1
        self._metrics['exec_ms_total'] += ms
        self._metrics['last_exec_ms'] = ms
        if isinstance(res, dict) and res.get('ok') is False:
            text = res.get('stderr', '').strip().splitlines()
            tail = text[-1] if text else 'error'
            self.status.config(text='Error: %s' % tail)
        else:
            self.status.config(text='OK (%.1f ms)' % ms)
    def clear_script(self):
        self.script_editor.delete('1.0', 'end')
    def reset_script(self):
        self.script_editor.delete('1.0', 'end')
        self.script_editor.insert('1.0', DEFAULT_SCRIPT)
        self._schedule_highlight(self.script_editor, delay=0)
    def _schedule_highlight(self, editor, delay=180):
        schedule_highlight(editor, delay)
    # ----------------------------------------------------------

    # canvas management

    # ----------------------------------------------------------

    def add_canvas(self, name=None):
        if name is None:
            i = 1
            while 'Canvas %d' % i in self.canvas_dict:
                i += 1
            name = 'Canvas %d' % i
        tab = ttk.Frame(self.canvas_nb)
        self.canvas_nb.add(tab, text=name)
        try:
            self.canvas_nb.select(tab)
        except tk.TclError:
            pass
        c = HeatmapCanvas(tab, pixel_callback=self._on_pixel)
        c.pack(fill=tk.BOTH, expand=True)
        self.canvas_dict[name] = c
        return c
    def create_canvas(self, name=None):
        return self.add_canvas(name)
    def delete_canvas(self):
        sel = self.canvas_nb.select()
        if not sel:
            return
        text = self.canvas_nb.tab(sel, 'text')
        if text in self.canvas_dict and len(self.canvas_dict) > 1:
            self.canvas_nb.forget(sel)
            del self.canvas_dict[text]
            self.status.config(text='Removed %s' % text)
    def rename_canvas(self):
        sel = self.canvas_nb.select()
        if not sel:
            return
        old = self.canvas_nb.tab(sel, 'text')
        new = simpledialog.askstring('Rename Canvas', 'New name:',
                                     initialvalue=old)
        if new and new != old and new not in self.canvas_dict:
            self.canvas_dict[new] = self.canvas_dict.pop(old)
            self.canvas_nb.tab(sel, text=new)
            self._update_canvas_name_bar()
    def get_canvas(self, name=None):
        if name is None:
            return self.get_current_canvas()
        return self.canvas_dict.get(name)
    def current_canvas_name(self):
        try:
            return self.canvas_nb.tab('current', 'text')
        except Exception:
            return None
    def _update_canvas_name_bar(self):
        """Refresh the docked canvas-name label (tab text may shrink)."""
        name = self.current_canvas_name()
        if name is None:
            self.canvas_name_bar.config(text='Canvas: --')
            return
        try:
            idx = self.canvas_nb.index('current') + 1
            total = len(self.canvas_nb.tabs())
        except tk.TclError:
            idx = total = 0
        self.canvas_name_bar.config(
            text='Canvas: %s   (%d/%d)' % (name, idx, total))
    def select_canvas(self, name):
        """Switch the visible canvas tab by name.  Returns bool."""
        c = self.canvas_dict.get(name)
        if c is None:
            return False
        try:
            self.canvas_nb.select(c.master)
        except tk.TclError:
            return False
        return True
    def get_current_canvas(self):
        sel = self.canvas_nb.select()
        if not sel:
            return None
        return self.canvas_dict[self.canvas_nb.tab(sel, 'text')]
    def _on_pixel(self, info):
        if info:
            val = ' val=%.3f' % info['value'] if info['value'] is not None else ''
            self.pixel_label.config(
                text='Pixel: (%d,%d) RGB=(%d,%d,%d)%s'
                     % (info['x'], info['y'], info['r'], info['g'], info['b'], val))
        else:
            self.pixel_label.config(text='Pixel: --')
    # ----------------------------------------------------------
    # canvas list / visibility (hiding keeps the data + API access)
    # ----------------------------------------------------------

    def canvas_names(self):
        """Names of all canvases (hidden ones included)."""
        return list(self.canvas_dict)
    def _unique_canvas_name(self, base):
        if base not in self.canvas_dict:
            return base
        i = 2
        while '%s %d' % (base, i) in self.canvas_dict:
            i += 1
        return '%s %d' % (base, i)
    def visible_canvas_names(self):
        return [n for n in self.canvas_dict if n not in self._hidden_canvases]
    def hide_canvas(self, name=None):
        """Hide a canvas tab; the canvas keeps existing and stays usable
        through the API.  Returns bool.  At least one tab stays visible.
        """
        if name is None:
            name = self.current_canvas_name()
        if name is None or name not in self.canvas_dict:
            return False
        if name in self._hidden_canvases:
            return False
        if len(self.visible_canvas_names()) <= 1:
            return False
        c = self.canvas_dict[name]
        try:
            self.canvas_nb.hide(c.master)
        except tk.TclError:
            return False
        self._hidden_canvases.add(name)
        self.status.config(text='Canvas hidden: %s' % name)
        return True
    def _show_recent_canvas(self):
        """Toolbar helper: restore one hidden canvas tab."""
        if not self._hidden_canvases:
            self.status.config(text='No hidden canvases')
            return False
        name = sorted(self._hidden_canvases)[0]
        ok = self.show_canvas(name)
        if ok:
            self.status.config(text='Canvas restored: %s' % name)
        return ok
    def show_canvas(self, name):
        """Restore a hidden canvas tab.  Returns bool."""
        if name not in self.canvas_dict or name not in self._hidden_canvases:
            return False
        c = self.canvas_dict[name]
        try:
            self.canvas_nb.add(c.master, text=name)
        except tk.TclError:
            return False
        self._hidden_canvases.discard(name)
        self.status.config(text='Canvas shown: %s' % name)
        return True
    def set_canvas_visible(self, name, visible):
        if visible:
            return self.show_canvas(name)
        return self.hide_canvas(name)
    def canvas_visibility_dialog(self):
        win = tk.Toplevel(self)
        win.title('Canvas Visibility')
        ttk.Label(win, text='Check a canvas to show it, uncheck to hide '
                            'it (data and API access are unaffected).'
                  ).pack(anchor='w', padx=12, pady=(10, 4))
        for name in self.canvas_names():
            var = tk.BooleanVar(value=name not in self._hidden_canvases)
            ttk.Checkbutton(
                win, text=name, variable=var,
                command=lambda n=name, v=var:
                self.set_canvas_visible(n, v.get())
            ).pack(anchor='w', padx=16, pady=1)
        ttk.Button(win, text='Close', command=win.destroy
                   ).pack(pady=(8, 8), side=tk.RIGHT, padx=12)
    # ----------------------------------------------------------
    # platform state / monitoring / performance
    # ----------------------------------------------------------

    def _tensor_info(self):
        """Compact description of the platform tensor (imported data)."""
        d = self.current_image_data
        if d is None:
            c = self.get_current_canvas()
            if c is not None and c.heatmap_data is not None:
                d = c.heatmap_data
        if d is None:
            return None
        try:
            return {'shape': (len(d), len(d[0]) if d and d[0] else 0),
                    'type': type(d).__name__}
        except Exception:
            return {'type': type(d).__name__}
    def get_widget_geometry(self, which='window'):
        """Live position/size of a monitored component.

        which: 'window' | 'status' | 'canvas_nb' | 'shell' | 'script'
        or a canvas name (e.g. 'Original').  Returns a dict or None.
        """
        widgets = {'window': self, 'status': self.status,
                   'canvas_nb': self.canvas_nb, 'shell': self.shell.text,
                   'script': self.script_editor}
        w = widgets.get(which)
        if w is None and which in self.canvas_dict:
            w = self.canvas_dict[which].canvas
        if w is None:
            return None
        try:
            return {'x': w.winfo_rootx(), 'y': w.winfo_rooty(),
                    'width': w.winfo_width(), 'height': w.winfo_height(),
                    'mapped': bool(w.winfo_ismapped())}
        except tk.TclError:
            return None
    def get_platform_state(self):
        """Full monitorable platform snapshot (dict; JSON-serialisable)."""
        try:
            vis = self.state() == 'normal'
        except Exception:
            vis = False
        state = {
            'version': __version__,
            'visible': vis,
            'window_geometry': self.get_widget_geometry('window'),
            'current_canvas': self.current_canvas_name(),
            'canvases': self.canvas_names(),
            'hidden_canvases': sorted(self._hidden_canvases),
            'canvas_geometry': {
                n: self.canvas_dict[n].get_geometry()
                for n in self.canvas_dict},
            'tensor': self._tensor_info(),
            'status_text': self.status.cget('text') if self.status else None,
            'shell': self.shell_metrics(),
            'remote': self.remote_info(),
            'metrics': dict(self._metrics),
            'handlers': {
                'loaders': sorted(IMAGE_LOADERS),
                'savers': sorted(IMAGE_SAVERS),
                'external': sorted(self.external_hooks),
            },
        }
        return state
    def shell_metrics(self):
        if self.shell is None:
            return {}
        return {'busy': self.shell.executing,
                'reading_input': self.shell.reading,
                'history_len': len(self.shell.history),
                'last_exec_ms': getattr(self.shell, 'exec_ms', 0.0)}
    # ----------------------------------------------------------
    # external hooks (injected remote / handler operations)
    # ----------------------------------------------------------

    def register_external(self, name, fn):
        """Register a named external operation (remote-callable).
        The callable may accept *args / **kwargs; injected via handler
        code, the --handlers file or external programs.
        """
        if not callable(fn):
            raise TypeError('external hook must be callable: %r' % (fn,))
        self.external_hooks[name] = fn
        return fn
    def unregister_external(self, name):
        return self.external_hooks.pop(name, None)
    def call_external(self, name, *args, **kwargs):
        fn = self.external_hooks.get(name)
        if fn is None:
            raise KeyError('no external operation registered: %r' % name)
        return fn(*args, **kwargs)
    # ----------------------------------------------------------
    # optional remote control (TCP JSON, localhost only, off by default)
    # ----------------------------------------------------------

    def start_remote(self, host=None, port=None):
        """Start the optional control server (returns the bound port).
        Safe to call again: an active server keeps its port.

        Flexible calling styles:
          start_remote()                    -> 127.0.0.1, OS-assigned port
          start_remote(8000)                -> old-style: 127.0.0.1:8000
          start_remote('0.0.0.0')           -> all IPv4 interfaces
          start_remote('::1', 8000)         -> explicit IPv6 loopback
          start_remote('')                  -> all interfaces (IPv6 dual)
          start_remote('127.0.0.1:8000')    -> host:port string
        """
        if host is None:
            host = '127.0.0.1'
        if isinstance(host, int) and port is None:
            host, port = '127.0.0.1', host
        if isinstance(host, str) and port is None:
            h = host
            if h.startswith('['):
                close = h.find(']')
                if close > 0:
                    inner = h[1:close]
                    rest = h[close + 1:]
                    host = inner
                    if rest.startswith(':'):
                        port = int(rest[1:])
            elif h.count(':') == 1 and h.rsplit(':', 1)[1].isdigit():
                host, port = h.rsplit(':', 1)[0], int(h.rsplit(':', 1)[1])
        if port is None:
            port = 0
        if self._remote is not None:
            return self._remote.port
        server = RemoteServer(self, port=port, host=host)
        server.start()
        self._remote = server
        self._remote_started_at = time.time()
        shown_host = server.host if server.host is not None else host
        self.status.config(text='Remote control on %s:%d'
                                % (shown_host, server.port))
        return server.port
    def stop_remote(self):
        """Stop the control server (if running)."""
        if self._remote is not None:
            self._remote.stop()
            self._remote = None
            self._remote_started_at = None
            self.status.config(text='Remote control stopped')
    def remote_info(self):
        if self._remote is None:
            return {'running': False, 'host': None, 'port': None}
        return {'running': True, 'host': self._remote.host,
                'port': self._remote.port}
    # ----------------------------------------------------------
    # diagnostics / docs windows + about
    # ----------------------------------------------------------

    def open_diagnostics(self):
        DiagnosticsWindow(self)
    def open_api_docs(self):
        ApiDocsWindow(self)
    def about_dialog(self):
        messagebox.showinfo(
            'About', 'Image Test Platform %s\n\n'
                     'Injected-code testing platform with IDLE-style '
                     'shell, script editor, unbounded canvases, pluggable '
                     'loaders/savers and optional TCP remote control.\n'
                     'Help > API Reference documents the full public API.'
                     % __version__)
    # ----------------------------------------------------------
    # canvas persistence: in-memory snapshot + JSON file state
    # ----------------------------------------------------------

    def snapshot_canvases(self):
        """In-memory canvas persistence: dict canvas_name -> rows."""
        snap = {}
        for name, c in self.canvas_dict.items():
            b = c.buffer.bounds()
            rows = c.get_data_full() if b is not None else []
            snap[name] = {'rows': rows,
                          'visible': name not in self._hidden_canvases}
        return snap
    def restore_canvases(self, snap):
        """Restore canvases from a snapshot_canvases() dict. Missing
        canvases are created; hidden state is preserved."""
        if not isinstance(snap, dict):
            raise ValueError('snapshot must be a dict')
        for name, data in snap.items():
            rows = data.get('rows') if isinstance(data, dict) else data
            if not rows:
                continue
            c = self.canvas_dict.get(name)
            if c is None:
                c = self.add_canvas(name)
            c.load_image(rows)
            if isinstance(data, dict) and not data.get('visible', True):
                self.hide_canvas(name)
        return True
    def save_platform_state(self, path):
        """Persist all canvases (+visibility) to a JSON file.
        Pixel data is stored as base64 PPM payloads (stdlib only)."""
        snap = self.snapshot_canvases()
        payload = {}
        for name, data in snap.items():
            rows = data['rows']
            h = len(rows)
            w = len(rows[0]) if h else 0
            ppm = _rows_to_ppm_bytes(rows) if w and h else b''
            payload[name] = {
                'ppm': base64.b64encode(ppm).decode('ascii'),
                'w': w, 'h': h,
                'visible': data.get('visible', True),
            }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'version': __version__, 'platform_state': True,
                       'saved': time.time(), 'canvases': payload,
                       'current': self.current_canvas_name()}, f,
                      indent=1)
        self.status.config(text='Platform state saved: %s' % path)
        return path
    def load_platform_state(self, path):
        """Restore canvases (+visibility) from save_platform_state() JSON.
        Creates missing canvases and reselects the saved current one."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        payload = data.get('canvases', {})
        if not payload:
            raise ValueError('not a platform state file: %r' % path)
        restore = {}
        for name, info in payload.items():
            ppm = base64.b64decode(info.get('ppm') or '')
            rows = _ppm_bytes_to_rows(ppm) if ppm else []
            restore[name] = {'rows': rows,
                             'visible': info.get('visible', True)}
        self.restore_canvases(restore)
        cur = data.get('current')
        if cur in self.canvas_dict:
            try:
                self.canvas_nb.select(self.canvas_dict[cur].master)
            except tk.TclError:
                pass
        self.status.config(text='Platform state loaded: %s' % path)
        return True
    # ----------------------------------------------------------

    # drawing

    # ----------------------------------------------------------

    def choose_color(self):
        rgb, color = colorchooser.askcolor(title='Pen Colour',
                                           initialcolor=self.pen_color_var.get())
        if color:
            self.pen_color_var.set(color)
            self.color_btn.config(bg=color)
            self.apply_pen()
    def apply_pen(self):
        for c in self.canvas_dict.values():
            c.set_pen_color(self.pen_color_var.get())
            c.set_pen_size(self.pen_size_var.get())
    def toggle_eraser(self):
        for c in self.canvas_dict.values():
            c.set_pen_erasing(not c.pen_erasing)
    def clear_drawings(self):
        for c in self.canvas_dict.values():
            c.clear_drawings()
        self.status.config(text='Drawings cleared')
    # ----------------------------------------------------------

    # image IO

    # ----------------------------------------------------------

    def open_image(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(
                filetypes=[('Images',
                            '*.png *.gif *.ppm *.pgm *.pbm *.rrgb')])
        if not path:
            return False
        try:
            w, h, rows = load_image_path(path, app=self,
                                         canvas=self.get_current_canvas())
        except Exception as e:
            messagebox.showerror('Load Error', str(e))
            return False
        self.load_image_rows(rows, path)
        return True
    def open_images(self, paths=None):
        """Batch open: multi-select images, each into its own canvas.
        Returns list of bools / names imported."""
        if paths is None:
            paths = filedialog.askopenfilenames(
                filetypes=[('Images',
                            '*.png *.gif *.ppm *.pgm *.pbm *.rrgb'),
                           ('All files', '*.*')])
        if not paths:
            return []
        if isinstance(paths, str):
            paths = [paths]
        results = []
        for path in paths:
            try:
                w, h, rows = load_image_path(path, app=self)
            except Exception as e:
                results.append(False)
                continue
            name = self._unique_canvas_name(os.path.splitext(
                os.path.basename(path))[0])
            c = self.add_canvas(name)
            if c is not None:
                self.load_image_rows(rows, path, canvas=c)
                results.append(True)
        self.status.config(text='Opened %d image(s)' % sum(results))
        return results
    def load_image(self, path=None):
        return self.open_image(path)
    def load_image_rows(self, rows, path=None, canvas=None):
        c = canvas or self.get_current_canvas()
        if c is None:
            return False
        ok = c.load_image(rows)
        if ok:
            self.current_image_data = rows_to_data(rows)
            label = os.path.basename(path) if path else 'rows'
            self.image_info.config(
                text='Image: %s (%dx%d)' % (label, c.canvas_width, c.canvas_height))
            self.status.config(text='Image loaded (%dx%d)' % (c.canvas_width, c.canvas_height))
        return bool(ok)
    def get_canvas_data(self, canvas=None):
        c = canvas or self.get_current_canvas()
        if c is None:
            raise ValueError('no canvas')
        return c.get_data()
    # ----- data import (matrix) -----

    def open_handler_settings(self):
        HandlerSettingsDialog(self, self)
    def import_data_dialog(self):
        paths = filedialog.askopenfilenames(
            filetypes=[('Data', '*.csv *.txt *.npy.txt'),
                       ('All files', '*.*')])
        if not paths:
            return
        self.import_files(paths, mode='matrix')
    def import_files(self, paths, mode='image'):
        """Batch import entry point shared by menu / API.
        mode: 'image' -> each file becomes a canvas image;
        'matrix' -> each numeric grid becomes a canvas heatmap.
        Returns list of {path, ok, name, width, height} or {path, ok, error}."""
        if isinstance(paths, str):
            paths = [paths]
        results = []
        for path in paths:
            entry = {'path': path}
            try:
                if mode == 'matrix':
                    rows = self._parse_matrix_text(path)
                    if rows is None:
                        entry.update(ok=False, error='empty header')
                        results.append(entry)
                        continue
                    name = self._unique_canvas_name(os.path.splitext(
                        os.path.basename(path))[0])
                    c = self.add_canvas(name)
                    c.display_heatmap(rows)
                    entry.update(ok=True, name=name, width=len(rows[0]),
                                 height=len(rows))
                else:
                    w, h, rows = load_image_path(path, app=self)
                    name = self._unique_canvas_name(os.path.splitext(
                        os.path.basename(path))[0])
                    c = self.add_canvas(name)
                    self.load_image_rows(rows, path, canvas=c)
                    entry.update(ok=True, name=name, width=c.canvas_width,
                                 height=c.canvas_height)
            except Exception as e:
                entry.update(ok=False, error=str(e))
            results.append(entry)
        ok = sum(1 for r in results if r.get('ok'))
        if mode == 'matrix':
            self.status.config(text='Imported %d/%d matrix file(s)'
                               % (ok, len(results)))
        else:
            self.status.config(text='Imported %d/%d image file(s)'
                               % (ok, len(results)))
        return results
    def _parse_matrix_text(self, path):
        """Parse CSV / whitespace / semicolon text into numeric 2D rows."""
        import csv
        rows = []
        with open(path, newline='', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [p.strip() for p in line.replace(';', ',').split(',') if p.strip()]
                if not parts:
                    continue
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    continue
        return rows
    # ----------------------------------------------------------

    # save / export (via IMAGE_SAVERS registry)

    # ----------------------------------------------------------

    def save_canvas_dialog(self):
        c = self.get_current_canvas()
        if c is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.ppm',
            filetypes=[('PPM (lossless)', '*.ppm'),
                       ('PGM (grey)', '*.pgm'),
                       ('PostScript', '*.ps')])
        if not path:
            return
        try:
            save_image_path(path, c.get_rows(), self)
            self.status.config(text='Saved %s' % path)
        except Exception as e:
            messagebox.showerror('Save Error', str(e))
    def save_ppm_dialog(self):
        path = filedialog.asksaveasfilename(defaultextension='.ppm',
                                            filetypes=[('PPM', '*.ppm')])
        if not path:
            return
        rows = self.current_image_data if self.current_image_data else             (self.get_current_canvas().get_rows() if self.get_current_canvas() else None)
        if rows is None:
            self.status.config(text='Nothing to save')
            return
        save_image_path(path, rows, self)
    def save_image_dialog(self):
        self.save_canvas_dialog()
    def save_all_dialog(self):
        directory = filedialog.askdirectory(title='Export All Canvases To...')
        if not directory:
            return
        written = self.export_files(directory)
        self.status.config(text='Exported %d file(s) to %s'
                           % (len(written), directory))
    def export_files(self, directory, fmt='ppm'):
        """Export every canvas into *directory* as *fmt* (an extension
        with a registered saver, default 'ppm').  Returns the list of
        written paths."""
        fmt = (fmt or 'ppm').lstrip('.').lower()
        if fmt not in IMAGE_SAVERS:
            fmt = 'ppm'
        import os
        os.makedirs(directory, exist_ok=True)
        written = []
        for name, c in list(self.canvas_dict.items()):
            rows = c.get_rows()
            if not rows:
                continue
            safe = ''.join(ch for ch in name
                           if ch.isalnum() or ch in '-_ ()')
            path = os.path.join(directory, '%s.%s' % (safe, fmt))
            try:
                save_image_path(path, rows, self)
                written.append(path)
            except Exception as e:
                self.status.config(text='Export failed %s: %s' % (name, e))
        return written
    # ----------------------------------------------------------

    # test execution

    # ----------------------------------------------------------

    def run_tests(self):
        failures = 0
        for name, fn in list(self.test_registry.items()):
            try:
                r = fn(self.current_image_data)
                self._last_result = r
            except Exception as e:
                failures += 1
                self.status.config(text='Test %s FAILED: %s' % (name, e))
        if failures:
            self.status.config(text='%d/%d tests failed' % (failures, len(self.test_registry)))
            return False
        self.status.config(text='All %d tests OK' % len(self.test_registry))
        return True
    def _add_test_stub(self, name, fn):
        self.test_registry[name] = fn
        self.status.config(text='Test registered: %s' % name)
    def _api_get_pixel(self, x, y, canvas=None):
        c = canvas or self.get_current_canvas()
        if c is None:
            raise ValueError('no canvas')
        return c.get_pixel_rgb(x, y)
    def _api_set_pixel(self, x, y, color, canvas=None):
        c = canvas or self.get_current_canvas()
        if c is None:
            raise ValueError('no canvas')
        return c.set_pixel(x, y, color)
    def _api_display_heatmap(self, values, canvas=None):
        c = canvas or self.get_current_canvas()
        if c is None:
            raise ValueError('no canvas')
        return c.display_heatmap(values)
    # ----------------------------------------------------------

    # synthesis from current canvas (kept compatible params)

    # ----------------------------------------------------------

    def synthesize_from_current_canvas(self):
        c = self.get_current_canvas()
        if c is None:
            return
        w = self.width_var.get()
        h = self.height_var.get()
        x = self.start_x_var.get()
        y = self.start_y_var.get()
        rows = c.get_data(x, y, w, h)
        self.current_image_data = rows_to_data(rows)
        self._last_result_rows = rows
        self.image_info.config(text='Synthesized %dx%d region at (%d,%d)' % (w, h, x, y))
        self.status.config(text='Synthesized region @ (%d,%d) %dx%d' % (x, y, w, h))
def rows_to_data(rows):
    """Normalise / materialise to list-of-rows (each (r,g,b))."""
    return [list(row) for row in rows]
def _rows_to_ppm_bytes(rows):
    """Render (r,g,b) rows as binary P6 PPM bytes (no dependency)."""
    h = len(rows)
    w = len(rows[0]) if h else 0
    out = io.BytesIO()
    out.write(('P6\n%d %d\n255\n' % (w, h)).encode('ascii'))
    for row in rows:
        out.write(bytes(v for px in row for v in px))
    return out.getvalue()
def _ppm_bytes_to_rows(ppm):
    """Parse binary P6 PPM bytes back into (r,g,b) rows."""
    f = io.BytesIO(ppm)
    if f.readline().strip() != b'P6':
        raise ValueError('state payload: expected P6 PPM')
    line = f.readline().strip()
    while line.startswith(b'#'):
        line = f.readline().strip()
    parts = line.split()
    if len(parts) < 2:
        raise ValueError('state payload: corrupt PPM header')
    w, h = int(parts[0]), int(parts[1])
    f.readline()                 # maxval
    raw = f.read()
    rows = []
    i = 0
    for y in range(h):
        row = []
        for x in range(w):
            row.append((raw[i], raw[i + 1], raw[i + 2]))
            i += 3
        rows.append(row)
    return rows
def _fmt_signature(fn):
    """'(x=0, y=0, canvas=None)' - compact signature for docs."""
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return '(...)'
    params = []
    for p in sig.parameters.values():
        if p.name == 'self':
            continue
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            params.append('**%s' % p.name)
            continue
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            params.append('*%s' % p.name)
            continue
        if p.default is inspect.Parameter.empty:
            params.append(p.name)
        else:
            try:
                params.append('%s=%r' % (p.name, p.default))
            except Exception:
                params.append(p.name)
    return '(%s)' % ', '.join(params)
def api_reference_text():
    """Full human-readable API reference (Help > API Reference)."""
    L = []
    add = L.append
    add('IMAGE TEST PLATFORM - API REFERENCE (version %s)' % __version__)
    add('=' * 60)
    add('')
    add('External (module) use')
    add('-' * 40)
    add('    import test_image_gui as tp')
    add('    p = tp.create_platform()          # hidden window')
    add('    p.show()                          # deiconify + pump')
    add('    p.set_pixel(10, 10, "red")')
    add('    p.get_pixel(10, 10)               # -> (255, 0, 0)')
    add('    p.set_tensor([[0, 0.5], [1, 0]])  # import matrix')
    add('    p.tensor_shape()                  # -> (2, 2)')
    add('    p.export_canvas("Original", "o.ppm")')
    add('    snap = p.snapshot(); p.restore(snap)')
    add('    p.hide(); p.close()')
    add('')
    add('    app = tp.PlatformApp()            # direct, windowed')
    add('    app.open_api_docs()               # GUI API browser')
    add('')
    add('CompanionPlatform methods (p.*):')
    add('-' * 40)
    for name in ('show hide close pump is_visible canvas load_rows '
                 'get_rows get_pixel set_pixel canvas_names state '
                 'geometry metrics start_remote stop_remote export_canvas '
                 'import_canvas snapshot restore save_state load_state '
                 'exec_code register_external call_external '
                 'get_canvas_data display_heatmap clear_canvas '
                 'add_canvas delete_canvas rename_canvas '
                 'hide_canvas show_canvas set_tensor get_tensor '
                 'tensor_shape tensor_value is_busy').split():
        fn = getattr(CompanionPlatform, name, None)
        if fn is None:
            continue
        doc = inspect.getdoc(fn) or ''
        first, _, _ = doc.partition('\n')
        add('    %s%s  %s' % (name, _fmt_signature(fn), first))
    add('')
    add('Injected namespace (Script / Shell / remote "exec")')
    add('-' * 40)
    for name in ('get_canvas create_canvas delete_canvas rename_canvas '
                 'canvas_names hide_canvas show_canvas export_canvas '
                 'import_canvas load_image load_image_rows get_canvas_data '
                 'get_pixel set_pixel display_heatmap get_tensor set_tensor '
                 'tensor_shape platform_state run_tests add_test '
                 'register_external call_external start_remote stop_remote '
                 'help_api').split():
        note = ''
        if name == 'export_canvas':
            note = '   (path, canvas=None, canvas_name=None)'
        elif name == 'import_canvas':
            note = '   (path, canvas=None, canvas_name=None)'
        elif name == 'get_tensor':
            note = '   (canvas=None, canvas_name=None)'
        elif name == 'set_tensor':
            note = '   (rows, canvas=None, canvas_name=None)'
        elif name == 'platform_state':
            note = '   ()'
        elif name == 'register_external':
            note = '   (name, fn)'
        elif name == 'call_external':
            note = '   (name, *args, **kwargs)'
        elif name == 'start_remote':
            note = '   (host="127.0.0.1", port=0) -> bound port'
        elif name == 'stop_remote':
            note = '   ()'
        elif name == 'help_api':
            note = '   (topic=None) e.g. help_api("set_pixel")'
        add('    %s%s' % (name, note))
    add('    app / api / current_canvas / canvases / tk / ttk / '
        'messagebox')
    add('')
    add('Short names (C-API style, also injected at top level)')
    add('-' * 40)
    add('    px / set_px / blit / rows / heatmap / canvas / new_canvas')
    add('    drop_canvas / rename / hide / show / names / load / export')
    add('    tensor / put_tensor / shape / listen / unlisten / rinfo')
    add('    version')
    add('    e.g.  px(3, 3)              -> get_pixel(3, 3)')
    add('          blit(grid, 0, 0)      -> set_rows(grid, 0, 0)')
    add('          listen("::1")         -> start_remote("::1")')
    add('')
    add('ApiFacade methods (api.*, canvas-targetable)')
    add('-' * 40)
    for name in sorted(n for n in dir(ApiFacade)
                       if not n.startswith('_')):
        fn = getattr(ApiFacade, name)
        doc = inspect.getdoc(fn) or ''
        first, _, _ = doc.partition('\n')
        add('    api.%s%s  %s' % (name, _fmt_signature(fn), first))
    add('')
    add('Canvases')
    add('-' * 40)
    for name in ('get_data get_rows get_pixel_rgb set_pixel '
                 'get_view get_region_rows get_world_bounds '
                 'get_data_full get_heatmap_value load_image '
                 'display_heatmap clear_drawings clear_all '
                 'save_ps get_photoimage pixel_callback get_buffer '
                 'get_geometry').split():
        fn = getattr(HeatmapCanvas, name, None)
        if fn is None:
            continue
        doc = inspect.getdoc(fn) or ''
        first, _, _ = doc.partition('\n')
        add('    %s%s  %s' % (name, _fmt_signature(fn), first))
    add('')
    add('Canvas-targetable convention: set_pixel / get_pixel / '
        'get_canvas_data /')
    add('display_heatmap / clear_canvas accept optional canvas= or '
        'canvas_name=;')
    add('the default is the current canvas.')
    add('')
    add('Tensor API')
    add('-' * 40)
    add('  get_tensor(canvas=, canvas_name=)  imported matrix, else the')
    add('        canvas heatmap grid, else its pixel rows (or None)')
    add('  set_tensor(rows, canvas=, ...)     import matrix -> storage +')
    add('        heatmap display; returns the cleaned rows')
    add('  tensor_shape(...) / tensor_value(x, y, ...)')
    add('')
    add('Monitoring')
    add('-' * 40)
    add('  platform_state() / api.state()     full JSON-serialisable')
    add('        snapshot: visibility, canvases, canvas_geometry'
        ' (position,')
    add('        size, bounds, tiles), tensor info, shell busy state,')
    add('        remote info, exec metrics, handler registries')
    add('  api.geometry(which)                live position/size of')
    add('        "window" | "status" | "canvas_nb" | "shell" | "script"')
    add('        or a canvas name')
    add('  api.metrics()                      exec count/ms total/last,')
    add('        per-canvas draw + photo-sync counters')
    add('  Help > Diagnostics                  live monitor window')
    add('')
    add('Remote control (off by default)')
    add('-' * 40)
    add('  Enable it via app.start_remote(port), the --remote CLI switch,')
    add('  or handler code.  Binds 127.0.0.1; protocol = newline-JSON:')
    add('')
    add('    {"cmd": "ping"}')
    add('    {"cmd": "state"}')
    add('    {"cmd": "exec", "code": "set_pixel(0, 0, \'red\')"}')
    add('    {"cmd": "call", "name": "set_pixel",')
    add('     "args": [1, 2, "green"], "kwargs": {}}')
    add('    {"cmd": "call", "name": "<external hook>", "args": [...]}')
    add('    {"cmd": "hooks"}')
    add('')
    add('  Replies: {"ok": true, "result": ...} or')
    add('           {"ok": false, "error": "..."}  (one JSON line).')
    add('  "call" accepts any ApiFacade.ALLOWED method name plus every')
    add('  registered external hook (register_external, or api.register).')
    add('')
    add('Handlers (load / save / remote logic)')
    add('-' * 40)
    add('  The Handlers window (Canvas workspace > Handlers) edits three')
    add('  categories: Load logic (register_image_loader), Save logic')
    add('  (register_image_saver) and Remote logic (register_external /')
    add('  start_remote).  Each tab has its own Import / Export buttons;')
    add('  --handlers FILE runs a file in the handler namespace at start.')
    add('')
    add('Performance')
    add('-' * 40)
    add('  Every exec reports wall time in the status bar and in the')
    add('  result dict (key "ms").  Canvas metrics count draw calls and')
    add('  PhotoImage sync time; tile_count() introspects the buffer.')
    return '\n'.join(L)
# ============================================================

#  ApiFacade - generic, canvas-targetable public API

# ============================================================

class ApiFacade(object):
    """Extensible public API facade exposed as ``api``.  Every canvas
    operation accepts either a canvas object (``canvas=``) or a canvas
    name (``canvas_name=``); with neither, the current canvas is used.

    A small dict-style registry (``api['key'] = value``) holds
    user-defined names, and the whitelisted method set in ``ALLOWED``
    is reachable through the optional remote control server.
    """

    #: Operation names exposed to the remote control protocol.
    ALLOWED = frozenset([
        'get_canvas_data', 'get_rows', 'set_rows', 'fill', 'list',
        'get_pixel', 'set_pixel', 'display_heatmap',
        'clear_canvas', 'canvases', 'canvas_count', 'select_canvas',
        'current_name', 'add_canvas', 'delete_canvas',
        'rename_canvas', 'hide_canvas', 'show_canvas', 'export_canvas',
        'import_canvas', 'import_files', 'export_all',
        'get_tensor', 'set_tensor', 'tensor_shape',
        'tensor_value', 'state', 'geometry', 'metrics', 'shell_metrics',
        'register', 'call', 'unregister', 'hooks', 'start_remote',
        'stop_remote', 'remote_info', 'run_script', 'run_tests',
        'snapshot', 'restore', 'save_state', 'load_state', 'current',
        'is_visible', 'is_busy',
        'px', 'set_px', 'blit', 'rows', 'heatmap', 'canvas', 'new_canvas',
        'select', 'drop_canvas', 'rename', 'hide', 'show', 'names', 'load',
        'export', 'tensor', 'put_tensor', 'shape', 'listen', 'unlisten',
        'rinfo', 'version',
    ])
    def __init__(self, app):
        self.app = app
        self.registry = {}
    # ---------- canvas resolution (robust, named or object) ----------

    def _resolve_canvas(self, canvas=None, canvas_name=None, create=False):
        if canvas is not None:
            return canvas
        if canvas_name is not None:
            c = self.app.canvas_dict.get(canvas_name)
            if c is None and create:
                c = self.app.add_canvas(canvas_name)
            return c
        return self.app.get_current_canvas()
    def _require_canvas(self, canvas=None, canvas_name=None, create=False):
        c = self._resolve_canvas(canvas=canvas, canvas_name=canvas_name,
                                 create=create)
        if c is None:
            raise ValueError('no canvas (create one with add_canvas)')
        return c
    # ---------- registry (dict-style compatibility) ----------

    def __getitem__(self, name):
        return self.registry[name]
    def __setitem__(self, name, value):
        self.registry[name] = value
    def __contains__(self, name):
        return name in self.registry
    def keys(self):
        return self.registry.keys()
    # ---------- canvas operations (canvas-targetable) ----------

    def get_canvas(self, canvas_name=None):
        """Canvas object by name, or the current canvas when omitted."""
        return self.app.get_canvas(canvas_name)
    def current(self):
        """Current canvas object (or None)."""
        return self.app.get_current_canvas()
    def canvases(self):
        """List of all canvas names (hidden ones included)."""
        return self.app.canvas_names()
    def canvas_count(self):
        """Number of canvases (hidden ones included)."""
        return len(self.app.canvas_dict)
    def select_canvas(self, canvas_name):
        """Switch the current (visible) canvas by name.  Returns bool."""
        return self.app.select_canvas(canvas_name)
    def current_name(self):
        """Name of the current canvas (or None)."""
        return self.app.current_canvas_name()
    def add_canvas(self, name=None):
        """Create a canvas tab; returns the HeatmapCanvas object."""
        return self.app.add_canvas(name)
    def delete_canvas(self, canvas_name=None):
        """Delete a canvas (defaults to the current one). Returns bool."""
        if canvas_name is None:
            canvas_name = self.app.current_canvas_name()
        if canvas_name is None or canvas_name not in self.app.canvas_dict:
            return False
        if len(self.app.canvas_dict) <= 1:
            return False
        c = self.app.canvas_dict[canvas_name]
        try:
            self.app.canvas_nb.select(c.master)
        except tk.TclError:
            pass
        self.app.delete_canvas()
        return True
    def rename_canvas(self, old, new):
        """Rename a canvas. Returns bool."""
        if (not new or old not in self.app.canvas_dict or
                new in self.app.canvas_dict):
            return False
        c = self.app.canvas_dict.pop(old)
        self.app.canvas_dict[new] = c
        try:
            self.app.canvas_nb.tab(c.master, text=new)
        except tk.TclError:
            return False
        self.app._update_canvas_name_bar()
        return True
    def hide_canvas(self, canvas_name=None):
        """Hide a canvas tab (data + API use keep working)."""
        return self.app.hide_canvas(canvas_name)
    def show_canvas(self, canvas_name=None):
        if canvas_name is None:
            return False
        return self.app.show_canvas(canvas_name)
    def get_canvas_data(self, x=0, y=0, w=None, h=None,
                        canvas=None, canvas_name=None):
        """Rows [(r,g,b)...] from a canvas region (default: whole)."""
        c = self._require_canvas(canvas=canvas, canvas_name=canvas_name)
        return c.get_data(x, y, w, h)
    def get_pixel(self, x, y, canvas=None, canvas_name=None):
        """RGB tuple of one pixel."""
        c = self._require_canvas(canvas=canvas, canvas_name=canvas_name)
        return c.get_pixel_rgb(x, y)
    def set_pixel(self, x, y, color, canvas=None, canvas_name=None):
        """Paint one pixel; returns bool."""
        c = self._require_canvas(canvas=canvas, canvas_name=canvas_name)
        return c.set_pixel(x, y, color)
    def display_heatmap(self, values, canvas=None, canvas_name=None):
        """Show a numeric grid as a heatmap overlay; returns bool."""
        c = self._require_canvas(canvas=canvas, canvas_name=canvas_name)
        return c.display_heatmap(values)
    def clear_canvas(self, canvas=None, canvas_name=None):
        """Wipe drawings + heatmap of a canvas."""
        c = self._require_canvas(canvas=canvas, canvas_name=canvas_name)
        c.clear_all()
        return True
    def get_rows(self, x=0, y=0, w=None, h=None, canvas=None,
                 canvas_name=None):
        """Alias of get_canvas_data(...)."""
        return self.get_canvas_data(x, y, w, h, canvas=canvas,
                                    canvas_name=canvas_name)
    def set_rows(self, rows, x=0, y=0, canvas=None, canvas_name=None):
        """Blit a pixel grid (rows of (r,g,b)) onto a canvas at (x, y).
        Returns bool.  Robust fallback: draws pixel by pixel when the
        buffer has no bulk blit support."""
        c = self._require_canvas(canvas=canvas, canvas_name=canvas_name)
        try:
            c.buffer.set_rows(rows, int(x), int(y))
            c._sync_photo(full=True)
            return True
        except AttributeError:
            for yy, row in enumerate(rows):
                for xx, rgb in enumerate(row):
                    c.set_pixel(int(x) + xx, int(y) + yy, rgb)
            return True
        except Exception:
            return False
    def fill(self, color, canvas=None, canvas_name=None):
        """Fill a whole canvas with one colour."""
        c = self._require_canvas(canvas=canvas, canvas_name=canvas_name)
        try:
            c.buffer.fill(color)
            c._sync_photo(full=True)
            return True
        except Exception:
            return False
    def list(self, kind='api'):
        """Introspection helper: kind = 'api' (facade methods) |
        'registry' (user names) | 'canvases'."""
        if kind == 'api':
            return sorted(n for n in dir(self) if not n.startswith('_')
                          and callable(getattr(self, n, None)))
        if kind == 'registry':
            return sorted(self.registry)
        if kind == 'canvases':
            return self.app.canvas_names()
        raise ValueError('unknown list kind: %r' % kind)
    # ---------- short names (Python C-API style: verb + object) ----------
    # Every alias is a plain forwarder to the long descriptive name, so
    # both styles stay interchangeable everywhere (script, shell, remote).

    def px(self, x, y, canvas=None, canvas_name=None):
        """Short name of get_pixel."""
        return self.get_pixel(x, y, canvas=canvas, canvas_name=canvas_name)
    def set_px(self, x, y, color, canvas=None, canvas_name=None):
        """Short name of set_pixel."""
        return self.set_pixel(x, y, color, canvas=canvas,
                              canvas_name=canvas_name)
    def blit(self, rows, x=0, y=0, canvas=None, canvas_name=None):
        """Short name of set_rows (bulk pixel blit)."""
        return self.set_rows(rows, x, y, canvas=canvas,
                             canvas_name=canvas_name)
    def rows(self, x=0, y=0, w=None, h=None, canvas=None, canvas_name=None):
        """Short name of get_rows / get_canvas_data."""
        return self.get_rows(x, y, w, h, canvas=canvas,
                             canvas_name=canvas_name)
    def heatmap(self, values, canvas=None, canvas_name=None):
        """Short name of display_heatmap."""
        return self.display_heatmap(values, canvas=canvas,
                                    canvas_name=canvas_name)
    def canvas(self, name=None):
        """Short name of get_canvas (current canvas when omitted)."""
        return self.get_canvas(name)
    def new_canvas(self, name=None):
        """Short name of add_canvas."""
        return self.add_canvas(name)
    def select(self, name):
        """Short name of select_canvas (switch the visible canvas)."""
        return self.select_canvas(name)
    def drop_canvas(self, canvas_name=None):
        """Short name of delete_canvas (optional name)."""
        return self.delete_canvas(canvas_name)
    def rename(self, new_name):
        """Rename the current canvas (short name of rename_canvas)."""
        old = self.app.current_canvas_name()
        if new_name and old and new_name not in self.app.canvas_dict:
            self.app.canvas_dict[new_name] = self.app.canvas_dict.pop(old)
            sel = self.app.canvas_nb.select()
            self.app.canvas_nb.tab(sel, text=new_name)
            self.app._update_canvas_name_bar()
            return True
        return False
    def hide(self, name=None):
        """Short name of hide_canvas."""
        return self.hide_canvas(name)
    def show(self, name=None):
        """Short name of show_canvas."""
        return self.show_canvas(name)
    def names(self):
        """Short name of canvases() (list of canvas names)."""
        return self.app.canvas_names()
    def load(self, path=None, canvas=None, canvas_name=None):
        """Short name of open_image; optional target canvas."""
        c = canvas
        if c is None and canvas_name is not None:
            c = self.app.get_canvas(canvas_name)
        if c is None:
            return self.app.open_image(path)
        try:
            w, h, rows = load_image_path(path, app=self.app, canvas=c)
        except Exception:
            return False
        return self.app.load_image_rows(rows, path, canvas=c)
    def export(self, path, canvas=None, canvas_name=None):
        """Short name of export_canvas."""
        return self.export_canvas(path, canvas=canvas,
                                  canvas_name=canvas_name)
    def tensor(self, canvas=None, canvas_name=None):
        """Short name of get_tensor."""
        return self.get_tensor(canvas=canvas, canvas_name=canvas_name)
    def put_tensor(self, tensor, canvas=None, canvas_name=None):
        """Short name of set_tensor."""
        return self.set_tensor(tensor, canvas=canvas,
                               canvas_name=canvas_name)
    def shape(self):
        """Short name of tensor_shape."""
        return self.tensor_shape()
    def listen(self, host=None, port=None):
        """Short name of start_remote (control server)."""
        return self.start_remote(host, port)
    def unlisten(self):
        """Short name of stop_remote."""
        return self.stop_remote()
    def rinfo(self):
        """Short name of remote_info."""
        return self.remote_info()
    def version(self):
        """Platform library version string."""
        return __version__
    # ---------- multi-file batch import / export ----------

    def import_files(self, paths, mode='image'):
        """Batch import: mode 'image' (png/gif/ppm/...) or 'matrix'
        (numeric text / csv).  Each file lands on its own canvas named
        after the file.  Returns a list of result dicts with keys
        path / ok / name / width / height (or error)."""
        return self.app.import_files(paths, mode=mode)
    def export_all(self, directory, fmt='ppm'):
        """Export every canvas into *directory* as *fmt*.  Returns the
        list of written paths."""
        return self.app.export_files(directory, fmt=fmt)
    # ---------- canvas persistence (file + in-memory snapshot) ----------

    def export_canvas(self, path, canvas=None, canvas_name=None):
        """Save one canvas to an image file (suffix picks the saver:
        .ppm / .pgm / .ps / custom registered savers). Returns path."""
        if not path:
            raise ValueError('path is required')
        c = self._require_canvas(canvas=canvas, canvas_name=canvas_name)
        save_image_path(path, c.get_rows(), self.app)
        return path
    def import_canvas(self, path, canvas=None, canvas_name=None):
        """Load an image file into a canvas (created when named missing).
        Returns (width, height)."""
        if not path:
            raise ValueError('path is required')
        c = self._require_canvas(canvas=canvas, canvas_name=canvas_name,
                                 create=True)
        w, h, rows = load_image_path(path, app=self.app, canvas=c)
        self.app.load_image_rows(rows, path, canvas=c)
        return (w, h)
    def snapshot(self):
        """In-memory canvas persistence: dict name -> pixel rows."""
        return self.app.snapshot_canvases()
    def restore(self, snap):
        """Restore canvases from a snapshot() dict (creates missing)."""
        return self.app.restore_canvases(snap)
    def save_state(self, path):
        """Persist all canvases (+ visibility) to a JSON file."""
        return self.app.save_platform_state(path)
    def load_state(self, path):
        """Restore canvases (+ visibility) from a save_state() file."""
        return self.app.load_platform_state(path)
    # ---------- tensor access (platform-imported data) ----------

    def get_tensor(self, canvas=None, canvas_name=None):
        """The platform tensor: imported matrix first, then the named
        canvas' heatmap grid, then its pixel rows as a fallback.
        Returns nested lists or None when nothing is available."""
        if self.app.current_image_data is not None:
            return self.app.current_image_data
        c = self._resolve_canvas(canvas=canvas, canvas_name=canvas_name)
        if c is not None and c.heatmap_data is not None:
            return c.heatmap_data
        if c is not None:
            try:
                b = c.buffer.bounds()
                if b is not None:
                    return c.get_data_full()
            except Exception:
                return None
        return None
    def set_tensor(self, rows, canvas=None, canvas_name=None):
        """Import a numeric matrix as the platform tensor: stored in
        current_image_data and shown as a heatmap on the canvas.
        Returns cleaned rows, or None on ragged/invalid input."""
        cleaned = []
        for row in rows:
            try:
                cleaned.append([float(v) for v in row])
            except Exception:
                return None
        if not cleaned or not cleaned[0]:
            return None
        self.app.current_image_data = cleaned
        c = self._resolve_canvas(canvas=canvas, canvas_name=canvas_name,
                                 create=(canvas_name is not None))
        if c is not None:
            try:
                c.display_heatmap(cleaned)
            except Exception:
                pass
        return cleaned
    def tensor_shape(self, canvas=None, canvas_name=None):
        """(rows, cols) of the platform tensor, or None."""
        t = self.get_tensor(canvas=canvas, canvas_name=canvas_name)
        if t is None:
            return None
        try:
            return (len(t), len(t[0]) if t and t[0] else 0)
        except Exception:
            return None
    def tensor_value(self, x, y, canvas=None, canvas_name=None):
        """Scalar value at (x, y) of the platform tensor, or None."""
        t = self.get_tensor(canvas=canvas, canvas_name=canvas_name)
        if t is None:
            return None
        try:
            if 0 <= y < len(t) and 0 <= x < len(t[y]):
                return t[y][x]
        except Exception:
            return None
        return None
    # ---------- monitoring / state ----------

    def state(self):
        """Full monitorable platform snapshot (dict)."""
        return self.app.get_platform_state()
    def geometry(self, which='window'):
        """Live position/size of a component (window, canvas_nb, shell,
        status, script or a canvas name)."""
        return self.app.get_widget_geometry(which)
    def metrics(self):
        """Performance + activity metrics (dict)."""
        m = dict(self.app._metrics)
        m['shell'] = self.app.shell_metrics()
        m['canvas'] = {n: dict(c.metrics) for n, c in
                       self.app.canvas_dict.items()}
        return m
    def shell_metrics(self):
        return self.app.shell_metrics()
    def is_visible(self):
        try:
            return self.app.state() == 'normal'
        except Exception:
            return False
    def is_busy(self):
        """True while injected code runs or input() is pending."""
        s = self.app.shell
        return s is not None and (s.executing or s.reading)
    # ---------- execution / lifecycle ----------

    def exec(self, code):
        """Run code in the shared namespace; dict ok/stdout/stderr/ms."""
        return self.app.shell.exec_code(code)
    def run_script(self):
        """Execute the Script tab content."""
        return self.app.run_script()
    def run_tests(self):
        """Run all registered tests against the platform tensor."""
        return self.app.run_tests()
    def show(self):
        try:
            self.app.deiconify()
            self.app.update_idletasks()
        except tk.TclError:
            pass
    def hide(self):
        try:
            self.app.withdraw()
        except tk.TclError:
            pass
    def quit(self):
        try:
            self.app.destroy()
        except tk.TclError:
            pass
    def mainloop(self):
        self.app.mainloop()
    # ---------- external hooks (remote-reachable operations) ----------

    def register(self, name, fn):
        """Register a named external operation (remote-callable)."""
        return self.app.register_external(name, fn)
    def unregister(self, name):
        """Remove a named external operation."""
        return self.app.unregister_external(name)
    def call(self, name, *args, **kwargs):
        """Invoke a registered external operation by name."""
        return self.app.call_external(name, *args, **kwargs)
    def hooks(self):
        """Names of all registered external operations."""
        return sorted(self.app.external_hooks)
    # ---------- remote ----------

    def start_remote(self, host=None, port=None):
        """Open the TCP control server; returns the bound port.
        Flexible: start_remote(), start_remote(port), start_remote(host),
        start_remote(host, port), start_remote('host:port')."""
        return self.app.start_remote(host, port)
    def stop_remote(self):
        self.app.stop_remote()
    def remote_info(self):
        return self.app.remote_info()
    # ---------- documentation ----------

    def help(self, topic=None):
        """Return the API reference text, or the doc of one method."""
        if topic is None:
            return api_reference_text()
        fn = getattr(self, topic, None)
        if callable(fn) and getattr(fn, '__doc__', None):
            doc = inspect.getdoc(fn) or ''
            first, _, _ = doc.partition('\n')
            return '%s%s\n  %s' % (topic, _fmt_signature(fn), first)
        return api_reference_text()
def save_image_write(path, rows, app=None):
    """Convenience exporter for scripts / drivers."""
    save_image_path(path, rows, app)
# ============================================================

#  Handlers settings dialog: in-window load/save logic editor

# ============================================================

HANDLER_LOADER_TEMPLATE = (
    "# Load / read logic: loader(path, **ctx) -> (w, h, rows).\n"
    "# ctx.file_path / ctx.app / ctx.canvas are provided when available.\n"
    "# Register with register_image_loader('ext', loader).\n"
    "\n"
    "# Example loader (a 4x3 rgb image made from raw bytes):\n"
    "\n"
    "def my_loader(path, **ctx):\n"
    "    with open(path, 'rb') as f:\n"
    "        raw = f.read()[:12]\n"
    "    return (4, 3, [[(int(raw[i]), int(raw[i + 1]), int(raw[i + 2]))\n"
    "                    for i in range(0, 12, 3)]])\n"
    "register_image_loader('myfmt', my_loader)\n"
)
HANDLER_SAVER_TEMPLATE = (
    "# Save logic: saver(path, rows, **ctx) with ctx carrying\n"
    "# file_path / pixel_rows / tensor / width / height /\n"
    "# canvas_name / canvas / app.\n"
    "\n"
    "# Example saver (writes a text dump including canvas name):\n"
    "\n"
    "def my_saver(path, rows, **ctx):\n"
    "    with open(path, 'w') as f:\n"
    "        f.write('MYSV %dx%d canvas=%s sum=%d' %\n"
    "                (ctx['width'], ctx['height'],\n"
    "                 ctx.get('canvas_name'),\n"
    "                 sum(v for r in rows for px in r for v in px)))\n"
    "register_image_saver('mysv', my_saver)\n"
)
HANDLER_REMOTE_TEMPLATE = (
    "# Remote / external operation logic (optional runtime API).\n"
    "# register_external('name', fn) makes fn callable remotely\n"
    "# ({\"cmd\": \"call\", \"name\": ..., \"args\": [...]}) and from\n"
    "# injected code via call_external('name', ...).\n"
    "\n"
    "def ping_op(**kw):\n"
    "    return {'pong': True, 'tensor_shape': api.tensor_shape()}\n"
    "register_external('ping_op', ping_op)\n"
    "\n"
    "# Open the TCP control server (127.0.0.1) when this tab applies:\n"
    "# start_remote(0)   # 0 = OS-assigned port, shown in the status bar\n"
)
class HandlerSettingsDialog(tk.Toplevel):
    """Modern 3-category handler editor: Load / Save / Remote logic.

    Each category is a tab with its own editor and Import/Export
    buttons.  Apply executes all non-empty tabs in one shared namespace
    so loaders, savers and remote hooks can cooperate.  The bottom
    section controls the optional TCP remote server.
    """
    CATS = (('loader', 'Load Logic (reading)', HANDLER_LOADER_TEMPLATE),
            ('saver', 'Save Logic (writing)', HANDLER_SAVER_TEMPLATE),
            ('remote', 'Remote Logic (ops)', HANDLER_REMOTE_TEMPLATE))
    def __init__(self, master, platform):
        super().__init__(master)
        self.title('Image Handler Settings')
        self.geometry('780x640')
        self.platform = platform
        intro = ('Three categories of custom logic: reading '
                 '(register_image_loader), writing (register_image_saver) '
                 'and remote/external operations (register_external, '
                 'optional start_remote).  Each tab applies independently; '
                 'code runs in a dedicated namespace (caller trusted).')
        ttk.Label(self, text=intro, wraplength=740,
                  justify='left').pack(padx=8, pady=(8, 4))
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.editors = {}
        for key, label, template in self.CATS:
            tab = ttk.Frame(self.nb)
            self.nb.add(tab, text='  %s  ' % label)
            bar = ttk.Frame(tab)
            bar.pack(fill=tk.X, padx=4, pady=(4, 0))
            edit = scrolledtext.ScrolledText(tab, font=('Consolas', 10),
                                             wrap='word')
            edit.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
            edit.insert('1.0', template)
            edit.bind('<KeyRelease>',
                      lambda e, ed=edit: schedule_highlight(ed))
            schedule_highlight(edit, delay=0)
            ttk.Button(bar, text='Import Code...',
                       command=lambda k=key: self._import_code(k)
                       ).pack(side=tk.LEFT, padx=(0, 6))
            ttk.Button(bar, text='Save Code...',
                       command=lambda k=key: self._export_code(k)
                       ).pack(side=tk.LEFT)
            self.editors[key] = edit
        remote = ttk.LabelFrame(self, text=' Remote control server (optional) ')
        remote.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(remote, text='Host:').pack(side=tk.LEFT, padx=(8, 2), pady=4)
        self.remote_host_var = tk.StringVar(value='127.0.0.1')
        ttk.Entry(remote, width=12, textvariable=self.remote_host_var
                  ).pack(side=tk.LEFT, pady=4)
        ttk.Label(remote, text='Port:').pack(side=tk.LEFT, padx=(8, 2), pady=4)
        self.remote_port_var = tk.StringVar(value='0')
        ttk.Entry(remote, width=6, textvariable=self.remote_port_var
                  ).pack(side=tk.LEFT, pady=4)
        ttk.Button(remote, text='Start', command=self._start_remote
                   ).pack(side=tk.LEFT, padx=(8, 4), pady=4)
        ttk.Button(remote, text='Stop',
                   command=lambda: self.platform.stop_remote()
                   ).pack(side=tk.LEFT, pady=4)
        self.remote_info_label = ttk.Label(remote, text='not running')
        self.remote_info_label.pack(side=tk.LEFT, padx=12)
        current = ttk.LabelFrame(self, text='Active handlers')
        current.pack(fill=tk.X, padx=8, pady=4)
        self.list_txt = tk.Text(current, height=6, state='disabled',
                                font=('Consolas', 9))
        self.list_txt.pack(fill=tk.X, padx=4, pady=4)
        self._refresh()
        btns = ttk.Frame(self)
        btns.pack(fill=tk.X, pady=6)
        ttk.Button(btns, text='Apply', command=self._apply).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text='Refresh List', command=self._refresh).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text='Close', command=self.destroy).pack(side=tk.RIGHT, padx=6)
    def _start_remote(self):
        host = self.remote_host_var.get().strip() or '127.0.0.1'
        try:
            port = int(self.remote_port_var.get().strip() or '0')
        except ValueError:
            messagebox.showerror('Remote Server', 'Port must be an integer.')
            return
        try:
            self.platform.start_remote(host, port)
        except Exception as e:
            messagebox.showerror('Remote Server', '%s: %s'
                                 % (type(e).__name__, e))
        self._refresh()
    def _refresh(self):
        self.list_txt.config(state='normal')
        self.list_txt.delete('1.0', tk.END)
        loaders = ', '.join(sorted(IMAGE_LOADERS)) or '(none)'
        savers = ', '.join(sorted(IMAGE_SAVERS)) or '(none)'
        external = ', '.join(sorted(self.platform.external_hooks)) or '(none)'
        self.list_txt.insert(tk.END,
                             'Loaders:  %s\nSavers:   %s\nExternal: %s\n'
                             % (loaders, savers, external))
        self.list_txt.config(state='disabled')
        info = self.platform.remote_info()
        if info.get('running'):
            self.remote_info_label.config(
                text='running on %s:%d' % (info['host'], info['port']))
        else:
            self.remote_info_label.config(text='not running')
    def _import_code(self, key, path=None):
        """Load an external .py file into one tab's editor."""
        if path is None:
            path = filedialog.askopenfilename(
                filetypes=[('Python', '*.py *.pyw'), ('All files', '*.*')])
        if not path:
            return False
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                code = f.read()
        except Exception as e:
            messagebox.showerror('Import Handler Code', str(e))
            return False
        self.editors[key].delete('1.0', tk.END)
        self.editors[key].insert('1.0', code)
        self.platform.status.config(text='Handler code imported from %s'
                                    % os.path.basename(path))
        return True
    def _export_code(self, key, path=None):
        """Save one tab's editor content to a .py file."""
        if path is None:
            path = filedialog.asksaveasfilename(
                defaultextension='.py',
                filetypes=[('Python', '*.py'), ('All files', '*.*')])
        if not path:
            return False
        code = self.editors[key].get('1.0', 'end-1c')
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(code)
        except Exception as e:
            messagebox.showerror('Save Handler Code', str(e))
            return False
        self.platform.status.config(text='Handler code saved to %s'
                                    % os.path.basename(path))
        return True
    def _apply(self):
        ns = {
            'register_image_loader': register_image_loader,
            'register_image_saver': register_image_saver,
            'register_external': self.platform.register_external,
            'call_external': self.platform.call_external,
            'start_remote': self.platform.start_remote,
            'stop_remote': self.platform.stop_remote,
            'IMAGE_LOADERS': IMAGE_LOADERS,
            'IMAGE_SAVERS': IMAGE_SAVERS,
            'load_image_path': load_image_path,
            'save_image_path': save_image_path,
            'heatmap_to_rows': heatmap_to_rows,
            'PixelBuffer': PixelBuffer,
            'app': self.platform,
            'platform': self.platform,
            'api': self.platform.api,
            'os': os,
            'traceback': traceback,
            'io': io,
            'json': json,
            '__builtins__': __builtins__,
        }
        applied = 0
        for key, _label, _tpl in self.CATS:
            code = self.editors[key].get('1.0', 'end-1c')
            if not code.strip():
                continue
            try:
                exec(compile(code, '<handler:%s>' % key, 'exec'), ns)
                applied += 1
            except Exception as e:
                messagebox.showerror(
                    'Handler Error (%s)' % key,
                    '%s: %s\n%s' % (type(e).__name__, e,
                                    traceback.format_exc()))
                return
        if applied == 0:
            messagebox.showwarning('Handler Settings', 'Nothing to apply.')
            return
        self._refresh()
        self.platform.status.config(text='Handlers applied (%d tabs).'
                                    % applied)
class CompanionPlatform(object):
    """Head-less-ish facade for external Python programs.
    Create once, keep anywhere, show/hide the window, pump events
    so Tk stays responsive without entering mainloop::
        comp = create_platform()
        comp.app.get_canvas('Original')
        comp.app.load_image_rows(canvas, rows)
        comp.show()               # deiconify + pump
        # ... draw / analyse ...

        comp.hide()
        comp.close()
    """
    def __init__(self, app=None):
        self.app = app or PlatformApp()
        self.app.withdraw()          # start hidden
        self._pumping = False
    # ---------- window visibility ----------

    def show(self):
        """Persistent presentation: deiconify the window and pump."""
        try:
            self.app.deiconify()
        except Exception:
            pass
        self.pump()
    def hide(self):
        try:
            self.app.withdraw()
        except Exception:
            pass
        self.pump()
    def close(self):
        try:
            self.app.destroy()
        except Exception:
            pass
    # ---------- event loop ----------

    def pump(self, msec=50):
        """Process pending Tk events without entering mainloop."""
        deadline = time.time() + msec / 1000.0
        while time.time() < deadline:
            try:
                self.app.update()
            except tk.TclError:
                break
    def is_visible(self):
        try:
            return self.app.state() == 'normal'
        except Exception:
            return False
    # ---------- canvas conveniences ----------

    def canvas(self, name='Original'):
        return self.app.get_canvas(name)
    def load_rows(self, rows, canvas_name='Original'):
        c = self.canvas(canvas_name)
        if c is None:
            return False
        return self.app.load_image_rows(rows, canvas=c)
    def get_rows(self, canvas_name='Original', x=0, y=0, w=None, h=None):
        c = self.canvas(canvas_name)
        if c is None:
            return None
        return c.get_data(x, y, w, h)
    def get_pixel(self, x, y, canvas_name='Original'):
        c = self.canvas(canvas_name)
        return c.get_pixel_rgb(x, y) if c else None
    def set_pixel(self, x, y, color, canvas_name='Original'):
        c = self.canvas(canvas_name)
        return c.set_pixel(x, y, color) if c else False
    # ---------- modern API passthrough (same names as api.*) ----------

    def __getattr__(self, name):
        """Delegate unknown attributes to the app's ApiFacade, e.g.
        p.state(), p.set_tensor(...), p.tensor_shape(), p.start_remote().
        """
        facade = self.__dict__.get('_facade')
        if facade is None:
            facade = self.app.api
            self.__dict__['_facade'] = facade
        if hasattr(facade, name):
            return getattr(facade, name)
        raise AttributeError(name)
    def exec_code(self, code):
        """Run code through the platform shell; returns the result dict."""
        return self.app.shell.exec_code(code)
    def register_external(self, name, fn):
        return self.app.register_external(name, fn)
    def call_external(self, name, *args, **kwargs):
        return self.app.call_external(name, *args, **kwargs)
    def canvas_names(self):
        return self.app.canvas_names()
    def get_platform_state(self):
        return self.app.get_platform_state()
    def state(self):
        return self.app.get_platform_state()
    def export_canvas(self, canvas_name, path):
        return self.app.api.export_canvas(path, canvas_name=canvas_name)
    def import_canvas(self, path, canvas_name=None):
        return self.app.api.import_canvas(path, canvas_name=canvas_name)
    def snapshot(self):
        return self.app.snapshot_canvases()
    def restore(self, snap):
        return self.app.restore_canvases(snap)
    def save_state(self, path):
        return self.app.save_platform_state(path)
    def load_state(self, path):
        return self.app.load_platform_state(path)
def create_platform(app=None):
    """Create a persistent (hidden) platform for external programs."""
    return CompanionPlatform(app)
create_companion = create_platform
def create_companion_platform(app=None):
    return CompanionPlatform(app)
# ============================================================

#  Entry points

# ============================================================

def launch(width=512, height=512):
    """Launch the integrated GUI application."""
    app = PlatformApp(width=width, height=height)
    app.mainloop()
    return app
# ============================================================

#  Integrated GUI entry point (self-contained)

# ============================================================

def _load_api_file(app, path):
    """Execute a python file with 'app' pre-bound; the file may call
    app.add_api / app.set_api_hook to customise the injection namespace."""
    ns = {'app': app, 'tp': sys.modules[__name__], '__name__': '__injected__',
          '__builtins__': __builtins__}
    with open(path, encoding='utf-8') as f:
        code = f.read()
    exec(compile(code, path, 'exec'), ns)
def _load_script_file(path):
    try:
        with open(path, encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print('script load error:', e)
        return None
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    api_file = None
    script_file = None
    image_file = None
    handlers_file = None
    remote_port = None
    smoke = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--smoke':
            smoke = True
            i += 1
        elif a == '--remote':
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            if nxt is not None and not nxt.startswith('--'):
                remote_port = nxt
                i += 2
            else:
                remote_port = '0'
                i += 1
        elif a in ('--api', '--script', '--image', '--handlers') and \
                i + 1 < len(argv):
            value = argv[i + 1]
            if a == '--api':
                api_file = value
            elif a == '--script':
                script_file = value
            elif a == '--image':
                image_file = value
            else:
                handlers_file = value
            i += 2
        else:
            i += 1
    app = PlatformApp()
    if smoke:
        app.withdraw()
        app.update_idletasks()
        if api_file:
            try:
                _load_api_file(app, api_file)
                print('api file applied:', api_file)
            except Exception:
                traceback.print_exc()
        if handlers_file:
            try:
                _load_handlers_file(app, handlers_file)
                print('handlers file applied:', handlers_file)
            except Exception:
                traceback.print_exc()
        if remote_port is not None:
            host, port = '127.0.0.1', 0
            if isinstance(remote_port, str):
                p = remote_port
                if p.startswith('['):
                    close = p.find(']')
                    if close > 0:
                        host = p[1:close]
                        rest = p[close + 1:]
                        if rest.startswith(':'):
                            port = int(rest[1:])
                elif p.count(':') == 1 and p.rsplit(':', 1)[1].isdigit():
                    host, port = p.rsplit(':', 1)[0], int(p.rsplit(':', 1)[1])
                elif p.isdigit():
                    port = int(p)
                else:
                    host = p
            else:
                port = remote_port
            bound = app.start_remote(host=host, port=port)
            print('remote control on %s:%s'
                  % (host if host else '*', bound))
        if script_file:
            code = _load_script_file(script_file)
            if code is not None:
                app.script_editor.delete('1.0', 'end')
                app.script_editor.insert('1.0', code)
        app.update_idletasks()
        ok_api = True
        if api_file:
            probe = None
            try:
                probe = app.get_current_canvas() is not None
            except Exception:
                probe = False
            ok_api = probe is True
        app.destroy()
        print('SMOKE_OK api=%s script=%s image=%s handlers=%s remote=%s' % (
            'applied' if api_file else '-',
            'applied' if script_file else '-',
            'applied' if image_file else '-',
            'applied' if handlers_file else '-',
            'started' if remote_port is not None else '-'))
        return 0 if ok_api else 1
    if api_file:
        try:
            _load_api_file(app, api_file)
            print('api file applied:', api_file)
        except Exception:
            traceback.print_exc()
    if handlers_file:
        try:
            _load_handlers_file(app, handlers_file)
            print('handlers file applied:', handlers_file)
        except Exception:
            traceback.print_exc()
    if remote_port is not None:
        port = app.start_remote(remote_port)
        print('remote control on 127.0.0.1:%d' % port)
    if script_file:
        code = _load_script_file(script_file)
        if code is not None:
            app.script_editor.delete('1.0', 'end')
            app.script_editor.insert('1.0', code)
    if image_file:
        app.open_image(image_file)
    app.mainloop()
    return 0
def _load_handlers_file(app, path):
    """Execute a handlers file (loaders/savers/remote hooks) before the
    GUI starts; same namespace as the Handlers dialog Apply."""
    ns = {
        'register_image_loader': register_image_loader,
        'register_image_saver': register_image_saver,
        'register_external': app.register_external,
        'call_external': app.call_external,
        'start_remote': app.start_remote,
        'stop_remote': app.stop_remote,
        'IMAGE_LOADERS': IMAGE_LOADERS,
        'IMAGE_SAVERS': IMAGE_SAVERS,
        'load_image_path': load_image_path,
        'save_image_path': save_image_path,
        'heatmap_to_rows': heatmap_to_rows,
        'PixelBuffer': PixelBuffer,
        'app': app,
        'platform': app,
        'api': app.api,
        'os': os,
        'traceback': traceback,
        'io': io,
        'json': json,
        '__builtins__': __builtins__,
    }
    with open(path, encoding='utf-8') as f:
        code = f.read()
    exec(compile(code, path, 'exec'), ns)
if __name__ == '__main__':
    sys.exit(main())
