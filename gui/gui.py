#!/usr/bin/env python3
# front-end for the shape recognition pipeline

import os
import sys
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

try:
    from PIL import Image, ImageTk
except ImportError:
    print("Pillow is required: pip install Pillow", file=sys.stderr)
    sys.exit(1)

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(SCRIPT_DIR)
PIPELINE_BIN = os.path.join(PROJECT_DIR, "pipeline_app")

STAGES = [
    ("0 – Input",      None),
    ("1 – Grayscale",  "out_1_gray.pgm"),
    ("2 – Blurred",    "out_2_blurred.pgm"),
    ("3 – Edges",      "out_3_edges.pgm"),
    ("4 – Binary",     "out_4_binary.pgm"),
    ("5 – Labels",     "out_5_labels.pgm"),
    ("6 – Contours",   "out_6_contours.pgm"),
]

BG      = "#1e1e1e"
PANEL   = "#252526"
INPUT   = "#3c3c3c"
FG      = "#d4d4d4"
FG_DIM  = "#858585"
ACCENT  = "#0e639c"
ACCENT2 = "#569cd6"
OK      = "#4ec9b0"
ERR     = "#f48771"
WARN    = "#dcdcaa"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Shape Recognition Pipeline")
        self.configure(bg=BG)
        self.minsize(960, 600)

        self._image_path   = tk.StringVar()
        self._blur_radius  = tk.IntVar(value=3)
        self._edge_thresh  = tk.IntVar(value=40)
        self._stage_idx    = tk.IntVar(value=0)
        self._current_photo = None   # prevent PhotoImage GC
        self._raw_image     = None
        self._pipeline_ran  = False

        self._build_ui()

    def _build_ui(self):
        left = tk.Frame(self, bg=PANEL, width=240)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        self._build_controls(left)

        right = tk.Frame(self, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_canvas(right)

    def _build_controls(self, parent):
        def lbl(text, **kw):
            kw.setdefault("font", ("Segoe UI", 9))
            tk.Label(parent, text=text, bg=PANEL, fg=FG,
                     **kw).pack(anchor="w", padx=12, pady=(8, 0))

        def sep():
            tk.Frame(parent, bg="#404040", height=1).pack(fill=tk.X, padx=12, pady=6)

        # file picker
        lbl("Input Image", font=("Segoe UI", 9, "bold"))
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill=tk.X, padx=12, pady=(2, 0))
        tk.Entry(row, textvariable=self._image_path,
                 bg=INPUT, fg=FG, insertbackground="white",
                 relief=tk.FLAT, font=("Consolas", 8)
                 ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(row, text="…", command=self._pick_file,
                  bg=INPUT, fg=FG, relief=tk.FLAT,
                  activebackground="#505050", cursor="hand2",
                  font=("Segoe UI", 9)
                  ).pack(side=tk.LEFT, padx=(3, 0))

        sep()

        # blur radius
        lbl("Blur Radius")
        r = tk.Frame(parent, bg=PANEL)
        r.pack(fill=tk.X, padx=12)
        tk.Scale(r, from_=1, to=5, orient=tk.HORIZONTAL,
                 variable=self._blur_radius,
                 bg=PANEL, fg=FG, troughcolor=INPUT,
                 highlightthickness=0, activebackground=ACCENT2
                 ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(r, textvariable=self._blur_radius, bg=PANEL, fg=ACCENT2,
                 width=2, font=("Consolas", 10, "bold")).pack(side=tk.LEFT)

        # edge threshold
        lbl("Edge Threshold")
        r2 = tk.Frame(parent, bg=PANEL)
        r2.pack(fill=tk.X, padx=12)
        tk.Scale(r2, from_=1, to=254, orient=tk.HORIZONTAL,
                 variable=self._edge_thresh,
                 bg=PANEL, fg=FG, troughcolor=INPUT,
                 highlightthickness=0, activebackground=ACCENT2
                 ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(r2, textvariable=self._edge_thresh, bg=PANEL, fg=ACCENT2,
                 width=3, font=("Consolas", 10, "bold")).pack(side=tk.LEFT)

        sep()

        # run button + status
        self._run_btn = tk.Button(parent, text="▶  Run Pipeline",
                                   command=self._run_pipeline,
                                   bg=ACCENT, fg="white",
                                   activebackground="#1177bb",
                                   relief=tk.FLAT, cursor="hand2",
                                   font=("Segoe UI", 10, "bold"), pady=6)
        self._run_btn.pack(fill=tk.X, padx=12, pady=4)

        self._status = tk.Label(parent, text="", bg=PANEL, fg=FG_DIM,
                                 font=("Segoe UI", 8), wraplength=210)
        self._status.pack(padx=12)

        sep()

        # stage toggle
        tk.Label(parent, text="Output Stage", bg=PANEL, fg=FG,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=12, pady=(0, 2))
        for i, (label, _) in enumerate(STAGES):
            tk.Radiobutton(parent, text=label, variable=self._stage_idx,
                           value=i, command=self._show_stage,
                           bg=PANEL, fg=FG, selectcolor=PANEL,
                           activebackground="#2d2d2d", activeforeground="white",
                           font=("Segoe UI", 9), cursor="hand2"
                           ).pack(anchor="w", padx=20, pady=1)

        sep()

        # shape results
        tk.Label(parent, text="Shape Results", bg=PANEL, fg=FG,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=12, pady=(0, 2))
        self._results = scrolledtext.ScrolledText(
            parent, bg="#1e1e1e", fg=OK,
            font=("Consolas", 8), relief=tk.FLAT, state=tk.DISABLED)
        self._results.pack(fill=tk.BOTH, padx=12, pady=(0, 12), expand=True)

    def _build_canvas(self, parent):
        self._canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas_img  = self._canvas.create_image(0, 0, anchor="center")
        self._canvas_hint = self._canvas.create_text(
            0, 0, anchor="center",
            text="Run the pipeline to see output",
            fill="#555555", font=("Segoe UI", 14))
        self._canvas.bind("<Configure>", self._on_resize)

    def _pick_file(self):
        path = filedialog.askopenfilename(
            title="Select input image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.ppm *.bmp"),
                       ("All files", "*.*")])
        if path:
            self._image_path.set(path)
            self._show_stage()

    def _run_pipeline(self):
        path = self._image_path.get().strip()
        if not path:
            messagebox.showwarning("No image", "Select an input image first.")
            return
        if not os.path.exists(path):
            messagebox.showerror("Not found", f"File not found:\n{path}")
            return

        self._run_btn.configure(state=tk.DISABLED, text="Running…")
        self._set_status("Running pipeline…", WARN)
        self._set_results("")

        threading.Thread(
            target=self._pipeline_thread, args=(path,), daemon=True
        ).start()

    def _pipeline_thread(self, image_path):
        cmd = [PIPELINE_BIN, image_path,
               str(self._blur_radius.get()),
               str(self._edge_thresh.get())]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=PROJECT_DIR, timeout=120)
            ok  = proc.returncode == 0
            out = proc.stdout
            err = proc.stderr
        except FileNotFoundError:
            ok  = False
            out = ""
            err = f"pipeline_app not found at:\n{PIPELINE_BIN}\n\nRun 'make' first."
        except subprocess.TimeoutExpired:
            ok  = False
            out = ""
            err = "Pipeline timed out."

        self.after(0, self._pipeline_done, ok, out, err)

    def _pipeline_done(self, ok, stdout, stderr):
        self._run_btn.configure(state=tk.NORMAL, text="▶  Run Pipeline")
        if not ok:
            self._set_status("Pipeline failed.", ERR)
            messagebox.showerror("Pipeline error", stderr or "Unknown error")
            return

        self._set_status("Done.", OK)
        self._pipeline_ran = True

        lines = [l.strip() for l in stdout.splitlines()
                 if l.strip().startswith("Shape")]
        self._set_results("\n".join(lines) if lines else "(no shapes detected)")

        self._show_stage()

    def _show_stage(self):
        _, filename = STAGES[self._stage_idx.get()]

        if filename is None:
            path = self._image_path.get().strip()
            if not path or not os.path.exists(path):
                return
        else:
            if not self._pipeline_ran:
                return
            path = os.path.join(PROJECT_DIR, filename)
            if not os.path.exists(path):
                self._set_status(f"{filename} not found", ERR)
                return

        try:
            self._raw_image = Image.open(path).copy()
            self._render_image()
        except Exception as e:
            self._set_status(str(e), ERR)

    def _render_image(self):
        if self._raw_image is None:
            return
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        img = self._raw_image.copy()
        img.thumbnail((cw, ch), Image.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")

        photo = ImageTk.PhotoImage(img)
        self._current_photo = photo

        cx, cy = cw // 2, ch // 2
        self._canvas.coords(self._canvas_img,  cx, cy)
        self._canvas.coords(self._canvas_hint, cx, cy)
        self._canvas.itemconfigure(self._canvas_img,  image=photo)
        self._canvas.itemconfigure(self._canvas_hint, text="")

    def _on_resize(self, _event):
        if self._raw_image is not None:
            self._render_image()
        else:
            cw = self._canvas.winfo_width()
            ch = self._canvas.winfo_height()
            self._canvas.coords(self._canvas_hint, cw // 2, ch // 2)

    def _set_status(self, text, color=FG_DIM):
        self._status.configure(text=text, fg=color)

    def _set_results(self, text):
        self._results.configure(state=tk.NORMAL)
        self._results.delete("1.0", tk.END)
        self._results.insert(tk.END, text)
        self._results.configure(state=tk.DISABLED)


if __name__ == "__main__":
    App().mainloop()
