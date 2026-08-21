from __future__ import annotations

from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .renderer import render_scene
from .scene import load_scene


def launch() -> None:
    root = tk.Tk()
    root.title("ParallaxStudio3D")
    root.geometry("720x390")
    root.minsize(660, 360)

    default_scene = Path(__file__).resolve().parents[1] / "scenes" / "child_mother.json"
    scene_value = tk.StringVar(value=str(default_scene))
    output_value = tk.StringVar(
        value=str(Path(__file__).resolve().parents[1] / "output" / "demo")
    )
    image_value = tk.StringVar()
    status = tk.StringVar(value="Ready — unified depth camera, not sticker drift.")

    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)

    ttk.Label(frame, text="Single image").grid(row=0, column=0, sticky="w", pady=8)
    ttk.Entry(frame, textvariable=image_value).grid(row=0, column=1, sticky="ew", padx=8)
    ttk.Button(
        frame,
        text="Browse",
        command=lambda: image_value.set(
            filedialog.askopenfilename(
                filetypes=[("Images", "*.png *.jpg *.jpeg *.webp")]
            )
            or image_value.get()
        ),
    ).grid(row=0, column=2)

    ttk.Label(frame, text="Scene file").grid(row=1, column=0, sticky="w", pady=8)
    ttk.Entry(frame, textvariable=scene_value).grid(row=1, column=1, sticky="ew", padx=8)
    ttk.Button(
        frame,
        text="Browse",
        command=lambda: scene_value.set(
            filedialog.askopenfilename(filetypes=[("JSON scene", "*.json")])
            or scene_value.get()
        ),
    ).grid(row=1, column=2)

    ttk.Label(frame, text="Output folder").grid(row=2, column=0, sticky="w", pady=8)
    ttk.Entry(frame, textvariable=output_value).grid(row=2, column=1, sticky="ew", padx=8)
    ttk.Button(
        frame,
        text="Browse",
        command=lambda: output_value.set(
            filedialog.askdirectory() or output_value.get()
        ),
    ).grid(row=2, column=2)

    progress = ttk.Progressbar(frame, maximum=100)
    progress.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(22, 8))
    ttk.Label(frame, textvariable=status).grid(row=4, column=0, columnspan=3, sticky="w")

    def update_progress(done: int, total: int) -> None:
        root.after(0, lambda: progress.configure(value=done * 100 / total))
        root.after(0, lambda: status.set(f"Rendering frame {done} of {total}"))

    def worker() -> None:
        try:
            result = render_scene(
                load_scene(scene_value.get()), output_value.get(), update_progress
            )
            root.after(0, lambda: status.set(f"Done: {result.video}"))
            root.after(
                0,
                lambda: messagebox.showinfo(
                    "Render complete", f"Video:\n{result.video}\n\nGIF:\n{result.gif}"
                ),
            )
        except Exception as exc:
            root.after(0, lambda: status.set("Render failed"))
            root.after(0, lambda: messagebox.showerror("Render failed", str(exc)))
        finally:
            root.after(0, lambda: render_button.configure(state="normal"))

    def start() -> None:
        render_button.configure(state="disabled")
        progress.configure(value=0)
        status.set("Preparing depth layers...")
        threading.Thread(target=worker, daemon=True).start()

    def auto_analyze() -> None:
        if not image_value.get():
            messagebox.showwarning("Choose an image", "Select a source image first.")
            return
        from .analyzer import create_auto_project

        draft = Path(output_value.get()).parent / "auto_project"
        try:
            scene = create_auto_project(image_value.get(), draft)
            scene_value.set(str(scene))
            status.set("Free local draft created. Review it, then render.")
        except Exception as exc:
            messagebox.showerror("Analysis failed", str(exc))

    button_bar = ttk.Frame(frame)
    button_bar.grid(row=5, column=0, columnspan=3, pady=22)
    ttk.Button(button_bar, text="Auto-analyze image", command=auto_analyze).pack(
        side="left", padx=6
    )
    render_button = ttk.Button(button_bar, text="Render parallax", command=start)
    render_button.pack(side="left", padx=6)
    root.mainloop()
