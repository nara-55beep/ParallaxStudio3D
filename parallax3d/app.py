from __future__ import annotations

from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .renderer import render_scene
from .scene import load_scene
from .workflow import render_image


def launch() -> None:
    root = tk.Tk()
    root.title("ParallaxStudio3D")
    root.geometry("720x390")
    root.minsize(660, 360)

    image_value = tk.StringVar()
    scene_value = tk.StringVar()
    output_value = tk.StringVar(
        value=str(Path(__file__).resolve().parents[1] / "output" / "render")
    )
    status = tk.StringVar(
        value="Choose one image. AI will cut objects into real depth layers."
    )

    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)

    ttk.Label(frame, text="Single image").grid(row=0, column=0, sticky="w", pady=8)
    ttk.Entry(frame, textvariable=image_value).grid(
        row=0, column=1, sticky="ew", padx=8
    )
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

    ttk.Label(frame, text="Scene (optional)").grid(
        row=1, column=0, sticky="w", pady=8
    )
    ttk.Entry(frame, textvariable=scene_value).grid(
        row=1, column=1, sticky="ew", padx=8
    )
    ttk.Button(
        frame,
        text="Browse",
        command=lambda: scene_value.set(
            filedialog.askopenfilename(filetypes=[("JSON scene", "*.json")])
            or scene_value.get()
        ),
    ).grid(row=1, column=2)

    ttk.Label(frame, text="Output folder").grid(
        row=2, column=0, sticky="w", pady=8
    )
    ttk.Entry(frame, textvariable=output_value).grid(
        row=2, column=1, sticky="ew", padx=8
    )
    ttk.Button(
        frame,
        text="Browse",
        command=lambda: output_value.set(
            filedialog.askdirectory() or output_value.get()
        ),
    ).grid(row=2, column=2)

    progress = ttk.Progressbar(frame, maximum=100)
    progress.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(22, 8))
    ttk.Label(frame, textvariable=status).grid(
        row=4, column=0, columnspan=3, sticky="w"
    )

    def update_progress(done: int, total: int) -> None:
        root.after(0, lambda: progress.configure(value=done * 100 / total))
        root.after(0, lambda: status.set(f"Rendering frame {done} of {total}"))

    def update_status(message: str) -> None:
        root.after(0, lambda: status.set(message))

    def worker(image_path: str, scene_path: str, output_path: str) -> None:
        try:
            if image_path:
                result = render_image(
                    image_path,
                    output_path,
                    update_progress,
                    update_status,
                )
            else:
                result = render_scene(
                    load_scene(scene_path), output_path, update_progress
                )
            root.after(0, lambda: status.set(f"Done: {result.video}"))
            root.after(
                0,
                lambda: messagebox.showinfo(
                    "Render complete",
                    f"Video:\n{result.video}\n\nGIF:\n{result.gif}",
                ),
            )
        except Exception as exc:
            error_message = str(exc)
            root.after(0, lambda: status.set("Render failed"))
            root.after(
                0,
                lambda: messagebox.showerror("Render failed", error_message),
            )
        finally:
            root.after(0, lambda: render_button.configure(state="normal"))

    def start() -> None:
        image_path = image_value.get().strip()
        scene_path = scene_value.get().strip()
        output_path = output_value.get().strip()
        if not image_path and not scene_path:
            messagebox.showwarning(
                "Choose an image",
                "Select a JPG, PNG, or WebP image. A scene file is optional.",
            )
            return
        if not output_path:
            messagebox.showwarning("Choose output", "Select an output folder.")
            return
        render_button.configure(state="disabled")
        progress.configure(value=0)
        status.set("Starting automatic semantic layer detection...")
        threading.Thread(
            target=worker,
            args=(image_path, scene_path, output_path),
            daemon=True,
        ).start()

    button_bar = ttk.Frame(frame)
    button_bar.grid(row=5, column=0, columnspan=3, pady=22)
    render_button = ttk.Button(
        button_bar, text="AI Cut + Create MP4 + GIF", command=start
    )
    render_button.pack(side="left", padx=6)
    root.mainloop()
