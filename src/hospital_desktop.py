"""
Hospital Federated Training Client & XAI Diagnostic Suite — Desktop Application
────────────────────────────────────────────────────────────────────────────────
Tkinter GUI that provides:
  1. Federated Training: Register/Login, Train on local patient data, submit weights.
  2. Clinical Diagnostics & XAI: Run Grad-CAM Explainable AI on chest X-rays with
     side-by-side visual heatmap rendering and lesion localization.
"""

import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import torch
from PIL import Image, ImageTk

# Add src to path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import CLASS_NAMES, MODELS_DIR
from hospital_client import login_hospital, register_hospital, train_and_submit
from model import build_model, load_checkpoint
from xai import explain_image


class HospitalApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hospital Federated Learning & XAI Diagnostic Client")
        self.geometry("900x760")
        self.minsize(820, 680)

        # ── State ──────────────────────────────────────────────────────────
        self.log_queue      = queue.Queue()
        self.progress_queue = queue.Queue()
        self.worker         = None
        self.token          = None          # Bearer token from register / login
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Config vars (Tab 1) ────────────────────────────────────────────
        self.server_url    = tk.StringVar(value="http://127.0.0.1:8000")
        self.hospital_name = tk.StringVar(value="")
        self.password      = tk.StringVar(value="")
        self.dataset_dir   = tk.StringVar(value="")
        self.round_id      = tk.StringVar(value="current")
        self.epochs        = tk.IntVar(value=1)
        self.batch_size    = tk.IntVar(value=16)
        self.force_cpu     = tk.BooleanVar(value=False)

        # ── Live status vars (Tab 1) ───────────────────────────────────────
        self.auth_status_var  = tk.StringVar(value="Not authenticated")
        self.token_display_var = tk.StringVar(value="—")
        self.train_status_var = tk.StringVar(value="Idle")
        self.progress_var     = tk.DoubleVar(value=0.0)
        self.epoch_var        = tk.StringVar(value="—")
        self.batch_var        = tk.StringVar(value="—")
        self.loss_var         = tk.StringVar(value="—")
        self.acc_var          = tk.StringVar(value="—")

        # ── State & vars (Tab 2: XAI) ──────────────────────────────────────
        self.xai_image_path   = tk.StringVar(value="")
        self.xai_model_path   = tk.StringVar(value=str(MODELS_DIR / "xray_global.pt"))
        self.xai_pred_var     = tk.StringVar(value="Waiting")
        self.xai_conf_var     = tk.StringVar(value="No image analyzed")
        self.xai_normal_var   = tk.StringVar(value="—")
        self.xai_pneu_var     = tk.StringVar(value="—")
        self.xai_alpha_var    = tk.DoubleVar(value=0.55)
        self.orig_photo       = None
        self.overlay_photo    = None
        self.last_xai_result  = None

        self._build_ui()
        self.after(150, self._flush)

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)

        # ── TAB 1: Federated Training ───────────────────────────────────────
        tab_train = ttk.Frame(notebook, padding=12)
        notebook.add(tab_train, text="  🏥 Federated Training & Submission  ")
        self._build_training_tab(tab_train)

        # ── TAB 2: Clinical Diagnostics & XAI ───────────────────────────────
        tab_xai = ttk.Frame(notebook, padding=12)
        notebook.add(tab_xai, text="  🩺 Clinical Diagnostics & Grad-CAM XAI  ")
        self._build_xai_tab(tab_xai)

    def _build_training_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(5, weight=1)

        # Title
        ttk.Label(
            parent,
            text="🏥  Hospital Federated Training Node",
            font=("Segoe UI", 13, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        # Server & Auth Frame
        auth = ttk.LabelFrame(parent, text="Hospital Identity & Server Connection", padding=8)
        auth.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        auth.columnconfigure(1, weight=1)

        ttk.Label(auth, text="Server URL:").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Entry(auth, textvariable=self.server_url).grid(row=0, column=1, columnspan=2, sticky="ew", pady=2)

        ttk.Label(auth, text="Hospital Name:").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Entry(auth, textvariable=self.hospital_name).grid(row=1, column=1, sticky="ew", pady=2, padx=(0, 6))

        btn_row = ttk.Frame(auth)
        btn_row.grid(row=1, column=2, sticky="e")
        self.reg_btn = ttk.Button(btn_row, text="Register", command=self._do_register, width=9)
        self.reg_btn.pack(side="left", padx=(0, 4))
        self.login_btn = ttk.Button(btn_row, text="Login", command=self._do_login, width=9)
        self.login_btn.pack(side="left")

        ttk.Label(auth, text="Password:").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Entry(auth, textvariable=self.password, show="*").grid(row=2, column=1, sticky="ew", pady=2, padx=(0, 6))

        self._auth_lbl = ttk.Label(auth, textvariable=self.auth_status_var, foreground="gray", font=("Segoe UI", 9))
        self._auth_lbl.grid(row=3, column=0, columnspan=3, sticky="w", pady=(2, 0))

        ttk.Label(auth, text="Bearer Token:").grid(row=4, column=0, sticky="w", padx=(0, 6), pady=(2, 2))
        tok_entry = ttk.Entry(auth, textvariable=self.token_display_var, state="readonly", font=("Consolas", 8))
        tok_entry.grid(row=4, column=1, sticky="ew", pady=(2, 2), padx=(0, 6))

        # Training Config
        cfg = ttk.LabelFrame(parent, text="Local Dataset & Training Setup", padding=8)
        cfg.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        cfg.columnconfigure(1, weight=1)

        ttk.Label(cfg, text="Dataset Folder:").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Entry(cfg, textvariable=self.dataset_dir).grid(row=0, column=1, sticky="ew", pady=2, padx=(0, 6))
        ttk.Button(cfg, text="Browse", command=self._pick_dataset, width=7).grid(row=0, column=2, pady=2)

        ttk.Label(cfg, text="Round ID:").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Entry(cfg, textvariable=self.round_id, width=18).grid(row=1, column=1, sticky="w", pady=2)

        params = ttk.Frame(cfg)
        params.grid(row=2, column=0, columnspan=3, sticky="w", pady=2)
        ttk.Label(params, text="Epochs:").pack(side="left")
        ttk.Spinbox(params, from_=1, to=20, textvariable=self.epochs, width=4).pack(side="left", padx=(3, 10))
        ttk.Label(params, text="Batch:").pack(side="left")
        ttk.Spinbox(params, from_=1, to=128, textvariable=self.batch_size, width=4).pack(side="left", padx=(3, 10))
        ttk.Checkbutton(params, text="Force CPU", variable=self.force_cpu).pack(side="left")

        # Train Action Button
        self.start_btn = ttk.Button(parent, text="⚡  Train Locally and Submit Weights", command=self._start_training)
        self.start_btn.grid(row=3, column=0, sticky="ew", pady=(0, 8), ipady=4)

        # Progress
        prog = ttk.LabelFrame(parent, text="Training Progress", padding=8)
        prog.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        prog.columnconfigure(3, weight=1)

        ttk.Label(prog, text="Status:").grid(row=0, column=0, sticky="w")
        ttk.Label(prog, textvariable=self.train_status_var, foreground="#0f766e", font=("Segoe UI", 9, "bold")).grid(row=0, column=1, columnspan=3, sticky="w", padx=(4, 0))

        self.pbar = ttk.Progressbar(prog, variable=self.progress_var, maximum=100, mode="determinate")
        self.pbar.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(4, 4))

        stats = ttk.Frame(prog)
        stats.grid(row=2, column=0, columnspan=4, sticky="w")
        for lbl, var in [("Epoch:", self.epoch_var), ("Batch:", self.batch_var), ("Loss:", self.loss_var), ("Accuracy:", self.acc_var)]:
            ttk.Label(stats, text=lbl).pack(side="left", padx=(0, 2))
            ttk.Label(stats, textvariable=var, foreground="#0f766e", font=("Segoe UI", 9, "bold"), width=9).pack(side="left", padx=(0, 10))

        # Log
        log_frame = ttk.LabelFrame(parent, text="Execution Log", padding=6)
        log_frame.grid(row=5, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_box = tk.Text(log_frame, height=7, wrap="word", state="disabled", font=("Consolas", 8), relief="flat")
        sb = ttk.Scrollbar(log_frame, command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=sb.set)
        self.log_box.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

    def _build_xai_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        # File Selectors Frame
        top_bar = ttk.LabelFrame(parent, text="Model & Patient Radiograph Selection", padding=8)
        top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        top_bar.columnconfigure(1, weight=1)

        ttk.Label(top_bar, text="Model Checkpoint:").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Entry(top_bar, textvariable=self.xai_model_path).grid(row=0, column=1, sticky="ew", pady=2, padx=(0, 6))
        ttk.Button(top_bar, text="Browse", command=self._pick_xai_model, width=7).grid(row=0, column=2, pady=2)

        ttk.Label(top_bar, text="Patient X-Ray:").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Entry(top_bar, textvariable=self.xai_image_path).grid(row=1, column=1, sticky="ew", pady=2, padx=(0, 6))
        ttk.Button(top_bar, text="Browse", command=self._pick_xai_image, width=7).grid(row=1, column=2, pady=2)

        # Controls & Action
        ctrl_frame = ttk.Frame(top_bar)
        ctrl_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Label(ctrl_frame, text="Heatmap Blend:").pack(side="left", padx=(0, 4))
        alpha_scale = ttk.Scale(ctrl_frame, from_=0.1, to=0.9, variable=self.xai_alpha_var, orient="horizontal", length=130)
        alpha_scale.pack(side="left", padx=(0, 14))

        self.xai_run_btn = ttk.Button(ctrl_frame, text="⚡  Run Diagnostic & Grad-CAM Analysis", command=self._run_xai_analysis)
        self.xai_run_btn.pack(side="left", padx=(0, 8))

        self.xai_save_btn = ttk.Button(ctrl_frame, text="💾 Save XAI Visualization", command=self._save_xai_image, state="disabled")
        self.xai_save_btn.pack(side="left")

        # Results Summary Banner
        res_bar = ttk.Frame(parent, padding=6)
        res_bar.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(res_bar, text="Diagnostic Result:", font=("Segoe UI", 11, "bold")).pack(side="left", padx=(0, 6))
        self.xai_pred_lbl = ttk.Label(res_bar, textvariable=self.xai_pred_var, font=("Segoe UI", 14, "bold"), foreground="#64748b")
        self.xai_pred_lbl.pack(side="left", padx=(0, 16))

        ttk.Label(res_bar, textvariable=self.xai_conf_var, font=("Segoe UI", 10)).pack(side="left", padx=(0, 20))
        ttk.Label(res_bar, text="Normal:").pack(side="left", padx=(0, 2))
        ttk.Label(res_bar, textvariable=self.xai_normal_var, font=("Segoe UI", 9, "bold"), foreground="#10b981").pack(side="left", padx=(0, 12))
        ttk.Label(res_bar, text="Pneumonia:").pack(side="left", padx=(0, 2))
        ttk.Label(res_bar, textvariable=self.xai_pneu_var, font=("Segoe UI", 9, "bold"), foreground="#f43f5e").pack(side="left")

        # Visual Comparison Frame (Side by Side)
        viewer_frame = ttk.LabelFrame(parent, text="Visual Explainability — Original Radiograph vs Grad-CAM Saliency", padding=8)
        viewer_frame.grid(row=2, column=0, sticky="nsew")
        viewer_frame.columnconfigure(0, weight=1)
        viewer_frame.columnconfigure(1, weight=1)
        viewer_frame.rowconfigure(0, weight=1)

        # Left: Original Image
        left_box = ttk.Frame(viewer_frame)
        left_box.grid(row=0, column=0, sticky="nsew", padx=4)
        ttk.Label(left_box, text="Patient Radiograph (Input)", font=("Segoe UI", 10, "bold")).pack(pady=(0, 4))
        self.orig_canvas = tk.Label(left_box, text="No Image Selected\nClick 'Browse' above", bg="#0f172a", fg="#94a3b8", width=34, height=18)
        self.orig_canvas.pack(fill="both", expand=True)

        # Right: Grad-CAM Overlay
        right_box = ttk.Frame(viewer_frame)
        right_box.grid(row=0, column=1, sticky="nsew", padx=4)
        ttk.Label(right_box, text="Grad-CAM Lesion Heatmap Overlay", font=("Segoe UI", 10, "bold"), foreground="#0891b2").pack(pady=(0, 4))
        self.overlay_canvas = tk.Label(right_box, text="Grad-CAM Activation Map\nRun analysis to generate", bg="#0f172a", fg="#94a3b8", width=34, height=18)
        self.overlay_canvas.pack(fill="both", expand=True)

    # ── XAI Actions ────────────────────────────────────────────────────────

    def _pick_xai_model(self):
        f = filedialog.askopenfilename(title="Select Model Checkpoint", filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")])
        if f:
            self.xai_model_path.set(f)

    def _pick_xai_image(self):
        f = filedialog.askopenfilename(title="Select Chest X-Ray Image", filetypes=[("Images", "*.jpeg;*.jpg;*.png;*.bmp"), ("All Files", "*.*")])
        if f:
            self.xai_image_path.set(f)
            self._preview_raw_image(f)

    def _preview_raw_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((320, 320))
            self.orig_photo = ImageTk.PhotoImage(img)
            self.orig_canvas.configure(image=self.orig_photo, text="")
        except Exception as exc:
            messagebox.showerror("Image Error", f"Could not load image: {exc}")

    def _run_xai_analysis(self):
        img_path = self.xai_image_path.get().strip()
        mod_path = self.xai_model_path.get().strip()
        alpha = self.xai_alpha_var.get()

        if not img_path or not os.path.exists(img_path):
            messagebox.showerror("Missing Image", "Please select a valid chest X-ray image.")
            return
        if not mod_path or not os.path.exists(mod_path):
            messagebox.showerror("Missing Model", f"Model checkpoint not found at:\n{mod_path}")
            return

        self.xai_run_btn.configure(state="disabled")
        self.xai_pred_var.set("Analyzing…")
        self.xai_conf_var.set("Computing feature gradients…")

        def work():
            try:
                model = build_model(pretrained=False).to(self.device)
                load_checkpoint(model, mod_path, self.device)
                model.eval()

                result = explain_image(
                    model=model,
                    image_input=img_path,
                    device=self.device,
                    alpha=alpha,
                )
                self.last_xai_result = result
                self.after(0, lambda: self._display_xai_results(result))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("XAI Diagnostic Error", str(exc)))
                self.after(0, lambda: self.xai_pred_var.set("Error"))
            finally:
                self.after(0, lambda: self.xai_run_btn.configure(state="normal"))

        threading.Thread(target=work, daemon=True).start()

    def _display_xai_results(self, result):
        pred = result["prediction"]
        conf = result["confidence"]
        self.xai_pred_var.set(pred)
        self.xai_conf_var.set(f"Confidence: {conf * 100:.1f}%")

        if pred == "NORMAL":
            self.xai_pred_lbl.configure(foreground="#10b981")
        else:
            self.xai_pred_lbl.configure(foreground="#f43f5e")

        for item in result["probabilities"]:
            pct = f"{item['probability'] * 100:.1f}%"
            if item["class_name"].upper() == "NORMAL":
                self.xai_normal_var.set(pct)
            else:
                self.xai_pneu_var.set(pct)

        # Scale and render images
        orig_disp = result["overlay_pil"].copy()
        # Original
        raw_pil = Image.open(self.xai_image_path.get().strip()).convert("RGB")
        raw_pil.thumbnail((340, 340))
        self.orig_photo = ImageTk.PhotoImage(raw_pil)
        self.orig_canvas.configure(image=self.orig_photo, text="")

        # Overlay
        orig_disp.thumbnail((340, 340))
        self.overlay_photo = ImageTk.PhotoImage(orig_disp)
        self.overlay_canvas.configure(image=self.overlay_photo, text="")

        self.xai_save_btn.configure(state="normal")

    def _save_xai_image(self):
        if not self.last_xai_result:
            return
        out_path = filedialog.asksaveasfilename(
            title="Save Grad-CAM Heatmap Image",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")],
        )
        if out_path:
            self.last_xai_result["overlay_pil"].save(out_path)
            messagebox.showinfo("Saved", f"Grad-CAM visualization saved to:\n{out_path}")

    # ── Auth helpers ───────────────────────────────────────────────────────

    def _set_auth(self, text: str, color: str = "gray"):
        self.auth_status_var.set(text)
        self._auth_lbl.configure(foreground=color)

    def _update_token_display(self, token: str):
        if token and len(token) > 12:
            preview = f"{token[:8]}...{token[-4:]}"
        elif token:
            preview = token[:8] + "..."
        else:
            preview = "—"
        self.token_display_var.set(preview)

    def _do_register(self):
        name = self.hospital_name.get().strip()
        password = self.password.get().strip()
        if not name:
            messagebox.showerror("Missing Name", "Please enter a hospital name first.")
            return
        if not password:
            messagebox.showerror("Missing Password", "Please enter a password.")
            return
        self.reg_btn.configure(state="disabled")
        self._set_auth("Registering…", "#6b7280")

        def work():
            try:
                res = register_hospital(
                    self.server_url.get().strip(), name, password, logger=self.log_queue.put
                )
                self.token = res["token"]
                self.after(0, lambda: self._set_auth(f'✓  Registered as "{name}"', "#067647"))
                self.after(0, lambda: self._update_token_display(res["token"]))
                self.after(0, lambda: self.log_queue.put(
                    "[Auth] Registration successful. Token stored (Bearer auth enabled)."
                ))
            except Exception as exc:
                self.after(0, lambda: self._set_auth(f"✗  {exc}", "#b42318"))
                self.after(0, lambda: self.log_queue.put(f"[Auth] Registration failed: {exc}"))
            finally:
                self.after(0, lambda: self.reg_btn.configure(state="normal"))

        threading.Thread(target=work, daemon=True).start()

    def _do_login(self):
        name = self.hospital_name.get().strip()
        password = self.password.get().strip()
        if not name:
            messagebox.showerror("Missing Name", "Please enter a hospital name first.")
            return
        if not password:
            messagebox.showerror("Missing Password", "Please enter your password.")
            return
        self.login_btn.configure(state="disabled")
        self._set_auth("Logging in…", "#6b7280")

        def work():
            try:
                res = login_hospital(
                    self.server_url.get().strip(), name, password, logger=self.log_queue.put
                )
                self.token = res["token"]
                self.after(0, lambda: self._set_auth(f'✓  Logged in as "{name}"', "#067647"))
                self.after(0, lambda: self._update_token_display(res["token"]))
                self.after(0, lambda: self.log_queue.put(
                    "[Auth] Login successful. Token stored (Bearer auth enabled)."
                ))
            except Exception as exc:
                self.after(0, lambda: self._set_auth(f"✗  {exc}", "#b42318"))
                self.after(0, lambda: self.log_queue.put(f"[Auth] Login failed: {exc}"))
            finally:
                self.after(0, lambda: self.login_btn.configure(state="normal"))

        threading.Thread(target=work, daemon=True).start()

    # ── Training ───────────────────────────────────────────────────────────

    def _pick_dataset(self):
        selected = filedialog.askdirectory(title="Select hospital dataset folder")
        if selected:
            self.dataset_dir.set(selected)

    def _validate(self) -> bool:
        dataset = Path(self.dataset_dir.get())
        if not dataset.exists():
            messagebox.showerror("Missing Dataset", "Please select a valid dataset folder.")
            return False
        if not (dataset / "NORMAL").exists() or not (dataset / "PNEUMONIA").exists():
            messagebox.showerror(
                "Invalid Dataset",
                "The dataset folder must contain NORMAL and PNEUMONIA subfolders.",
            )
            return False
        if not self.hospital_name.get().strip():
            messagebox.showerror("Missing Name", "Please enter a hospital name.")
            return False
        if not self.token:
            messagebox.showerror("Not Authenticated", "Please Register or Login first to obtain a token.")
            return False
        return True

    def _start_training(self):
        if self.worker and self.worker.is_alive():
            return
        if not self._validate():
            return

        self.start_btn.configure(state="disabled")
        self.progress_var.set(0.0)
        self.train_status_var.set("Starting…")
        for v in (self.epoch_var, self.batch_var, self.loss_var, self.acc_var):
            v.set("—")
        self.log_queue.put("─" * 60)
        self.log_queue.put("Starting local training workflow…")
        token_preview = self.token_display_var.get()
        self.log_queue.put(f"[Auth] Using Bearer token: {token_preview}")

        self.worker = threading.Thread(target=self._run_training, daemon=True)
        self.worker.start()

    def _run_training(self):
        epochs            = self.epochs.get()
        current_epoch_ref = [1]

        def on_epoch(epoch, total_epochs, metrics):
            current_epoch_ref[0] = epoch
            pct = (epoch / total_epochs) * 100
            self.progress_queue.put({
                "epoch": epoch, "total_epochs": total_epochs,
                "loss": metrics["loss"], "accuracy": metrics["accuracy"],
                "pct": pct,
                "status": f"Epoch {epoch}/{total_epochs} complete — loss {metrics['loss']:.4f}",
            })

        def on_batch(batch, total_batches, loss, accuracy):
            e   = current_epoch_ref[0]
            pct = ((e - 1) + batch / max(total_batches, 1)) / max(epochs, 1) * 100
            self.progress_queue.put({
                "batch": batch, "total_batches": total_batches,
                "loss": loss, "accuracy": accuracy,
                "pct": min(pct, 99.0),
                "status": f"Training epoch {e}/{epochs}…",
            })

        try:
            self.after(0, lambda: self.train_status_var.set("Downloading global model…"))
            result = train_and_submit(
                server_url=self.server_url.get().strip(),
                hospital_name=self.hospital_name.get().strip(),
                dataset_dir=self.dataset_dir.get().strip(),
                round_id=self.round_id.get().strip() or "current",
                epochs=epochs,
                batch_size=self.batch_size.get(),
                force_cpu=self.force_cpu.get(),
                token=self.token,
                logger=self.log_queue.put,
                on_epoch=on_epoch,
                on_batch=on_batch,
            )
            self.progress_queue.put({
                "pct": 100.0, "status": "✓  Submitted successfully!",
            })
            self.log_queue.put(f"Server response: {result}")
            self.after(0, lambda: messagebox.showinfo(
                "Submitted", "Local weights submitted successfully."
            ))
        except Exception as exc:
            msg = str(exc)
            self.log_queue.put(f"Error: {msg}")
            self.progress_queue.put({"pct": 0.0, "status": f"✗  {msg}"})
            self.after(0, lambda: messagebox.showerror("Failed", msg))
        finally:
            self.after(0, lambda: self.start_btn.configure(state="normal"))

    # ── UI update loop ─────────────────────────────────────────────────────

    def append_log(self, text: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{text}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _flush(self):
        while not self.log_queue.empty():
            self.append_log(self.log_queue.get())

        latest = None
        while not self.progress_queue.empty():
            latest = self.progress_queue.get()
        if latest:
            self.progress_var.set(latest.get("pct", self.progress_var.get()))
            if "status" in latest:
                self.train_status_var.set(latest["status"])
            if "epoch" in latest:
                self.epoch_var.set(f"{latest['epoch']}/{latest.get('total_epochs', '?')}")
            if "batch" in latest:
                self.batch_var.set(f"{latest['batch']}/{latest.get('total_batches', '?')}")
            if "loss" in latest:
                self.loss_var.set(f"{latest['loss']:.4f}")
            if "accuracy" in latest:
                self.acc_var.set(f"{latest['accuracy'] * 100:.1f}%")

        self.after(150, self._flush)


if __name__ == "__main__":
    HospitalApp().mainloop()
