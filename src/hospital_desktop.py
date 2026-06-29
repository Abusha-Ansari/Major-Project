"""
Hospital Federated Training Client — Desktop Application
─────────────────────────────────────────────────────────
Tkinter GUI that lets a hospital operator:
  1. Register / Login with the central server to obtain a token.
  2. Point to a local dataset folder (NORMAL + PNEUMONIA subfolders).
  3. Train a local model and stream live epoch/batch progress.
  4. Submit the encrypted weight update to the server.

Duplicate-submission protection: the server rejects a second submission
from the same hospital for the same round with a 409 error, which the
app surfaces as a clear user-facing message.
"""

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from hospital_client import login_hospital, register_hospital, train_and_submit


class HospitalApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hospital Federated Training Client")
        self.geometry("840x720")
        self.minsize(740, 600)

        # ── State ──────────────────────────────────────────────────────────
        self.log_queue      = queue.Queue()
        self.progress_queue = queue.Queue()
        self.worker         = None
        self.token          = None          # Bearer token from register / login

        # ── Config vars ────────────────────────────────────────────────────
        self.server_url    = tk.StringVar(value="http://127.0.0.1:8000")
        self.hospital_name = tk.StringVar(value="")
        self.password      = tk.StringVar(value="")
        self.dataset_dir   = tk.StringVar(value="")
        self.round_id      = tk.StringVar(value="current")
        self.epochs        = tk.IntVar(value=1)
        self.batch_size    = tk.IntVar(value=16)
        self.force_cpu     = tk.BooleanVar(value=False)

        # ── Live status vars ───────────────────────────────────────────────
        self.auth_status_var  = tk.StringVar(value="Not authenticated")
        self.token_display_var = tk.StringVar(value="—")   # masked preview
        self.train_status_var = tk.StringVar(value="Idle")
        self.progress_var     = tk.DoubleVar(value=0.0)
        self.epoch_var        = tk.StringVar(value="—")
        self.batch_var        = tk.StringVar(value="—")
        self.loss_var         = tk.StringVar(value="—")
        self.acc_var          = tk.StringVar(value="—")

        self._build_ui()
        self.after(150, self._flush)

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        outer = ttk.Frame(self, padding=14)
        outer.pack(fill="both", expand=True)

        # Title
        ttk.Label(
            outer,
            text="🏥  Hospital Federated Training Client",
            font=("Segoe UI", 14, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

        # ── Server Config ─────────────────────────────────────────────────
        srv = ttk.LabelFrame(outer, text="Server", padding=8)
        srv.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        srv.columnconfigure(1, weight=1)
        ttk.Label(srv, text="Server URL:").grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(srv, textvariable=self.server_url).grid(row=0, column=1, sticky="ew")

        # ── Hospital Identity ─────────────────────────────────────────────
        auth = ttk.LabelFrame(outer, text="Hospital Identity", padding=8)
        auth.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        auth.columnconfigure(1, weight=1)

        ttk.Label(auth, text="Hospital Name:").grid(
            row=0, column=0, sticky="w", padx=(0, 6), pady=3
        )
        ttk.Entry(auth, textvariable=self.hospital_name).grid(
            row=0, column=1, sticky="ew", pady=3, padx=(0, 6)
        )
        btn_row = ttk.Frame(auth)
        btn_row.grid(row=0, column=2)
        self.reg_btn = ttk.Button(btn_row, text="Register", command=self._do_register, width=9)
        self.reg_btn.pack(side="left", padx=(0, 4))
        self.login_btn = ttk.Button(btn_row, text="Login", command=self._do_login, width=9)
        self.login_btn.pack(side="left")

        ttk.Label(auth, text="Password:").grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=3
        )
        ttk.Entry(auth, textvariable=self.password, show="*").grid(
            row=1, column=1, sticky="ew", pady=3, padx=(0, 6)
        )

        self._auth_lbl = ttk.Label(
            auth, textvariable=self.auth_status_var,
            foreground="gray", font=("Segoe UI", 9),
        )
        self._auth_lbl.grid(row=2, column=0, columnspan=3, sticky="w", pady=(3, 0))

        ttk.Label(auth, text="Active Token:").grid(
            row=3, column=0, sticky="w", padx=(0, 6), pady=(2, 4)
        )
        tok_entry = ttk.Entry(
            auth, textvariable=self.token_display_var,
            state="readonly", font=("Consolas", 8),
        )
        tok_entry.grid(row=3, column=1, sticky="ew", pady=(2, 4), padx=(0, 6))
        ttk.Label(
            auth, text="(used as Bearer token on submit)",
            foreground="gray", font=("Segoe UI", 8),
        ).grid(row=3, column=2, sticky="w")

        # ── Training Config ───────────────────────────────────────────────
        cfg = ttk.LabelFrame(outer, text="Training Configuration", padding=8)
        cfg.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        cfg.columnconfigure(1, weight=1)

        ttk.Label(cfg, text="Dataset Folder:").grid(
            row=0, column=0, sticky="w", padx=(0, 6), pady=3
        )
        ttk.Entry(cfg, textvariable=self.dataset_dir).grid(
            row=0, column=1, sticky="ew", pady=3, padx=(0, 6)
        )
        ttk.Button(cfg, text="Browse", command=self._pick_dataset, width=7).grid(
            row=0, column=2, pady=3
        )

        ttk.Label(cfg, text="Round ID:").grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=3
        )
        ttk.Entry(cfg, textvariable=self.round_id, width=20).grid(
            row=1, column=1, sticky="w", pady=3
        )

        params = ttk.Frame(cfg)
        params.grid(row=2, column=0, columnspan=3, sticky="w", pady=3)
        ttk.Label(params, text="Epochs:").pack(side="left")
        ttk.Spinbox(params, from_=1, to=20, textvariable=self.epochs, width=5).pack(
            side="left", padx=(3, 12)
        )
        ttk.Label(params, text="Batch:").pack(side="left")
        ttk.Spinbox(params, from_=1, to=128, textvariable=self.batch_size, width=5).pack(
            side="left", padx=(3, 12)
        )
        ttk.Checkbutton(params, text="Force CPU", variable=self.force_cpu).pack(side="left")

        # ── Train Button ──────────────────────────────────────────────────
        self.start_btn = ttk.Button(
            outer,
            text="⚡  Train Locally and Submit Weights",
            command=self._start_training,
        )
        self.start_btn.grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(0, 8), ipady=5
        )

        # ── Progress ──────────────────────────────────────────────────────
        prog = ttk.LabelFrame(outer, text="Training Progress", padding=8)
        prog.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        prog.columnconfigure(3, weight=1)

        ttk.Label(prog, text="Status:").grid(row=0, column=0, sticky="w")
        ttk.Label(
            prog, textvariable=self.train_status_var,
            foreground="#0f766e", font=("Segoe UI", 9, "bold"),
        ).grid(row=0, column=1, columnspan=3, sticky="w", padx=(4, 0))

        self.pbar = ttk.Progressbar(
            prog, variable=self.progress_var, maximum=100, mode="determinate"
        )
        self.pbar.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(6, 6))

        stats = ttk.Frame(prog)
        stats.grid(row=2, column=0, columnspan=4, sticky="w")
        for lbl, var in [
            ("Epoch:",    self.epoch_var),
            ("Batch:",    self.batch_var),
            ("Loss:",     self.loss_var),
            ("Accuracy:", self.acc_var),
        ]:
            ttk.Label(stats, text=lbl).pack(side="left", padx=(0, 2))
            ttk.Label(
                stats, textvariable=var,
                foreground="#0f766e", font=("Segoe UI", 9, "bold"), width=10,
            ).pack(side="left", padx=(0, 12))

        # ── Log ───────────────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(outer, text="Training Log", padding=6)
        log_frame.grid(row=6, column=0, columnspan=2, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_box = tk.Text(
            log_frame, height=10, wrap="word", state="disabled",
            font=("Consolas", 9), relief="flat",
        )
        sb = ttk.Scrollbar(log_frame, command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=sb.set)
        self.log_box.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(6, weight=1)

    # ── Auth helpers ───────────────────────────────────────────────────────

    def _set_auth(self, text: str, color: str = "gray"):
        self.auth_status_var.set(text)
        self._auth_lbl.configure(foreground=color)

    def _update_token_display(self, token: str):
        """Show a masked preview: first 8 chars + '...' + last 4 chars."""
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
                    f"[Auth] Registration successful. Token stored (Bearer auth enabled)."
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
                    f"[Auth] Login successful. Token stored (Bearer auth enabled)."
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
        current_epoch_ref = [1]          # mutable container for closure

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
        # Drain log queue
        while not self.log_queue.empty():
            self.append_log(self.log_queue.get())

        # Drain progress queue (only process the latest update per drain cycle)
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
