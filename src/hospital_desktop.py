import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from hospital_client import train_and_submit


class HospitalApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hospital Federated Training Client")
        self.geometry("760x560")
        self.minsize(680, 500)

        self.log_queue = queue.Queue()
        self.worker = None

        self.server_url = tk.StringVar(value="http://127.0.0.1:8000")
        self.hospital_name = tk.StringVar(value="Hospital 1")
        self.dataset_dir = tk.StringVar(value="")
        self.round_id = tk.StringVar(value="current")
        self.epochs = tk.IntVar(value=1)
        self.batch_size = tk.IntVar(value=16)
        self.force_cpu = tk.BooleanVar(value=False)

        self.create_widgets()
        self.after(200, self.flush_logs)

    def create_widgets(self):
        root = ttk.Frame(self, padding=18)
        root.pack(fill="both", expand=True)

        title = ttk.Label(root, text="Hospital Federated Training Client", font=("Segoe UI", 18, "bold"))
        title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 18))

        self.add_labeled_entry(root, "Server URL", self.server_url, 1)
        self.add_labeled_entry(root, "Hospital Name", self.hospital_name, 2)
        self.add_labeled_entry(root, "Round ID", self.round_id, 3)

        ttk.Label(root, text="Dataset Folder").grid(row=4, column=0, sticky="w", pady=8)
        ttk.Entry(root, textvariable=self.dataset_dir).grid(row=4, column=1, sticky="ew", pady=8)
        ttk.Button(root, text="Browse", command=self.pick_dataset).grid(row=4, column=2, sticky="ew", padx=(10, 0), pady=8)

        ttk.Label(root, text="Local Epochs").grid(row=5, column=0, sticky="w", pady=8)
        ttk.Spinbox(root, from_=1, to=20, textvariable=self.epochs, width=10).grid(row=5, column=1, sticky="w", pady=8)

        ttk.Label(root, text="Batch Size").grid(row=6, column=0, sticky="w", pady=8)
        ttk.Spinbox(root, from_=1, to=128, textvariable=self.batch_size, width=10).grid(row=6, column=1, sticky="w", pady=8)

        ttk.Checkbutton(root, text="Force CPU", variable=self.force_cpu).grid(row=7, column=1, sticky="w", pady=8)

        self.start_button = ttk.Button(root, text="Train Locally and Submit Weights", command=self.start_training)
        self.start_button.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(14, 12))

        ttk.Label(root, text="Training Log").grid(row=9, column=0, columnspan=3, sticky="w")
        self.log_box = tk.Text(root, height=14, wrap="word")
        self.log_box.grid(row=10, column=0, columnspan=3, sticky="nsew", pady=(6, 0))
        self.log_box.configure(state="disabled")

        root.columnconfigure(1, weight=1)
        root.rowconfigure(10, weight=1)

    def add_labeled_entry(self, parent, label, variable, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=8)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, columnspan=2, sticky="ew", pady=8)

    def pick_dataset(self):
        selected = filedialog.askdirectory(title="Select hospital dataset folder")
        if selected:
            self.dataset_dir.set(selected)

    def append_log(self, message):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{message}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def queue_log(self, message):
        self.log_queue.put(message)

    def flush_logs(self):
        while not self.log_queue.empty():
            self.append_log(self.log_queue.get())
        self.after(200, self.flush_logs)

    def validate_inputs(self):
        dataset_path = Path(self.dataset_dir.get())
        if not dataset_path.exists():
            messagebox.showerror("Missing Dataset", "Please select a valid hospital dataset folder.")
            return False
        if not (dataset_path / "NORMAL").exists() or not (dataset_path / "PNEUMONIA").exists():
            messagebox.showerror("Invalid Dataset", "Dataset folder must contain NORMAL and PNEUMONIA folders.")
            return False
        if not self.hospital_name.get().strip():
            messagebox.showerror("Missing Hospital", "Please enter a hospital name.")
            return False
        return True

    def start_training(self):
        if self.worker and self.worker.is_alive():
            return
        if not self.validate_inputs():
            return

        self.start_button.configure(state="disabled")
        self.queue_log("Starting local training workflow...")

        self.worker = threading.Thread(target=self.run_training, daemon=True)
        self.worker.start()

    def run_training(self):
        try:
            result = train_and_submit(
                server_url=self.server_url.get().strip(),
                hospital_name=self.hospital_name.get().strip(),
                dataset_dir=self.dataset_dir.get().strip(),
                round_id=self.round_id.get().strip() or "current",
                epochs=int(self.epochs.get()),
                batch_size=int(self.batch_size.get()),
                force_cpu=bool(self.force_cpu.get()),
                logger=self.queue_log,
            )
            self.queue_log(f"Completed: {result}")
            self.after(0, lambda: messagebox.showinfo("Submitted", "Local weights submitted successfully."))
        except Exception as exc:
            message = str(exc)
            self.queue_log(f"Error: {message}")
            self.after(0, lambda: messagebox.showerror("Training Failed", message))
        finally:
            self.after(0, lambda: self.start_button.configure(state="normal"))


if __name__ == "__main__":
    HospitalApp().mainloop()
