import os
import subprocess
import tkinter as tk
from tkinter import messagebox
import socket
import threading
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
WEBAPP_SCRIPT = os.path.join(SCRIPT_DIR, "run_webapp.sh")

class AppLauncher(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Welding Simulator Launcher")
        self.geometry("600x450")
        self.resizable(True, True)

        # Main frame
        frame = tk.Frame(self, padx=40, pady=40)
        frame.pack(expand=True, fill=tk.BOTH)

        # Title
        tk.Label(frame, text="Welding Simulator", font=("Helvetica", 28, "bold")).pack(pady=(0, 10))
        tk.Label(frame, text="Isaac Sim Backend & Frontend", font=("Helvetica", 16)).pack(pady=(0, 30))

        # Status Label
        self.status_var = tk.StringVar()
        self.status_var.set("Status: Stopped")
        self.status_label = tk.Label(frame, textvariable=self.status_var, fg="red", font=("Helvetica", 16, "bold"))
        self.status_label.pack(pady=(0, 25))

        # Buttons
        self.start_btn = tk.Button(frame, text="Start Backend Server", command=self.start_backend, width=30, height=2, bg="#4CAF50", fg="white", font=("Helvetica", 14, "bold"))
        self.start_btn.pack(pady=10)

        self.open_btn = tk.Button(frame, text="Open Web App", command=self.open_frontend, width=30, height=2, bg="#2196F3", fg="white", font=("Helvetica", 14, "bold"), state=tk.DISABLED)
        self.open_btn.pack(pady=10)

        self.stop_btn = tk.Button(frame, text="Stop Backend Server", command=self.stop_backend, width=30, height=2, bg="#F44336", fg="white", font=("Helvetica", 14, "bold"), state=tk.DISABLED)
        self.stop_btn.pack(pady=10)

        self.backend_process = None

        # Check if already running
        if self._is_port_open():
            self._on_server_ready()

    def _is_port_open(self, host="localhost", port=8000):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect((host, port))
                return True
            except (socket.timeout, ConnectionRefusedError):
                return False

    def start_backend(self):
        if self._is_port_open():
            self._on_server_ready()
            return

        self.start_btn.config(state=tk.DISABLED)
        self.status_var.set("Status: Starting...")
        self.status_label.config(fg="orange")
        self.update()

        try:
            # We open it in a new session so it detaches properly from the launcher
            self.backend_process = subprocess.Popen(
                ["bash", WEBAPP_SCRIPT],
                cwd=ROOT_DIR,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            
            # Wait a moment for uvicorn to bind to port 8000
            threading.Thread(target=self._check_server_ready, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start backend:\n{e}")
            self.reset_ui()

    def _check_server_ready(self):
        max_retries = 15
        for _ in range(max_retries):
            if self._is_port_open():
                self.after(0, self._on_server_ready)
                return
            time.sleep(1)
        
        # If we get here, it didn't start in time
        def show_err():
            messagebox.showerror("Error", "Backend took too long to start. Check terminal or logs.")
        self.after(0, show_err)
        self.after(0, self.reset_ui)

    def _on_server_ready(self):
        self.status_var.set("Status: Running on port 8000")
        self.status_label.config(fg="green")
        self.open_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)

    def open_frontend(self):
        import webbrowser
        webbrowser.open("http://localhost:8000/")

    def stop_backend(self):
        self.status_var.set("Status: Stopping...")
        self.status_label.config(fg="orange")
        self.update()
        
        try:
            # Uvicorn with --reload spawns multiple processes. This safely stops all of them.
            subprocess.run("pkill -9 -f uvicorn", shell=True)
            subprocess.run("pkill -9 -f welding_simulator.api.main:app", shell=True)
            subprocess.run("pkill -9 -f run_webapp.sh", shell=True)
            # Give it a second to release the port
            time.sleep(1.5)
        except Exception as e:
            print(f"Error stopping: {e}")
            
        self.backend_process = None
        self.reset_ui()

    def reset_ui(self):
        self.status_var.set("Status: Stopped")
        self.status_label.config(fg="red")
        self.start_btn.config(state=tk.NORMAL)
        self.open_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)

    def on_closing(self):
        if self._is_port_open():
            if messagebox.askokcancel("Quit", "Backend is still running. Stop it and quit?"):
                self.stop_backend()
                self.destroy()
        else:
            self.destroy()

if __name__ == "__main__":
    app = AppLauncher()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
