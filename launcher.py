"""Desktop wrapper; the original script is executed unchanged in a worker process."""
import os
import sys
from pathlib import Path

ROOT = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent

def worker():
    import runpy
    log = sys.argv[2]
    args = sys.argv[3:]
    with open(log, 'w', encoding='utf-8', buffering=1) as stream:
        sys.stdout = sys.stderr = stream
        sys.argv = [str(ROOT / 'elestrals_mask.py'), *args]
        try:
            runpy.run_path(sys.argv[0], run_name='__main__')
        except BaseException:
            import traceback
            traceback.print_exc()
            return 1
    return 0

def gui():
    import subprocess
    import tempfile
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    app = tk.Tk()
    app.title('Elestrals Mask')
    app.geometry('820x650')
    app.minsize(680, 540)
    card = tk.StringVar()
    output = tk.StringVar(value=str(ROOT / 'out'))
    mode = tk.StringVar(value='build')
    status = tk.StringVar(value='Choose a card image to begin.')
    modes = {
        'build': 'Build mask — character, text, border and bar',
        'frame': 'Frame preview — text, border and bar only',
        'sweep': 'Model sweep — all original models and variants',
        'fixed': 'Rebuild fixed frame — updates the shared frame file',
    }
    box = ttk.Frame(app, padding=20)
    box.pack(fill='both', expand=True)
    ttk.Label(box, text='Elestrals Mask', font=('Segoe UI', 22, 'bold')).pack(anchor='w')
    ttk.Label(box, text='Original segmentation workflow, with a desktop interface.').pack(anchor='w', pady=(0, 18))
    controls = []
    def field(label, variable, browse):
        ttk.Label(box, text=label).pack(anchor='w')
        row = ttk.Frame(box)
        row.pack(fill='x', pady=(4, 12))
        entry = ttk.Entry(row, textvariable=variable)
        entry.pack(side='left', fill='x', expand=True)
        button = ttk.Button(row, text='Browse…', command=browse)
        button.pack(side='left', padx=(8, 0))
        controls.extend([entry, button])
    def choose_card():
        value = filedialog.askopenfilename(filetypes=[('Images', '*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff'), ('All files', '*.*')])
        if value:
            card.set(value)
    def choose_out():
        value = filedialog.askdirectory()
        if value:
            output.set(value)
    field('Card image', card, choose_card)
    field('Output folder', output, choose_out)
    for key, label in modes.items():
        button = ttk.Radiobutton(box, text=label, variable=mode, value=key)
        button.pack(anchor='w', pady=2)
        controls.append(button)
    ttk.Label(box, text='Build uses the CPU and about 10 GB RAM. First use downloads a ~970 MB model.\nSweep downloads additional models and can take much longer.', wraplength=740).pack(anchor='w', pady=12)
    row = ttk.Frame(box)
    row.pack(fill='x')
    progress = ttk.Progressbar(box, mode='indeterminate')
    progress.pack(fill='x', pady=(12, 6))
    ttk.Label(box, textvariable=status).pack(anchor='w')
    log_view = tk.Text(box, height=10, wrap='word', state='disabled', font=('Consolas', 10))
    log_view.pack(fill='both', expand=True, pady=(8, 0))
    process = None
    log_path = None
    last_log = None
    def show_log(text):
        log_view.configure(state='normal')
        log_view.delete('1.0', 'end')
        log_view.insert('end', text)
        log_view.see('end')
        log_view.configure(state='disabled')
    def poll():
        nonlocal process, last_log
        text = Path(log_path).read_text(encoding='utf-8', errors='replace')
        if text != last_log:
            show_log(text)
            last_log = text
        result = process.poll()
        if result is None:
            app.after(400, poll)
            return
        process = None
        progress.stop()
        for control in controls:
            control.configure(state='normal')
        status.set('Finished. Open the output folder to view results.' if result == 0 else 'Failed. See the log below for details.')
        Path(log_path).unlink(missing_ok=True)
    def start():
        nonlocal process, log_path, last_log
        if not Path(card.get()).is_file():
            messagebox.showerror('Choose a card', 'Select an existing card image.')
            return
        if not output.get().strip():
            messagebox.showerror('Output folder', 'Choose an output folder.')
            return
        if mode.get() == 'fixed' and not messagebox.askyesno('Rebuild fixed frame', 'This replaces frames/ElestralsFullArt.png, which all cards use. Continue?'):
            return
        handle, log_path = tempfile.mkstemp(prefix='elestrals-', suffix='.log')
        os.close(handle)
        args = ['--worker', log_path, mode.get(), str(Path(card.get()).resolve()), '--out', str(Path(output.get()).resolve())]
        command = [sys.executable, *args] if getattr(sys, 'frozen', False) else [sys.executable, str(Path(__file__).resolve()), *args]
        try:
            process = subprocess.Popen(command, cwd=ROOT, creationflags=subprocess.CREATE_NO_WINDOW)
        except OSError as error:
            Path(log_path).unlink(missing_ok=True)
            messagebox.showerror('Unable to start', str(error))
            return
        last_log = None
        for control in controls:
            control.configure(state='disabled')
        status.set('Running… model downloads and processing may take several minutes.')
        progress.start()
        app.after(400, poll)
    run_button = ttk.Button(row, text='Run selected workflow', command=start)
    run_button.pack(side='left')
    controls.append(run_button)
    def open_output():
        folder = Path(output.get())
        if folder.is_dir():
            os.startfile(str(folder.resolve()))
        else:
            messagebox.showinfo('Output folder', 'The output folder will be created when you run a workflow.')
    ttk.Button(row, text='Open output folder', command=open_output).pack(side='left', padx=8)
    def close():
        if process is not None:
            messagebox.showinfo('Workflow running', 'Let the workflow finish before closing the window.')
            return
        app.destroy()
    app.protocol('WM_DELETE_WINDOW', close)
    app.mainloop()

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        raise SystemExit(worker())
    gui()
