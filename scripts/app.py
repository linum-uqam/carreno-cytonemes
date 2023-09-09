import tkinter as tk
import tkinter.filedialog
import os
import numpy as np
import imageio
import threading
from carreno.pipeline import pipeline

class Window:
    def __init__(self, src, title="Window", width=100, height=100, resizable=False):
        self.root = src
        self.root.title(title)
        self.root.geometry("{}x{}".format(width, height))
        self.root.resizable(resizable, resizable)
        self.root.configure(bg='#dddddd')
        self.root.update_idletasks()
        self.btn_std_width = 10
        self.about_msg = "UNDEFINED"

    def _add_2_menu(self, menu, labels, commands):
        for lb, cmd in zip(labels, commands):
            menu.add_command(label=lb, command=cmd)

    def _set_txt_in_entry(self, entry, txt):
        entry.delete(0, tk.END)
        entry.insert(0, txt)

    def _get_file_path_in_entry(self, entry, filetypes=[]):
        # grab window to avoid losing focus when we are done choosing a file
        self.root.grab_set()
        filename = tk.filedialog.askopenfilename(filetypes=filetypes)
        self._set_txt_in_entry(entry, filename)
        self.root.grab_release()
    
    def _get_folder_path_in_entry(self, entry):
        # grab window to avoid losing focus when we are done choosing a file
        self.root.grab_set()
        filename = tk.filedialog.askdirectory()
        self._set_txt_in_entry(entry, filename)
        self.root.grab_release()
    
    def _start_thread_and_check_status(self, func, status_holder, idx):
        # https://stackoverflow.com/questions/73639645/how-to-wait-for-a-server-response-without-freezing-the-tkinter-gui
        def update_status():
            try:
                func()
                status_holder[idx] = 1  # done
            except Exception as e:
                status_holder[idx] = 2  # error
                print(e)
        thread = threading.Thread(update_status)
        thread.start()
        def _wait_for_completion():
            if (status_holder == 0).sum() == 0:
                return
            else:
                self.root.after(5, _wait_for_completion)
        _wait_for_completion()

    def show(self):
        self.root.mainloop()


class MenuWindow(Window):
    def __init__(self):
        super().__init__(tk.Tk(), "Carreno Project", 500, 50, False)
        
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        supported_pipelines = ["Threshold", "U-Net 2D", "U-Net 3D"]

        denoise_menu = tk.Menu(menubar, tearoff=False)
        self._add_2_menu(denoise_menu,
                         supported_pipelines[:1],
                         [lambda: DenoiseThresholdPipelineWindow(tk.Toplevel(self.root))])
        menubar.add_cascade(
            label="Denoise",
            menu=denoise_menu
        )

        segmentation_menu = tk.Menu(menubar, tearoff=False)
        self._add_2_menu(segmentation_menu,
                         supported_pipelines,
                         [lambda: ThresholdPipelineWindow(tk.Toplevel(self.root)),
                          lambda: PipelineWindow(tk.Toplevel(self.root), title="Segmentation U-Net 2D"),
                          MenuWindow])
        menubar.add_cascade(
            label="Segmentation",
            menu=segmentation_menu
        )

        analysis_menu = tk.Menu(menubar, tearoff=False)
        self._add_2_menu(analysis_menu,
                         ["DFS"],
                         [MenuWindow])
        menubar.add_cascade(
            label="Analysis",
            menu=analysis_menu
        )

        training_menu = tk.Menu(menubar, tearoff=False)
        self._add_2_menu(training_menu,
                         supported_pipelines[1:],
                         [MenuWindow]*2)
        menubar.add_cascade(
            label="Training",
            menu=training_menu
        )        


class PipelineWindow(Window):
    def __init__(self, src, title, width=500, heigth=500):
        super().__init__(src, title=title, width=width, height=heigth, resizable=True)
        self.form_entries = {}

        cur_width, cur_heigth = self.root.winfo_width(), self.root.winfo_height()
        pad = 0.01
        self.padx, self.pady = int(pad * cur_width), int(pad * cur_heigth)
        self.pad_edge = min(self.padx, self.pady) * 5

        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=0)

    def _add_entry(self, label, entry_type, validation=None, error_msg=None, param=None, default_value=None):
        entry, getter = None, None
        row = len(self.form_entries)
        pad_top = self.pady if len(self.form_entries) > 0 else self.pad_edge
        pady = (pad_top, self.pady)

        if not label is None:
            lb = tk.Label(self.root, text=label)
            lb.grid(row=row, column=0, sticky="e", padx=(self.pad_edge, self.padx), pady=pady)

        if entry_type == "txt":
            entry = tk.Entry(self.root)
            getter = entry.get
            if param is None:
                entry.grid(row=row, column=1, columnspan=2, sticky= "ew", padx=(self.padx, self.pad_edge), pady=pady)
            else:
                # add specification after input field
                entry.grid(row=row, column=1, sticky="ew", padx=(self.padx, self.padx), pady=pady)
                spec = tk.Label(self.root, text=param)
                spec.grid(row=row, column=2, sticky="w", padx=(self.padx, self.pad_edge), pady=pady)
        elif entry_type == "file":
            entry = tk.Entry(self.root)
            getter = entry.get
            if param is None:
                entry.grid(row=row, column=1, columnspan=2, sticky="ew", padx=(self.padx, 0), pady=pady)
            else:
                # btn to call function (file explorer)
                entry.grid(row=row, column=1, sticky="ew", padx=(self.padx, 0), pady=pady)
                file_btn = tk.Button(self.root, text=' / ', command=lambda:param(entry))
                file_btn.grid(row=row, column=2, sticky="w", padx=(0, self.pad_edge), pady=pady)
        elif entry_type == "select":
            raise NotImplementedError
        elif entry_type == "end":
            help_btn = tk.Button(self.root, text='Help', width=self.btn_std_width, command=self._output_description)
            help_btn.grid(row=len(self.form_entries),
                            column=0,
                            sticky="w",
                            padx=(self.pad_edge, self.padx),
                            pady=(self.pady, self.pad_edge))
            submit_btn = tk.Button(self.root, text='Submit', width=self.btn_std_width, command=self._output_entries)
            submit_btn.grid(row=len(self.form_entries),
                            column=1,
                            columnspan=2,
                            sticky="e",
                            padx=(self.padx, self.pad_edge),
                            pady=(self.pady, self.pad_edge))
            return
        else:
            raise NotImplementedError

        self.form_entries[label] = {
            "value":     getter,
            "validate":  validation,
            "error_msg": error_msg
        }
    
    def _output_description(self):
        popup = tk.Toplevel(self.root)
        desc = tk.Label(popup, text=self.about_msg)
        desc.grid(row=0, column=0, sticky="ewns", padx=self.padx, pady=self.pady)

    def _output_entries(self):
        """
        Prints forms entries
        """
        for k, v in self.form_entries.items():
            print(k, v)
    
    def validate(self):
        """
        Validates form entries before calling process.
        If there is an error, a popup is sent.
        """
        raise NotImplementedError

    def process(self):
        """
        Calls desired pipeline to run in a different thread.
        """
        raise NotImplementedError


class DenoiseThresholdPipelineWindow(PipelineWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, title="Threshold Pipeline")
        self.about_msg = """Parameters :
        - Volume directory : Path
            Folder containing TIFF volumes to denoise.
        - PSF file : Path
            TIFF file containing PSF for denoising. Will use non-local mean if empty.
        - Denoised directory : Path
            Folder where denoised volumes are outputted.
        Returns :
        - TODO
        """
        # fill the form
        self._add_entry(label="Volume directory",
                        entry_type="file",
                        validation=None,
                        error_msg=None,
                        param=lambda field:self._get_folder_path_in_entry(field),
                        default_value="input")
        self._add_entry(label="PSF file",
                        entry_type="file",
                        error_msg=None,
                        param=lambda field:self._get_file_path_in_entry(field, [('TIFF Files', '.tif .tiff')]),
                        default_value="psf.tif")
        self._add_entry(label="Denoised directory",
                        entry_type="file",
                        validation=None,
                        error_msg=None,
                        param=lambda field:self._get_folder_path_in_entry(field),
                        default_value="output")
        self._add_entry(label="Voxel size Z axis",
                        entry_type="txt",
                        validation=None,
                        error_msg=None,
                        param="microns",
                        default_value=1)
        self._add_entry(label="Voxel size Y axis",
                        entry_type="txt",
                        validation=None,
                        error_msg=None,
                        param="microns",
                        default_value=1)
        self._add_entry(label="Voxel size X axis",
                        entry_type="txt",
                        validation=None,
                        error_msg=None,
                        param="microns",
                        default_value=1)
        self._add_entry(None, "end")
    
    def validate(self):
        return super().validate()
    
    def process(self):
        pipeline = pipeline.Threshold(self.form_entries['distances']['value']())
        input_dir = self.form_entries["Volume directory"]["value"]
        files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        status = np.zeros((len(files)), dtype=np.uint8)
        for i in range(status.shape[0]):
            f = files[i]
            x = imageio.imread(f)
            self._start_thread_and_check_status(lambda:pipeline.denoise(x, psf=None, butterworth=None), status, i)
        return


class ThresholdPipelineWindow(PipelineWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, title="Threshold Pipeline")
        self.about_msg = """Parameters :
        - Volume directory : Path
            Folder containing TIFF volumes to segment.
        - PSF file : Path
            TIFF file containing PSF for denoising. Will use non-local mean if empty.
        - Segmentation directory : Path
            Folder where segmented volumes are outputted.
        Returns :
        - TODO
        """
        # fill the form
        self._add_entry(label="Volume directory",
                        entry_type="file",
                        validation=None,
                        error_msg=None,
                        param=lambda field:self._get_folder_path_in_entry(),
                        default_value="")
        self._add_entry(label="PSF file",
                        entry_type="file",
                        error_msg=None,
                        param=lambda field:self._get_file_path_in_entry(field, [('TIFF Files', '.tif .tiff')]),
                        default_value="")
        self._add_entry(None, "end")
    
    def validate(self):
        return super().validate()
    
    def process(self):
        pipeline = pipeline.Threshold(self.form_entries['distances']['value']())
        pipeline.segmentation()
        return super().process()


"""
root = tk.Tk()
root.geometry('500x500')
window = PipelineWindow(root)
root.mainloop()
"""
root = MenuWindow()
root.show()
