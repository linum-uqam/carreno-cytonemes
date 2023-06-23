import tkinter as tk
import tkinter.filedialog
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
        self.about_msg = "TODO"

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
    
    def show(self):
        self.root.mainloop()


class MenuWindow(Window):
    def __init__(self):
        super().__init__(tk.Tk(), "Carreno Project", 500, 50, False)

        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        segmentation_menu = tk.Menu(menubar, tearoff=False)
        
        self._add_2_menu(segmentation_menu,
                         ["Threshold", "U-Net 2D", "U-Net 3D"],
                         [lambda: print('Threshold'),
                          lambda: PipelineWindow(tk.Toplevel(self.root), title="Segmentation U-Net 2D"),
                          MenuWindow])

        menubar.add_cascade(
            label="Segmentation",
            menu=segmentation_menu
        )

        menubar.add_cascade(
            label="Analysis",
            menu=segmentation_menu
        )

        menubar.add_cascade(
            label="Training",
            menu=segmentation_menu
        )
        """
        # Dropdown menu options
        options = [
            "Threshold",
            "U-Net 2D",
            "U-Net 3D"
        ]
       
        clicked = tk.StringVar()
        # initial menu text
        clicked.set("Segmentation")
        print(clicked.get())
        # Create Dropdown menu
        drop = tk.OptionMenu( self.root , clicked , *options )
        drop.pack()

        # Change the label text
        def show():
            print(clicked.get())

        # Create button, it will change label text
        button = tk.Button( self.root , text = "click Me" , command = show ).pack()
        """


class PipelineWindow(Window):
    def __init__(self, src, title, width=500, heigth=500):
        super().__init__(src, title=title, width=width, height=heigth, resizable=True)
        self.form_entries = {}

        cur_width, cur_heigth = self.root.winfo_width(), self.root.winfo_height()
        pad = 0.01
        self.padx, self.pady = int(pad * cur_width), int(pad * cur_heigth)
        self.pad_edge = min(self.padx, self.pady) * 5

        self._add_entry("Some text", "txt", None, None)
        self._add_entry("A file", "file", None, None, command=lambda field:self._get_file_path_in_entry(field, [('TIFF Files', '.tif .tiff')]))

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

        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=0)

    def _add_entry(self, label, entry_type, validation, error_msg, command=None, default_value=None):
        entry, getter = None, None
        row = len(self.form_entries)
        pad_top = self.pady if len(self.form_entries) > 0 else self.pad_edge
        pady = (pad_top, self.pady)

        lb = tk.Label(self.root, text=label)
        lb.grid(row=row, column=0, sticky="e", padx=(self.pad_edge, self.padx), pady=pady)

        if entry_type == "txt":
            entry = tk.Entry(self.root)
            getter = entry.get
            entry.grid(row=row, column=1, columnspan=2, sticky = "ew", padx=(self.padx, self.pad_edge), pady=pady)
        elif entry_type == "file":
            entry = tk.Entry(self.root)
            getter = entry.get
            entry.grid(row=row, column=1, sticky = "ew", padx=(self.padx, 0), pady=pady)

            file_btn = tk.Button(self.root, text='/', command=lambda:command(entry))
            file_btn.grid(row=row, column=2, sticky="e", padx=(0, self.pad_edge), pady=pady)
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

    def process(self):
        """
        Calls desired pipeline to run in a different thread.
        """
        raise NotImplementedError

        
class ThresholdPipelineWindow(PipelineWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, title="Threshold Pipeline")
        self.about_msg = """Parameters :
        - TODO
        - ...
        Returns :
        - TODO
        """

        # fill the form
        self._add_entry()
    
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
