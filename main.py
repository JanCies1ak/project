import re
import string
import tkinter as tk
import sqlite3
from tkinter import ttk
import pandas as pd
from typing import Literal, List, Any

from screeninfo import get_monitors


# TODO: documentation


# TODO: database controller
class DBController:
    def __init__(self, database: str, columns: str):
        self.database = database


class EntryFrame(tk.Frame):
    """
    Frame with label and entry.
    Used to set label to the left of entry and to reduce code length.
    """
    entry = None
    label = None

    def __init__(self,
                 master: tk.Misc | None,
                 *,
                 label: str,
                 width: int = 60,
                 borderwidth: str | float = 0,
                 relief: Literal["raised", "sunken", "flat", "ridge", "solid", "groove"] = "flat"):
        super(EntryFrame, self).__init__(master, borderwidth=borderwidth, relief=relief)
        self.entry = tk.Entry(self, width=width)
        self.label = tk.Label(self, text=label)

    def pack(self):
        self.label.pack(side="left", padx=5, pady=5)
        self.entry.pack(side="right", padx=5, pady=5, before=self.label)
        super(EntryFrame, self).pack(anchor="e")

    def get(self):
        return self.entry.get()


class Bayes:
    def __init__(self, train: list[list]):
        if len(train) == 0:
            raise ValueError("At least 1 vector for train set expected.")

        self.data_size = len(train[0])
        for v in train:
            if self.data_size != len(v):
                raise ValueError("All training vectors must be the same size.")
        self.data_size -= 1  # No need to include decision attribute anymore

        self.all_train_vectors = train  # All train vectors
        self.unique_values = dict()  # Unique values for each attribute except for decision attribute
        for i in range(self.data_size):
            self.unique_values[i] = set()
        self.train_vectors = dict()  # All train vectors separated by decision attributes

        for v in train:
            for i in range(self.data_size):
                self.unique_values[i].add(v[i])
            if v[-1] not in self.train_vectors.keys():
                self.train_vectors[v[-1]] = []
            self.train_vectors[v[-1]].append(v[:len(v) - 1])

    def classify(self, to_classify: list) -> Any:
        if len(to_classify) != self.data_size and (len(to_classify)) != self.data_size + 1:
            # to_classify must be the same size the train vectors are,
            # or one less (if they don't have decision attribute)
            raise ValueError("Incorrect size.")

        for v in self.all_train_vectors:
            if v[:self.data_size] == to_classify[:self.data_size]:
                return v[-1]

        all_class_prob = dict()
        for _class in self.train_vectors.keys():
            class_size = len(self.train_vectors[_class])
            class_prob = class_size / len(self.all_train_vectors)
            for i in range(self.data_size):
                atr_count = len(list(filter(lambda x: str(x) == str(to_classify[i]), self.train_vectors[_class])))
                if atr_count == 0:  # smoothening
                    class_prob *= (atr_count + 1) / (class_size + len(self.unique_values[i]))
                else:
                    class_prob *= atr_count / class_size
            all_class_prob[_class] = class_prob

        return max(all_class_prob, key=all_class_prob.get)

    def classify_all(self, to_classify: list[list]) -> list:
        return [self.classify(a) for a in to_classify]


# TODO:
#  new window to classify data,
#  rebuild button (?)
class Root(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Classificator")
        screen_width = get_monitors()[0].width
        screen_height = get_monitors()[0].height

        data_enter_window = tk.Toplevel(self)
        data_enter_window.title("Data enter")

        url_entry = EntryFrame(data_enter_window, label="URL:", width=100)
        url_entry.pack()

        headers_entry = EntryFrame(data_enter_window, label="Columns", width=100)
        headers_entry.pack()

        info = tk.Text(data_enter_window)
        info.insert(tk.END, '''Columns must be separated with punctuation character.
Column count must be the same as number of attributes.
Columns must be unique.
First attribute is a decision attribute.
Default data are car.data for url and lines from car.headers for headers.
Default data are used when url is empty.''')
        info.config(state=tk.DISABLED, height=6)
        info.pack(anchor="e", padx=5, pady=5)
        self.headers = []
        self.data = pd.DataFrame()

        def show_on_table():
            tree_view_window = tk.Toplevel(self)
            tree_view_window.title("Table data")

            table = ttk.Treeview(tree_view_window, columns=self.headers)
            table_scroll_bar = tk.Scrollbar(tree_view_window, command=table.yview)
            table.config(yscrollcommand=table_scroll_bar)

            table.column("# 0", width=0)
            for h in self.headers:
                table.heading(h, text=h)
            for i in range(1, len(self.headers) + 1):
                table.column(f"# {i}", width=screen_width // (2 * len(self.headers)))
            for v in self.data.values:
                table.insert("", "end", values=list(v))

            table.pack(side="left", fill="both", expand=1)
            table_scroll_bar.pack(side="right", fill="both", expand=0)

        def show_on_canvas():  # TODO:(optional) show on canvas
            print("Not implemented yet")

        def load_data():
            url = url_entry.get()
            if url == "":
                url = "car.data"
                self.headers = [line.removesuffix("\n") for line in open("car.headers", mode='r')]
            else:
                self.headers = re.findall(r'\w+', headers_entry.get())

            # Not in else because anyone can change data in car files
            for h in self.headers:
                if self.headers.count(h) != 1:
                    raise ValueError("Columns must be unique")
            self.data = pd.read_csv(url, header=None)

            if len(self.data.columns) != len(self.headers):
                raise ValueError("Column count must be the same as number of attributes.")
            else:
                data_enter_window.destroy()
                column_rename = {i: self.headers[i] for i in self.data.columns.to_list()}

                # TODO: train and test data separation, probably shuffle and take 75-85%
                self.model = Bayes([list(v) for v in self.data.values])

                self.geometry(f"{screen_width // 2}x{int(screen_height / 1.6)}")
                self.data = self.data.rename(columns=column_rename)

                self.show_on_table_button = tk.Button(self, text="Show table", command=show_on_table)
                self.show_on_table_button.pack(side="left")

                self.show_on_canvas_button = tk.Button(self, text="Show canvas", command=show_on_canvas)
                self.show_on_canvas_button.pack(side="left")

                def enter_to_classify():
                    vector_enter_window = tk.Toplevel(self)
                    vector_enter_window.title("Vector enter")

                    entry_frames = []
                    for _h in self.headers[:-1]:
                        frame = EntryFrame(vector_enter_window, label=_h)
                        entry_frames.append(frame)
                        frame.pack()

                    def classify():
                        vec = [ef.get() for ef in entry_frames]
                        vector_enter_window.destroy()
                        print(f"Vector: {vec}\nAnswer: {self.model.classify(vec)}")

                    classify_button = tk.Button(vector_enter_window, text="Classify", command=classify)
                    classify_button.pack(side="left")

                    def show_enter_info():  # TODO: show info in new window with only text.
                        print("Not implemented yet. Not important.")

                    info_button = tk.Button(vector_enter_window, text="Info", command=show_enter_info)
                    info_button.pack(side="right")

                self.classify_new_button = tk.Button(self, text="Classify new", command=enter_to_classify)
                self.classify_new_button.pack(side="right")

        load_button = tk.Button(data_enter_window, text="Load", command=load_data, width=20)
        load_button.pack(anchor='w', pady=10, padx=10)

        self.resizable(width=False, height=False)
        self.mainloop()


Root()
