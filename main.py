import random
import re
import tkinter as tk
import sqlite3
from tkinter import ttk
import pandas as pd
from typing import Literal, Any

from screeninfo import get_monitors


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
        for i in range(self.data_size + 1):
            self.unique_values[i] = set()
        self.train_vectors = dict()  # All train vectors separated by decision attributes

        for v in train:
            for i in range(self.data_size + 1):
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
                atr_count = len(list(filter(lambda x: str(x[i]) == str(to_classify[i]), self.train_vectors[_class])))
                if atr_count == 0:  # smoothening
                    class_prob *= (atr_count + 1) / (class_size + len(self.unique_values[i]))
                else:
                    class_prob *= atr_count / class_size
            all_class_prob[_class] = class_prob

        return max(all_class_prob, key=all_class_prob.get)

    def classify_all(self, to_classify: list[list]) -> list:
        return [self.classify(a) for a in to_classify]

    def test(self, test: list[list]) -> tuple[int, float, dict]:
        for v in test:
            if len(v) != self.data_size + 1:
                raise ValueError("Incorrect test vector size.")
        # dict of dicts of zeros with size class_num*class_num
        # Rows are real classes, columns are classes counted by naive Bayes
        # Example:
        #   a b c
        # a 0 0 0
        # b 0 0 0
        # c 0 0 0
        confusion_matrix = {c: {c: 0 for c in self.train_vectors.keys()} for c in self.train_vectors.keys()}
        answers = {repr(v): self.classify(v) for v in test}
        for vector, _class in answers.items():
            try:
                confusion_matrix[eval(vector)[-1]][_class] += 1
            except KeyError:
                print(f'Unknown class "{_class}" found')

        correct = 0
        for c in self.train_vectors.keys():
            correct += confusion_matrix[c][c]
        correct_percent = correct / len(test)
        return correct, correct_percent, confusion_matrix


class DBController:

    def connect(self) -> sqlite3.dbapi2.Connection:
        return sqlite3.connect(self.database)

    def __init__(self, database: str):
        self.database = database
        query = "CREATE TABLE IF NOT EXISTS headers ( id INTEGER PRIMARY KEY," \
                "table_name TEXT," \
                "header_name TEXT)"
        connection = self.connect()
        connection.execute(query)
        connection.close()

    def save(self, model: Bayes, headers: list, table_name):
        query = f"DROP TABLE IF EXISTS '{table_name}'"
        connection = self.connect()
        connection.execute(query)
        connection.close()
        connection = self.connect()
        query = "DELETE FROM headers WHERE table_name = ?"
        connection.execute(query, [table_name])

        query = f"CREATE TABLE '{table_name}' ( id INTEGER PRIMARY KEY"
        for c in headers:
            query += f", '{c}' TEXT"
        query += " )"
        connection.execute(query)

        query = "INSERT INTO headers (table_name, header_name) VALUES (?, ?)"
        for h in headers:
            connection.execute(query, [table_name, h])

        query = f"INSERT INTO '{table_name}' ("
        for h in headers[:-1]:
            query += f"'{h}', "
        query += f"{headers[-1]}) VALUES ("
        for i in range(len(headers) - 1):
            query += "?, "
        query += "?)"

        for v in model.all_train_vectors:
            connection.execute(query, v)
        connection.commit()
        connection.close()

    def read_model(self, table_name: str) -> list[list[str]]:
        query = f"SELECT * FROM '{table_name}'"
        connection = self.connect()
        res = connection.execute(query)
        vecs = res.fetchall()
        connection.close()
        vectors = []
        for v in vecs:
            vectors.append(list(v[1:]))
        return vectors

    def read_headers(self, table_name: str) -> list[str]:
        query = "SELECT header_name FROM headers WHERE table_name = ?"
        connection = self.connect()
        res = connection.execute(query, [table_name])
        vecs = res.fetchall()
        connection.close()
        vectors = []
        for v in vecs:
            vectors.append(*v)
        return vectors


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
                 width: int = 100,
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


class Root(tk.Tk):

    def __init__(self):
        super().__init__()
        self.model = None
        self.headers = []
        self.database_controller = DBController("model.db")
        self.title("Classificator")
        screen_width = get_monitors()[0].width

        load_from_file_button = tk.Button(self, text="Load")
        create_new_model_button = tk.Button(self, text="Create new")

        def load_from_file():
            load_window = tk.Toplevel(self)
            load_window.title("Load")

            save_name_entry = EntryFrame(load_window, label="Save name")
            save_name_entry.pack()

            def load():
                save_name = save_name_entry.get()

                if save_name == "":
                    save_name = "model"

                if re.compile("\\W").findall(save_name):
                    print("Incorrect save name.")
                    save_name_entry.entry.config(background="Red")
                    return

                self.model = Bayes(self.database_controller.read_model(save_name))
                self.train_vectors = self.model.all_train_vectors
                self.headers = self.database_controller.read_headers(save_name)
                load_window.destroy()
                load_from_file_button.destroy()
                create_new_model_button.destroy()

                setup()

            load_model_button = tk.Button(load_window, text="Load")
            load_model_button.config(command=load)
            load_model_button.pack()

        def create_new_model():
            new_model_enter_window = tk.Toplevel(self)
            new_model_enter_window.title("New model")

            url_entry = EntryFrame(new_model_enter_window, label="URL")
            url_entry.pack()

            headers_entry = EntryFrame(new_model_enter_window, label="Columns")
            headers_entry.pack()

            train_row_number_entry = EntryFrame(new_model_enter_window, label="Train row number")
            train_row_number_entry.pack()

            test_row_number_entry = EntryFrame(new_model_enter_window, label="Test row number")
            test_row_number_entry.pack()

            info = tk.Text(new_model_enter_window)
            info.insert(tk.END, '''Columns must be separated with punctuation character.
Column count must be the same as number of attributes.
Columns must be unique.
First attribute is a decision attribute.
Default data are car.data for url and lines from car.headers for headers.
Default data are used when url is empty.''')
            info.config(state=tk.DISABLED, height=6)
            info.pack(anchor="e", padx=5, pady=5)

            def create():
                load_from_file_button.destroy()
                create_new_model_button.destroy()

                url = url_entry.get()

                train_row_number = train_row_number_entry.get()
                test_row_number = test_row_number_entry.get()

                if train_row_number == "":
                    train_row_number = 50
                else:
                    train_row_number = int(train_row_number)

                if test_row_number == "":
                    # 25% of all data read
                    test_row_number = int(0.33 * train_row_number)
                else:
                    test_row_number = int(test_row_number)

                if url == "":
                    url = "car.data"
                    self.headers = [line.removesuffix("\n") for line in open("car.headers", mode='r')]
                else:
                    self.headers = re.findall(r'\w+', headers_entry.get())

                for h in self.headers:
                    if self.headers.count(h) != 1:
                        raise ValueError("Columns must be unique")
                data = pd.read_csv(url, header=None)

                if len(data.columns) != len(self.headers):
                    raise ValueError("Column count must be the same as number of attributes.")

                column_rename = {i: self.headers[i] for i in data.columns.to_list()}
                data.rename(columns=column_rename)

                self.train_vectors = []
                test_vectors = []
                all_vectors = [list(v) for v in data.values]
                random.shuffle(all_vectors)
                all_vectors = all_vectors[:train_row_number + test_row_number]
                for i in range(len(all_vectors)):
                    if i < train_row_number:
                        self.train_vectors.append(all_vectors[i])
                    else:
                        test_vectors.append(all_vectors[i])

                self.model = Bayes(self.train_vectors)

                test_results = self.model.test(test_vectors)
                print("Test results:\n"
                      f"\tcorrect: {test_results[0]}({int(test_results[1] * 100)}%)\n"
                      "Confusion matrix:")

                for k in test_results[2].keys():
                    print(*[f" {v}" for v in test_results[2][k].values()], sep="\t |")
                    print("---------" * len(test_results))

                new_model_enter_window.destroy()

                setup()

            create_button = tk.Button(new_model_enter_window, text="Create", command=create)
            create_button.pack()

        def setup():

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

            def add_new_train_vector():
                vector_enter_window = tk.Toplevel(self)
                vector_enter_window.title("Vector enter")

                entry_frames = []
                for _h in self.headers:
                    frame = EntryFrame(vector_enter_window, label=_h)
                    entry_frames.append(frame)
                    frame.pack()

                def add_new():
                    vec = [ef.get() for ef in entry_frames]
                    vector_enter_window.destroy()
                    if vec not in self.model.all_train_vectors:
                        self.model.train_vectors[vec[-1]].append(vec[:-1])
                        self.model.all_train_vectors.append(vec)
                    else:
                        print("Vector already in train vectors")

                add_new_button = tk.Button(vector_enter_window, text="Add new", command=add_new)
                add_new_button.pack(side="left", pady=5, padx=5)

            self.add_new_vector_button = tk.Button(self, text="Add new", command=add_new_train_vector)
            self.add_new_vector_button.pack(side="left", padx=5, pady=5)

            def save_model():
                save_window = tk.Toplevel(self)
                save_window.title("Save")

                save_name_entry = EntryFrame(save_window, label="Save name")
                save_name_entry.pack()

                def save():
                    save_name = save_name_entry.get()
                    if save_name == "":
                        save_name = "model"
                    if re.compile("\\W").findall(save_name):
                        print("Incorrect save name.")
                        save_name_entry.entry.config(background="Red")
                        return
                    self.database_controller.save(self.model, self.headers, save_name)
                    save_window.destroy()

                save_button = tk.Button(save_window, text="Save", command=save)
                save_button.pack(pady=5, padx=5)

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
                for v in self.model.all_train_vectors:
                    table.insert("", "end", values=v)

                table.pack(side="left", fill="both", expand=1)
                table_scroll_bar.pack(side="right", fill="both", expand=0)

            def show_on_canvas():
                print("Not implemented yet")

            self.show_on_table_button = tk.Button(self, text="Show table", command=show_on_table)
            self.show_on_table_button.pack(side="left", padx=5, pady=5)

            self.show_on_canvas_button = tk.Button(self, text="Show canvas", command=show_on_canvas)
            self.show_on_canvas_button.pack(side="left", padx=5, pady=5)

            self.classify_new_button = tk.Button(self, text="Classify new", command=enter_to_classify)
            self.classify_new_button.pack(side="right", padx=5, pady=5)

            self.save_button = tk.Button(self, text="Save", command=save_model)
            self.save_button.pack(side="right", padx=5, pady=5)

        load_from_file_button.config(command=load_from_file)
        load_from_file_button.pack(side="left", padx=20, pady=5)

        create_new_model_button.config(command=create_new_model)
        create_new_model_button.pack(side="right", pady=5, padx=20)

        self.resizable(width=False, height=False)
        self.mainloop()


Root()
