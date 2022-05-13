import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
from pandastable import Table, TableModel
from sklearn.cluster import KMeans

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True


def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    top = Toplevel1 (root)
    root.mainloop()

w = None
def create_Toplevel1(rt, *args, **kwargs):
    '''Starting point when module is imported by another module.
       Correct form of call: 'create_Toplevel1(root, *args, **kwargs)' .'''
    global w, w_win, root
    #rt = root
    root = rt
    w = tk.Toplevel (root)
    top = Toplevel1 (w)
    return (w, top)

def destroy_Toplevel1():
    global w
    w.destroy()
    w = None

class Toplevel1:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#ececec' # Closest X11 color: 'gray92'
        font12 = "-family {Tahoma} -size 12"
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.',background=_bgcolor)
        self.style.configure('.',foreground=_fgcolor)
        self.style.configure('.',font="TkDefaultFont")
        self.style.map('.',background=
            [('selected', _compcolor), ('active',_ana2color)])

        top.geometry("1100x850+50+0")
        top.minsize(148, 1)
        top.maxsize(4112, 1055)
        top.resizable(0, 0)
        top.title("Assignment 3-sahdan chung")
        top.configure(background="#400040")
        top.configure(highlightbackground="#d9d9d9")
        top.configure(highlightcolor="black")

        self.Output_gui = tk.Label(top)
        self.Output_gui.place(relx=0.027, rely=0.182, height=619, width=750)
        self.Output_gui.configure(activebackground="#f9f9f9")
        self.Output_gui.configure(activeforeground="black")
        self.Output_gui.configure(background="#ffffff")
        self.Output_gui.configure(disabledforeground="#a3a3a3")
        self.Output_gui.configure(justify='left', anchor='nw')
        self.Output_gui.configure(foreground="#000000")
        self.Output_gui.configure(highlightbackground="#d9d9d9")
        self.Output_gui.configure(highlightcolor="black")
        self.Output_gui.configure(font=font12)

        self.Frame_output = tk.Frame(self.Output_gui)
        self.Frame_output.configure(relief='groove')
        self.Frame_output.configure(borderwidth="2")
        self.Frame_output.configure(relief="groove")
        self.Frame_output.configure(background="#d9d9d9")
        self.Frame_output.pack()

        self.TCombobox_csv = ttk.Combobox(top)
        self.TCombobox_csv.place(relx=0.191, rely=0.082, relheight=0.034
                , relwidth=0.297)
        self.value_list = ['Cars','Film',]
        self.TCombobox_csv.configure(values=self.value_list)
        self.TCombobox_csv.configure(font="-family Tahoma -size 14 -weight normal -slant roman -underline 0 -overstrike 0")
        self.TCombobox_csv.configure(justify='center')
        self.TCombobox_csv.configure(state='readonly')
        self.TCombobox_csv.configure(takefocus="")

        self.labelChoose = tk.Label(top)
        self.labelChoose.place(relx=0.027, rely=0.082, height=29, width=182)
        self.labelChoose.configure(activebackground="#f9f9f9")
        self.labelChoose.configure(activeforeground="black")
        self.labelChoose.configure(background="#400040")
        self.labelChoose.configure(disabledforeground="#a3a3a3")
        self.labelChoose.configure(font="-family {Segoe UI} -size 12 -weight bold -slant roman -underline 0 -overstrike 0")
        self.labelChoose.configure(foreground="#ffff80")
        self.labelChoose.configure(highlightbackground="#d9d9d9")
        self.labelChoose.configure(highlightcolor="black")
        self.labelChoose.configure(text='''Choose CSV file:''')

        self.labelAnalysis = tk.Label(top)
        self.labelAnalysis.place(relx=0.027, rely=0.129, height=29, width=182)
        self.labelAnalysis.configure(activebackground="#f9f9f9")
        self.labelAnalysis.configure(activeforeground="black")
        self.labelAnalysis.configure(background="#400040")
        self.labelAnalysis.configure(disabledforeground="#a3a3a3")
        self.labelAnalysis.configure(font="-family {Segoe UI} -size 12 -weight bold -slant roman -underline 0 -overstrike 0")
        self.labelAnalysis.configure(foreground="#ffff80")
        self.labelAnalysis.configure(highlightbackground="#d9d9d9")
        self.labelAnalysis.configure(highlightcolor="black")
        self.labelAnalysis.configure(text='''Analysis Type:''')

        self.TCombobox_analysis = ttk.Combobox(top)
        self.TCombobox_analysis.place(relx=0.191, rely=0.129, relheight=0.046
                , relwidth=0.518)
        self.value_list = ['a. Find and cap outliers from a series or dataframe column','b. Compute the mean squared error on a truth and predicted series','c. Find all the local maxima (or peaks) in a numeric series','d. Compute the autocorrelations of a numeric series','e. Compute the correlation of each row with the succeeding row','f. Compute and display Linear Regression analysis','g. Depends on the nature of data, compute and display any methods of Machine Learning Classification/Clustering/',]
        self.TCombobox_analysis.configure(values=self.value_list)
        self.TCombobox_analysis.configure(font="-family Tahoma -size 14 -weight normal -slant roman -underline 0 -overstrike 0")
        self.TCombobox_analysis.configure(state='readonly')
        self.TCombobox_analysis.configure(takefocus="")

        self.b_select = tk.Button(top)
        self.b_select.place(relx=0.732, rely=0.135, height=35, width=100)
        self.b_select.configure(activebackground="#ececec")
        self.b_select.configure(activeforeground="#000000")
        self.b_select.configure(background="#80ff80")
        self.b_select.configure(disabledforeground="#a3a3a3")
        self.b_select.configure(font="-family Tahoma -size 12 -weight normal -slant roman -underline 0 -overstrike 0")
        self.b_select.configure(foreground="#000000")
        self.b_select.configure(highlightbackground="#d9d9d9")
        self.b_select.configure(highlightcolor="black")
        self.b_select.configure(pady="0")
        self.b_select.configure(text='''select''')
        self.b_select.configure(command=self.select_question)

        self.Labelframe1 = tk.LabelFrame(top)
        self.Labelframe1.place(relx=0.718, rely=0.365, relheight=0.418
                , relwidth=0.264)
        self.Labelframe1.configure(relief='groove')
        self.Labelframe1.configure(font="-family Century -size 14 -weight bold -slant roman -underline 0 -overstrike 0")
        self.Labelframe1.configure(foreground="black")
        self.Labelframe1.configure(labelanchor="n")
        self.Labelframe1.configure(text='''Setting''')
        self.Labelframe1.configure(background="#c0c0c0")
        self.Labelframe1.configure(highlightbackground="#d9d9d9")
        self.Labelframe1.configure(highlightcolor="black")

        self.Labelframe2 = tk.LabelFrame(self.Labelframe1)
        self.Labelframe2.place(relx=0.069, rely=0.169, relheight=0.38
                , relwidth=0.862, bordermode='ignore')
        self.Labelframe2.configure(relief='groove')
        self.Labelframe2.configure(font="-family System -size 10 -weight bold -slant roman -underline 0 -overstrike 0")
        self.Labelframe2.configure(foreground="black")
        self.Labelframe2.configure(labelanchor="n")
        self.Labelframe2.configure(background="#d9d9d9")
        self.Labelframe2.configure(highlightbackground="#d9d9d9")
        self.Labelframe2.configure(highlightcolor="black")

        self.b_analysis = tk.Button(self.Labelframe1)
        self.b_analysis.place(relx=0.345, rely=0.700, height=40, width=96
                , bordermode='ignore')
        self.b_analysis.configure(activebackground="#ececec")
        self.b_analysis.configure(activeforeground="#000000")
        self.b_analysis.configure(background="#80ff80")
        self.b_analysis.configure(disabledforeground="#a3a3a3")
        self.b_analysis.configure(font="-family Tahoma -size 12 -weight normal -slant roman -underline 0 -overstrike 0")
        self.b_analysis.configure(foreground="#000000")
        self.b_analysis.configure(highlightbackground="#d9d9d9")
        self.b_analysis.configure(highlightcolor="black")
        self.b_analysis.configure(pady="0")
        self.b_analysis.configure(text='''analyse''')
        self.b_analysis.configure(command=self.analyse)

        self.title = tk.Label(top)
        self.title.place(relx=0.286, rely=0.012, height=42, width=470)
        self.title.configure(activebackground="#f9f9f9")
        self.title.configure(activeforeground="black")
        self.title.configure(background="#400040")
        self.title.configure(disabledforeground="#a3a3a3")
        self.title.configure(font="-family Righteous -size 17 -weight bold -slant roman -underline 1 -overstrike 0")
        self.title.configure(foreground="#dfdf00")
        self.title.configure(highlightbackground="#d9d9d9")
        self.title.configure(highlightcolor="black")
        self.title.configure(text='''Data Analysis and Visualization''')

        self.original = tk.Label(top)
        self.original.place(relx=-0.036, rely=0.929, height=42, width=250)
        self.original.configure(activebackground="#f9f9f9")
        self.original.configure(activeforeground="black")
        self.original.configure(background="#400040")
        self.original.configure(disabledforeground="#a3a3a3")
        self.original.configure(font="-family {Times New Roman} -size 8 -weight bold -slant roman -underline 0 -overstrike 0")
        self.original.configure(foreground="#dfdf00")
        self.original.configure(highlightbackground="#d9d9d9")
        self.original.configure(highlightcolor="black")
        self.original.configure(text='''® created by Sahdan Chung''')

        self.hidecap = tk.Label(top)
        self.hidecap.place(relx=0.730, rely=0.200, height=100, width=270)
        self.hidecap.configure(activebackground="#f9f9f9")
        self.hidecap.configure(activeforeground="black")
        self.hidecap.configure(background="#400040")

        self.cars_data = pd.read_csv('cars.csv', skiprows=[1], sep=';')
        self.film_data = pd.read_csv('film.csv', skiprows=[1], sep=';', encoding='cp1252')

    def capOutliers_Ui(self):
        self.Entry1 = tk.Entry(self.hidecap)
        self.Entry1.place(relx=0.4, rely=0.1, height=30, width=80)
        self.Entry1.configure(background="white")
        self.Entry1.configure(disabledforeground="#a3a3a3")
        self.Entry1.configure(font="TkFixedFont")
        self.Entry1.configure(foreground="#000000")
        self.Entry1.configure(highlightbackground="#d9d9d9")
        self.Entry1.configure(highlightcolor="black")
        self.Entry1.configure(insertbackground="black")
        self.Entry1.configure(selectbackground="blue")
        self.Entry1.configure(selectforeground="white")

        self.l_entry1 = tk.Label(self.hidecap)
        self.l_entry1.place(relx=0.05, rely=0.1, height=30, width=80)
        self.l_entry1.configure(activebackground="#f9f9f9")
        self.l_entry1.configure(activeforeground="black")
        self.l_entry1.configure(background="#d9d9d9")
        self.l_entry1.configure(disabledforeground="#a3a3a3")
        self.l_entry1.configure(font="-family System -size 10 -weight bold -slant roman -underline 0 -overstrike 0")
        self.l_entry1.configure(foreground="#000000")
        self.l_entry1.configure(highlightbackground="#d9d9d9")
        self.l_entry1.configure(highlightcolor="black")
        self.l_entry1.configure(text='''min_cap:''')

        self.l_entry2 = tk.Label(self.hidecap)
        self.l_entry2.place(relx=0.05, rely=0.50, height=30, width=80)
        self.l_entry2.configure(activebackground="#f9f9f9")
        self.l_entry2.configure(activeforeground="black")
        self.l_entry2.configure(background="#d9d9d9")
        self.l_entry2.configure(disabledforeground="#a3a3a3")
        self.l_entry2.configure(font="-family System -size 10 -weight bold -slant roman -underline 0 -overstrike 0")
        self.l_entry2.configure(foreground="#000000")
        self.l_entry2.configure(highlightbackground="#d9d9d9")
        self.l_entry2.configure(highlightcolor="black")
        self.l_entry2.configure(text='''max_cap:''')

        self.Entry2 = tk.Entry(self.hidecap)
        self.Entry2.place(relx=0.4, rely=0.5, height=30, width=80)
        self.Entry2.configure(background="white")
        self.Entry2.configure(disabledforeground="#a3a3a3")
        self.Entry2.configure(font="TkFixedFont")
        self.Entry2.configure(foreground="#000000")
        self.Entry2.configure(highlightbackground="#d9d9d9")
        self.Entry2.configure(highlightcolor="black")
        self.Entry2.configure(insertbackground="black")
        self.Entry2.configure(selectbackground="blue")
        self.Entry2.configure(selectforeground="white")

    def kmean_ui(self):
        self.label_kmeans = tk.Label(self.Labelframe1)
        self.label_kmeans.place(relx=0.055, rely=0.550, height=30, width=100)
        self.label_kmeans.configure(activebackground="#f9f9f9")
        self.label_kmeans.configure(activeforeground="black")
        self.label_kmeans.configure(background="#c0c0c0")
        self.label_kmeans.configure(disabledforeground="#a3a3a3")
        self.label_kmeans.configure(font="-family System -size 10 -weight bold -slant roman -underline 0 -overstrike 0")
        self.label_kmeans.configure(foreground="#000000")
        self.label_kmeans.configure(highlightbackground="#d9d9d9")
        self.label_kmeans.configure(highlightcolor="black")
        self.label_kmeans.configure(text='''num of cluster:''')

        self.TCombobox_kmeans = ttk.Combobox(self.Labelframe1)
        self.TCombobox_kmeans.place(relx=0.5, rely=0.575, height=30
                               , width=100, bordermode='ignore')
        self.value_list_kmeans= ['3', '4', ]
        self.TCombobox_kmeans.configure(values=self.value_list_kmeans)
        self.TCombobox_kmeans.configure(state='readonly')
        self.TCombobox_kmeans.configure(takefocus="")

    def destroy_kmeanUI(self):
        try:
            self.label_kmeans.destroy()
            self.TCombobox_kmeans.destroy()
        except:
            pass

    def column_ui(self):
        self.Labelframe2 = tk.LabelFrame(self.Labelframe1)
        self.Labelframe2.place(relx=0.069, rely=0.169, relheight=0.38
                               , relwidth=0.862, bordermode='ignore')
        self.Labelframe2.configure(relief='groove')
        self.Labelframe2.configure(font="-family System -size 10 -weight bold -slant roman -underline 0 -overstrike 0")
        self.Labelframe2.configure(foreground="black")
        self.Labelframe2.configure(labelanchor="n")
        self.Labelframe2.configure(background="#d9d9d9")
        self.Labelframe2.configure(highlightbackground="#d9d9d9")
        self.Labelframe2.configure(highlightcolor="black")
        self.Labelframe2.configure(text='''Column Selection''')

        self.Tcombobox_column = ttk.Combobox(self.Labelframe2)

        self.Tcombobox_column.place(relx=0.48, rely=0.222, relheight=0.222
                                    , relwidth=0.5, bordermode='ignore')
        if self.TCombobox_csv.get() == "Cars":
            self.value_list_column = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration',
                                      'Model', ]
        elif self.TCombobox_csv.get() == "Film":
            self.value_list_column = ['Year', 'Length', 'Popularity', ]
        self.Tcombobox_column.configure(values=self.value_list_column)
        self.Tcombobox_column.configure(state='readonly')
        self.Tcombobox_column.configure(takefocus="")

        self.Label_column = tk.Label(self.Labelframe2)
        self.Label_column.place(relx=0.08, rely=0.222, height=34, width=62
                           , bordermode='ignore')
        self.Label_column.configure(activebackground="#f9f9f9")
        self.Label_column.configure(activeforeground="black")
        self.Label_column.configure(background="#d9d9d9")
        self.Label_column.configure(disabledforeground="#a3a3a3")
        self.Label_column.configure(font="-family {Segoe UI} -size 12 -weight bold -slant roman -underline 0 -overstrike 0")
        self.Label_column.configure(foreground="#000000")
        self.Label_column.configure(highlightbackground="#d9d9d9")
        self.Label_column.configure(highlightcolor="black")
        self.Label_column.configure(text='''column''')

    def x_y_axis_selectionUi(self):
        self.Labelframe3 = tk.LabelFrame(self.Labelframe1)
        self.Labelframe3.place(relx=0.069, rely=0.169, relheight=0.38
                               , relwidth=0.862, bordermode='ignore')
        self.Labelframe3.configure(relief='groove')
        self.Labelframe3.configure(font="-family System -size 10 -weight bold -slant roman -underline 0 -overstrike 0")
        self.Labelframe3.configure(foreground="black")
        self.Labelframe3.configure(labelanchor="n")
        self.Labelframe3.configure(background="#d9d9d9")
        self.Labelframe3.configure(highlightbackground="#d9d9d9")
        self.Labelframe3.configure(highlightcolor="black")
        self.Labelframe3.configure(text='''Axis Selection''')

        self.TCombobox_x = ttk.Combobox(self.Labelframe3)
        self.TCombobox_x.place(relx=0.48, rely=0.222, relheight=0.222
                                    , relwidth=0.5, bordermode='ignore')
        if self.TCombobox_csv.get() == "Cars":
            self.value_list_x = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model', ]
        elif self.TCombobox_csv.get() == "Film":
            self.value_list_x = ['Year', 'Length', 'Popularity', ]
        self.TCombobox_x.configure(values=self.value_list_x)
        self.TCombobox_x.configure(state='readonly')
        self.TCombobox_x.configure(takefocus="")

        self.TCombobox_y = ttk.Combobox(self.Labelframe3)
        self.TCombobox_y.place(relx=0.48, rely=0.593, relheight=0.222
                                    , relwidth=0.5, bordermode='ignore')
        if self.TCombobox_csv.get() == "Cars":
            self.value_list_y = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model', ]
        elif self.TCombobox_csv.get() == "Film":
            self.value_list_y = ['Year', 'Length', 'Popularity', ]
        self.TCombobox_y.configure(values=self.value_list_y)
        self.TCombobox_y.configure(state='readonly')
        self.TCombobox_y.configure(takefocus="")

        self.Label_x = tk.Label(self.Labelframe3)
        self.Label_x.place(relx=0.08, rely=0.222, height=34, width=62
                           , bordermode='ignore')
        self.Label_x.configure(activebackground="#f9f9f9")
        self.Label_x.configure(activeforeground="black")
        self.Label_x.configure(background="#d9d9d9")
        self.Label_x.configure(disabledforeground="#a3a3a3")
        self.Label_x.configure(font="-family {Segoe UI} -size 12 -weight bold -slant roman -underline 0 -overstrike 0")
        self.Label_x.configure(foreground="#000000")
        self.Label_x.configure(highlightbackground="#d9d9d9")
        self.Label_x.configure(highlightcolor="black")
        self.Label_x.configure(text='''x-axis''')

        self.Label_y = tk.Label(self.Labelframe3)
        self.Label_y.place(relx=0.08, rely=0.593, height=34, width=64
                           , bordermode='ignore')
        self.Label_y.configure(activebackground="#f9f9f9")
        self.Label_y.configure(activeforeground="black")
        self.Label_y.configure(background="#d9d9d9")
        self.Label_y.configure(disabledforeground="#a3a3a3")
        self.Label_y.configure(font="-family {Segoe UI} -size 12 -weight bold -slant roman -underline 0 -overstrike 0")
        self.Label_y.configure(foreground="#000000")
        self.Label_y.configure(highlightbackground="#d9d9d9")
        self.Label_y.configure(highlightcolor="black")
        self.Label_y.configure(text='''y-axis''')


    def destroy_capUi(self):
        try:
            self.Entry1.destroy()
            self.Entry2.destroy()
            self.l_entry2.destroy()
            self.l_entry1.destroy()
        except:
            pass

    def csv_selection(self):
        if self.TCombobox_csv.get() == "Cars":
            return self.cars_data
        elif self.TCombobox_csv.get() == "Film":
            return self.film_data
        else:
            pass

    def output(self, result):
        self.Frame_output.pack()
        pt = Table(self.Frame_output, dataframe=result, height=550, width=620)
        pt.show()

    def output2(self, result):
        self.Frame_output.pack_forget()
        self.Output_gui.configure(text=result)

    def question_a(self):
        a = self.csv_selection()
        column = str(self.Tcombobox_column.get())
        min_input = float(self.Entry1.get())
        max_input = float(self.Entry2.get())
        min_bound = a[str(column)].quantile(float(min_input))
        max_bound = a[str(column)].quantile(float(max_input))
        print(min_bound, max_bound)
        self.output2("\n\nminimum cap outlier\t: "+str(min_bound)+"\nmaximum cap outlier\t: " + str(max_bound))

    def question_b(self):
        a = self.csv_selection()
        x = str(self.TCombobox_x.get())
        y = str(self.TCombobox_y.get())
        reg = linear_model.LinearRegression()
        reg.fit(a[[x]], a[[y]])
        pred = reg.predict(a[[x]])
        self.output2('Mean squared error: %.4f'
              % mean_squared_error(a[[y]], pred))
        plt.scatter(a[x], a[y], color ='blue')
        plt.plot(a[x], pred, color='black')
        plt.show()

    def question_c(self):
        a = self.csv_selection()
        column = str(self.Tcombobox_column.get())
        single_column = a[a[column] == a[column].max()]
        self.output(single_column)

    def question_d(self):
        a = self.csv_selection()
        x = str(self.TCombobox_x.get())
        y = str(self.TCombobox_y.get())
        reg = linear_model.LinearRegression()
        reg.fit(a[[x]], a[[y]])
        pred = reg.predict(a[[x]])
        a['pred_y'] = pred
        gg = a['pred_y'].autocorr()
        self.output2("autocorrelations = "+str(gg))
        plt.scatter(a[x], a[y], color='blue')
        plt.plot(a[x], pred, color='black')
        plt.show()

    def question_e(self):
        a = self.csv_selection()
        if self.TCombobox_csv.get()=="Cars":
            a = a.drop(columns=['Car','Origin'])
        elif self.TCombobox_csv.get() == "Film":
            a = a.drop(columns=['Title','Subject','Actor','Actress','Director','Awards','*Image'])
        a["corr"] = 0
        for i in range(len(a) - 1):
            values1 = a.iloc[i, :-1].astype('float64')
            values2 = a.iloc[i + 1, :-1].astype('float64')
            corr = values1.corr(values2)
            a["corr"].iloc[i] = corr
        self.output(a)

    def question_f(self):
        a = self.csv_selection()
        x = str(self.TCombobox_x.get())
        y = str(self.TCombobox_y.get())
        a.dropna(axis=0, inplace=True)
        reg = linear_model.LinearRegression()
        reg.fit(a[[x]], a[[y]])
        pred = reg.predict(a[[x]])
        r2 = r2_score(a[[y]], pred)
        self.output2("coefficient= " +str(reg.coef_[0][0])+
                     "\nintercept= " +str(reg.intercept_[0])+
                     "\n\ny="+str(reg.coef_[0][0])+"x + "+str(reg.intercept_[0])+
                     '\n\nMean squared error: %.2f'
                     % mean_squared_error(a[[y]], pred)+
                     "\nCoefficient of determination(r square): " +str(r2))

        plt.scatter(a[x], a[y], color ='blue')
        plt.plot(a[x], pred, color='black')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend()
        plt.show()

    def question_g(self):
        a = self.csv_selection()
        x = str(self.TCombobox_x.get())
        y = str(self.TCombobox_y.get())
        a.dropna(axis=0, inplace=True)
        if self.TCombobox_kmeans.get() == '3':
            kmeans = KMeans(n_clusters=3)
            y_pred = kmeans.fit_predict(a[[x, y]])
            a['cluster'] = y_pred
            a1 = a[a['cluster'] == 0]
            a2 = a[a['cluster'] == 1]
            a3 = a[a['cluster'] == 2]
            self.output(a)
            plt.scatter(a1[x], a1[y], color='red')
            plt.scatter(a2[x], a2[y], color='cyan')
            plt.scatter(a3[x], a3[y], color='coral')
            plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', marker="*",
                        label='centroid')
            plt.xlabel(x)
            plt.ylabel(y)
            plt.legend()
            plt.show()
        elif self.TCombobox_kmeans.get() == '4':
            kmeans = KMeans(n_clusters=4)
            y_pred = kmeans.fit_predict(a[[x, y]])
            a['cluster'] = y_pred
            a1 = a[a['cluster'] == 0]
            a2 = a[a['cluster'] == 1]
            a3 = a[a['cluster'] == 2]
            a4 = a[a['cluster'] == 3]
            self.output(a)
            plt.scatter(a1[x], a1[y], color='red')
            plt.scatter(a2[x], a2[y], color='cyan')
            plt.scatter(a3[x], a3[y], color='coral')
            plt.scatter(a4[x], a4[y], color='yellow')
            plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', marker="*",
                        label='centroid')
            plt.xlabel(x)
            plt.ylabel(y)
            plt.legend()
            plt.show()
        else:
            pass

    def select_question(self):
        if self.TCombobox_analysis.get() == self.value_list[0]:
            self.capOutliers_Ui()
            self.column_ui()
            self.destroy_kmeanUI()
        elif self.TCombobox_analysis.get() == self.value_list[2]:
            self.column_ui()
            self.destroy_capUi()
            self.destroy_kmeanUI()
        elif self.TCombobox_analysis.get() == self.value_list[4]:
            self.destroy_capUi()
            self.destroy_kmeanUI()
            pass
        elif self.TCombobox_analysis.get() == self.value_list[6]:
            self.x_y_axis_selectionUi()
            self.destroy_capUi()
            self.kmean_ui()
        else:
            self.x_y_axis_selectionUi()
            self.destroy_capUi()
            self.destroy_kmeanUI()

    def analyse(self):
        if self.TCombobox_analysis.get() == self.value_list[0]:
            self.question_a()
        elif self.TCombobox_analysis.get() == self.value_list[1]:
            self.question_b()
        elif self.TCombobox_analysis.get() == self.value_list[2]:
            self.question_c()
        elif self.TCombobox_analysis.get() == self.value_list[3]:
            self.question_d()
        elif self.TCombobox_analysis.get() == self.value_list[4]:
            self.question_e()
        elif self.TCombobox_analysis.get() == self.value_list[5]:
            self.question_f()
        elif self.TCombobox_analysis.get() == self.value_list[6]:
            self.question_g()
        else:
            pass

if __name__ == '__main__':
    vp_start_gui()





