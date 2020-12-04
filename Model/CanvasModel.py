

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt

class Canvas(FigureCanvas):
    def __init__(self,parent=None,rf=1,nn=1):
        '''fig = Figure(figsize=(5, 4), dpi=200)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)'''
        self.plot(rf,nn)

    # plot and compare the scores
    def plot(self,rf,nn):
        try:
            x = [rf*100,nn*100]
            indx = np.arange(len(x))
            score_label = np.arange(0,110,10)
            labels = ["Random Forest", "Neural Network"]
            fig,ax = plt.subplots()
            barshow = ax.bar(indx,x,0.3,label='Accuracy')

            #inserting x axis label
            ax.set_xticks(indx)
            ax.set_xticklabels(labels)

            #inserting y axis label
            ax.set_yticks(score_label)
            ax.set_yticklabels(score_label)

            ax.set_title('Prediction Accuracy')
            plt.show()
        except Exception as e:
            print('Plot Error:',str(e))