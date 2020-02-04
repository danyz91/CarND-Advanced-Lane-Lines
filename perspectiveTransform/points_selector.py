import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


class PointSelector():

    def __init__(self, img, fig, ax):
        self.coords = []
        self.prev_len_coords = 0
        self.image = img
        self.fig = fig
        self.ax = ax

    def show_image(self):
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.ax.imshow(self.image)
        plt.show()


    def onclick(self, event):

        global ix, iy
        ix, iy = event.xdata, event.ydata

        self.coords.append((ix, iy))

        if len(self.coords) == 4:
            self.fig.canvas.mpl_disconnect(self.cid)
            print('Thanks for selecting the rectangle!')
            print('Saving pickle file')
            with open('selected_coords.pickle', 'wb') as handle:
                pickle.dump(self.coords, handle)

        if self.prev_len_coords != len(self.coords):
            self.ax.clear()
            for x, y in self.coords:
                self.ax.plot(x, y, marker='o', markersize=3, color="red")

            self.ax.imshow(self.image)
            plt.gcf().canvas.draw_idle()

        self.prev_len_coords = len(self.coords)

        return self.coords




def main():

    test_image = mpimg.imread('straight_lines1.jpg')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(test_image)

    ps = PointSelector(test_image, fig, ax)
    ps.show_image()


if __name__=='__main__':
    main()