import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import argparse

class PointSelector():
    '''
        Point selector class that stores all point clicked on image
    '''

    def __init__(self, img, fig, ax):
        '''

        :param img: baseline image for clicked points
        :param fig: figure object of plot
        :param ax: axes object of plot
        '''
        self.coords = []
        self.prev_len_coords = 0
        self.image = img
        self.fig = fig
        self.ax = ax

    def show_image(self):
        '''
            The function show the class image parameter
        '''
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.ax.imshow(self.image)
        plt.show()

    def refresh_plot(self):
        self.ax.clear()
        for x, y in self.coords:
            self.ax.plot(x, y, marker='o', markersize=3, color="red")

        self.ax.imshow(self.image)
        plt.gcf().canvas.draw_idle()

    def save_pickle(self):
        print('Thanks for selecting the rectangle!')
        print('Saving pickle file')
        with open('selected_coords.pickle', 'wb') as handle:
            pickle.dump(self.coords, handle)

    def onclick(self, event):
        '''
            On click callback function to store the point.
            When the fourth point is selecte, the function writes out the
            pickle file with selected points and disconnect the callback
        :param event: click event that woke up the callback
        :return: new coordinates array with selected points stored
        '''

        ix, iy = event.xdata, event.ydata

        self.coords.append((ix, iy))

        # When the fourth point is selected, disconnect and save file
        if len(self.coords) == 4:
            self.fig.canvas.mpl_disconnect(self.cid)
            self.save_pickle()

        # When a new point is selected, refresh the plot
        if self.prev_len_coords != len(self.coords):
            self.refresh_plot()

        self.prev_len_coords = len(self.coords)

        return self.coords


parser = argparse.ArgumentParser(description='Point Selector GUI')
parser.add_argument('-i', '--image', help='test image', type=str, required=True)

def main():

    args = parser.parse_args()

    test_image = mpimg.imread(args.image)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(test_image)

    ps = PointSelector(test_image, fig, ax)
    ps.show_image()


if __name__=='__main__':
    main()