from scipy import signal
import os


class Smooth:
    def __init__(self, window_length=7, polyorder=2):
        """
        :param window_length: int, length of the smoothing window
        :param polyorder: int, degree of polynomial used for smoothing
        """
        self.df_smooth = None
        self.window_length = window_length
        self.polyorder = polyorder

    def smooth(self, df=None):
        """
        Smooth the data using savgol filter
        :param df: data frame for pose data
        :return: pose df and face df after smoothing
        """

        # Smooth it out
        if df is not None:
            # fill empty slots with data from before and after (limit the distance to 3) (if available)
            df = df.fillna(method='ffill', limit=3)
            df = df.fillna(method='bfill', limit=3)
            # save filled data spots
            mask = df.isnull()
            # fill all slots to be able to smooth
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')
            # smooth each column in the data frame
            for i in range(len(df.columns)):
                if df.iloc[0, i] is not None:
                    df.iloc[:, i] = signal.savgol_filter(df.iloc[:, i], self.window_length, self.polyorder)
            # remove slots where we could not fill using before or after data
            # (we remove the slots where we can't be sure the data is right)
            self.df_smooth = df.mask(mask)

        print('Smoothing completed')
        return self.df_smooth

    def save_to_csv(self, output_folder):
        """
        Saves the data frames after smoothing
        :param output_folder: str, path to output folder
        """
        if self.df_smooth is not None:
            outfile_path = os.path.join(output_folder, 'stickman_data_smoothed.csv')
            self.df_smooth.to_csv(outfile_path, index=False)




