from scipy import signal
import os
import numpy as np
from scipy import stats
import pandas as pd


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
            #df = self._swap_sides(df)
            # fill empty slots with data from before and after (limit the distance to 3) (if available)
            df = df.fillna(method='ffill', limit=3)
            df = df.fillna(method='bfill', limit=3)
            # save filled data spots
            mask = df.isnull()

            # fill all slots to be able to smooth
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')

            # Detect outliers
            for col in df.columns:
                df.loc[:, col], detected_outliers = hampel_filter_pandas(df.loc[:, col], self.window_length)

            # smooth each column in the data frame
            for i in range(len(df.columns)):
                if df.iloc[0, i] is not None:
                    df.iloc[:, i] = signal.savgol_filter(df.iloc[:, i], self.window_length, self.polyorder)
            # remove slots where we could not fill using before or after data
            # (we remove the slots where we can't be sure the data is right)
            self.df_smooth = df.mask(mask)

        print('Smoothing completed')
        return self.df_smooth

    def _swap_sides(self, df):
        body_parts = [
            'eye', 'ear', 'shoulder', 'elbow',
            'wrist', 'hip', 'knee', 'ankle'
        ]
        prev_row = df.iloc[0, :]
        for index, curr_row in df.iloc[1:, :].iterrows():
            for part in body_parts:
                curr_left_part = np.array([curr_row[f'left_{part}_x'], curr_row[f'left_{part}_y']])
                curr_right_part = np.array([curr_row[f'right_{part}_x'], curr_row[f'right_{part}_y']])

                if curr_left_part[0] < curr_right_part[0]:
                    temp = curr_row[f'left_{part}_x'].copy()
                    curr_row[f'left_{part}_x'] = curr_row[f'right_{part}_x'].copy()
                    curr_row[f'right_{part}_x'] = temp
                    temp = curr_row[f'left_{part}_y'].copy()
                    curr_row[f'left_{part}_y'] = curr_row[f'right_{part}_y'].copy()
                    curr_row[f'right_{part}_y'] = temp

            prev_row = curr_row
        return df

    def save_to_csv(self, output_folder):
        """
        Saves the data frames after smoothing
        :param output_folder: str, path to output folder
        """
        if self.df_smooth is not None:
            outfile_path = os.path.join(output_folder, 'stickman_data_smoothed.csv')
            self.df_smooth.to_csv(outfile_path, index=False)


def hampel_filter_pandas(input_series, window_size, n_sigmas=2):
    """
    Remove outliers using hamper filter
    """
    k = 1.4826  # scale factor for Gaussian distribution
    new_series = input_series.copy()

    # helper lambda function
    MAD = lambda x: np.median(np.abs(x - np.median(x)))

    rolling_median = input_series.rolling(window=2 * window_size, center=True).median()
    rolling_mad = k * input_series.rolling(window=2 * window_size, center=True).apply(MAD)
    diff = np.abs(input_series - rolling_median)

    indices = list(np.argwhere((diff > (n_sigmas * rolling_mad)).values).flatten())
    new_series[indices] = rolling_median[indices]

    return new_series, indices


if __name__ == "__main__":
    df = pd.read_csv('output/stickman_data.csv')

    smoother = Smooth()
    smoother.smooth(df)
