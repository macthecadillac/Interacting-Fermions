"""
This file is part of spinsys.

Spinsys is free software: you can redistribute it and/or modify
it under the terms of the BSD 3-clause license. See LICENSE.txt
for exact terms and conditions.


Timer class.

1-14-2017
"""

import time
import sys
import shutil
import os


class EstimateTime():

    def __init__(self, njobs, mode='moving_average'):
        self.njobs = njobs
        self.mode = mode
        self.last_timestamp = time.time()
        # Differential time log that moving average uses
        self.dtime_log = []
        # Number of time points to keep in differential time log for
        #  moving average
        self.__keep_pts = max(0.2 * njobs, 100)
        self.__iteration = 0
        self.__elapsed = 0
        self.__alg = {
            'moving_average': self.__moving_average,
            'average': self.__average
        }

    def __average(self):
        """Returns the average time per iteration"""
        return sum(self.dtime_log) / len(self.dtime_log)

    def __moving_average(self):
        """Returns the moving average time per iteration"""
        if len(self.dtime_log) > self.__keep_pts:
            self.dtime_log.pop(0)
        return sum(self.dtime_log) / len(self.dtime_log)

    def est(self):
        """This method returns the estimated time"""
        try:
            time_per_iteration = self.__alg[self.mode]()
            return (self.njobs - self.__iteration) * time_per_iteration
        except ZeroDivisionError:
            return 0

    def stop_watch(self):
        """This method needs to be called to initiate another
        round of computation for time estimation
        """
        self.__iteration += 1
        dtime = time.time() - self.last_timestamp
        self.last_timestamp = time.time()
        # Discard first iteration since its timing is likely longer
        #  due to load times
        if self.__iteration > 1:
            self.dtime_log.append(dtime)


class Timer():
    """
    The timer object times the operations and prints out a nice progress bar.
    At the end of the operation, the timer will print out the total time
    taken so far.

    Usage: Initialize the class by calling timer = Timer(total_number_of_jobs),
           then at the end of each job (such as in a for loop after every
           task) invoke timer.progress() and the progress bar and the time
           estimation will be updated.

           At the end of the program, that is after all specified jobs
           are finished, the total elapsed time will be automatically shown
           unless the show_elapsed option is set to False.
    """

    def __init__(self, total, mode='moving_average'):
        """
        Initializes the timer object

        Args: "total" is the total number of jobs that would take roughly
              the same amount of time to finish.
              "mode" is the mode of time estimation. Options are "average"
              and "moving_average".
        """
        self.__start_time = time.time()
        self.iteration = 0
        self.total = total

        # Initiate EstimateTime class to enable precise time estimation.
        self.estimatetime = EstimateTime(self.total, mode)

        # Show progress bar
        self.__show_progress()

    def __update_progress(self):
        """
        Increments self.iteration by 1 every time this method is called.
        """
        self.iteration += 1

    def __sec_to_human_readable_format(self, time):
        """
        Converts time (in seconds) into the HHMMSS format. Returns a string.
        """
        Days = str(int(time // 86400))
        Hr = int((time % 86400) // 3600)
        if Hr < 10:
            Hr = "0" + str(Hr)
        else:
            Hr = str(Hr)
        Min = int((time % 3600) // 60)
        if Min < 10:
            Min = "0" + str(Min)
        else:
            Min = str(Min)
        Sec = round(time % 60)
        if Sec < 10:
            Sec = "0" + str(Sec)
        else:
            Sec = str(Sec)
        if Days == str(0):
            ET = Hr + ":" + Min + ":" + Sec + "        "
        elif Days == str(1):
            ET = Days + " Day " + Hr + ":" + Min + ":" + Sec + "  "
        else:
            ET = Days + " Days " + Hr + ":" + Min + ":" + Sec

        return ET

    def __show_progress(self):
        """
        Shows the progress bar on screen. When being called after the
        first time, it updates the progress bar.
        """
        term_width = list(shutil.get_terminal_size())[0]
        if self.iteration == 0:
            report_time = ""
            barlength = term_width - 80
            filledlength = 0
            percent = 0
        else:
            # Calculate time used for progress report purposes.
            elapsed = self.elapsed_time()
            est_time = self.estimatetime.est()

            if self.iteration == self.total:
                est_time = 0
            ET = self.__sec_to_human_readable_format(est_time)

            report_time = "Est. time: {}  Elapsed: {}".format(ET, elapsed)
            barlength = term_width - len(report_time) - 26
            filledlength = int(round(barlength * (self.iteration) / self.total))
            percent = round(100.00 * ((self.iteration) / self.total), 1)

        bar = '\u2588' * filledlength + '\u00B7' * (barlength - filledlength)
        sys.stdout.write('\r%s |%s| %s%s %s' % ('  Progress:', bar,
                                                percent, '%  ', report_time)),
        sys.stdout.flush()
        if self.iteration == self.total:
            sys.stdout.write('\n')
            sys.stdout.flush()

        if self.iteration == self.total:
            print('  Done.')

    def elapsed_time(self):
        """Prints the total elapsed time."""
        elapsed = time.time() - self.__start_time
        elapsed_time = self.__sec_to_human_readable_format(elapsed)
        return elapsed_time

    def progress(self):
        """Prints the progress on screen"""
        # Update the progress.
        if self.iteration < self.total:
            self.estimatetime.stop_watch()
            self.__update_progress()
            self.__show_progress()
