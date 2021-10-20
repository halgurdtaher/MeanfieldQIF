"""
Contains simulation data base class and help function for data loading and saving.
"""

import os
import dill
from NumSim.Parameters import Params


def make_path(path, isfile=True):
    if (isfile):
        path = os.path.dirname(path)
    if (len(path) > 0):
        os.makedirs(path, exist_ok=True)


def save_object(path, o, ):
    dill.dump(o, open("%s.dill" % path, 'wb'), dill.HIGHEST_PROTOCOL)


def load_object(path):
    return dill.load(open(path, 'rb'))


class SData:
    """
        A base class to store, save and load simulation data.

        Attributes:
            p (Parameter.Params): Provides integration and system parameters
            base_path (static string): Base directory for loading and saving data
    """

    base_path = ""

    def __init__(self, path: str = None, relative_path=True):
        """
             Creates or loads data.

             Parameters:
                 path (string): The file to load from

                 relative_path (bool): If true, loads data from base_path/path instead of path
         """
        self.path = path
        self.p: Params = None
        if (path is not None):
            self.from_file(path, relative_path=relative_path)

    def save(self, path: str, relative_path=True):
        """
             Saves the SData object to path.

             Parameters:
                 path (string): The path to save to

                relative_path (bool): If true, save data to base_path/path instead of path
         """
        if (relative_path):
            self.path = self.base_path + path
        else:
            self.path = path
        make_path(self.path)
        save_object(self.path, self)
        print(f"Saving {type(self)} to:\n{self.path}.dill")
        return

    def from_file(self, path: str, relative_path=True):
        """
             Updates all members of SData from the loaded .dill file.

             Parameters:
                 path (string): The file to load from

                 relative_path (bool): If true, loads data from base_path/path instead of path
         """
        if (relative_path):
            self.path = self.base_path + path
        else:
            self.path = path

        obj = load_object(self.path)
        self.__dict__.update(obj.__dict__)

        return
