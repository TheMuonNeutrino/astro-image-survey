import os, shutil

def printC(colour,*args):
    print(colour,*args,bcolors.ENDC)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def ensureFolder(directory):
    """Checks that a directory is present, creating if it is not.
    Cannot handle cases where the parent directory doesn't exist either.
    """
    if not os.path.exists(directory):
        os.mkdir(directory)

def clearFolder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    ensureFolder(path)

def pad(x,n):
    s = str(x)
    if len(s) < n:
        s = "0"*n + s
        s = s[-n:]
    return s
