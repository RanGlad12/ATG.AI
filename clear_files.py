import os
import shutil

def clear_files(folder):
    '''
    Clears the files in folder
    '''
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as error:
            print(f'Failed to delete {file_path}. Reason: {error}')