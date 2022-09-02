import os

def create_folder(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def file_search(dirname, ret, list_avoid_dir=[]):
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        #print('filename',filename)
        #print("fullname",full_filename)
        if os.path.isdir(full_filename):
            if full_filename.split('/')[-1] in list_avoid_dir:
                continue
            else:
                file_search(full_filename, ret, list_avoid_dir)
        else:
            ret.append(full_filename)