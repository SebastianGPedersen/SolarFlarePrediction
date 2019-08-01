import os
"""
Import this script to setup paths
"""

# Add all individual paths (one for every machine)
#   Path to: A folder named 'LSDA'
#       containing: the repo + the folder '.../LSDA/Processed_Data' (containing the data)
'''
__colab_path = '/gdrive'  # Assumes colab is mounted as '/gdrive'
__mw_path = 'C:/Users/mw23/Google Drive'
__mw_path2 = 'C:/Users/Mathias/Google Drive'
__sgp_path = '/Users/SebastianGPedersen/Google Drive'
__lp_path = '/home/lassepetersen/Google Drive'

# Test to figure out which machine is currently in use (also handles colab)
if os.path.isdir(__colab_path):
    __path = __colab_path
elif os.path.isdir(__mw_path):
    __path = __mw_path
elif os.path.isdir(__mw_path2):
    __path = __mw_path2
elif os.path.isdir(__sgp_path):
    __path = __sgp_path
elif os.path.isdir(__lp_path):
    __path = __lp_path
else:
    raise FileExistsError('None of the folder options exists')

# Recursively traverse folder-system from the specified __path folder, until 'LSDA' is found
folders_found = 0
for root, dirs, files in os.walk(__path):
    for name in dirs:
        if name.lower() == 'processed_data':
            folders_found += 1
            Data_Path = os.path.join(root, name).replace('\\', '/')
        if name.lower() == 'lsda':
            Gitlab_Path = os.path.join(root, name).replace('\\', '/')
            folders_found += 1

        # print(Root_Path)
            # print(Data_Path)

if folders_found != 2:
    raise FileExistsError('The specified folder exists, but does not contain "lsda" and/or "processed_data" folder')
'''

Data_Path = '/Users/SebastianGPedersen/Dropbox/KU/6. aar/LSDA/Kaggle/Data'
Gitlab_Path = '/Users/SebastianGPedersen/Dropbox/KU/6. aar/LSDA/Kaggle/lsda'
