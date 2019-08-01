"""
Functions for working with Sentinel-2 data stored in the Microsoft Azure
Block-Blob service.

"""

from collections import OrderedDict
from sentinelsat import SentinelAPI
from datetime import date
from azure.storage.blob import BlockBlobService, PublicAccess
import numpy as np
import os
import shutil
import fnmatch
import glob


class AzureAccess:



    _acc_name = None
    _acc_key = None
    block_blob_service = None



    def __init__(self, acc_name, acc_key):
        self._acc_name = acc_name
        self._acc_key = acc_key
        self.connect_to_service()



    def connect_to_service(self):
        self.block_blob_service = BlockBlobService(account_name=self._acc_name, 
            account_key=self._acc_key)



    def send_to_blob(tile, L1Cpath, check_blobs=False):

        """
        Function uploads processed L2A products to blob storage. Check if container matching tile name already exists -if so
        add files to that container, if not, create new container with name = tile.

        :param blob_account_name:
        :param blob_account_key:
        :param tile:
        :return:
        """

        container_name = tile.lower() #convert string to lower case because Azure blobs named in lower case

        local_path = L1Cpath

        # find files to upload and append names to list
        for folder in os.listdir(local_path):

            file_names = []
            file_paths = []
            filtered_paths = []
            filtered_names = []

            folder_path = str(local_path+folder+"/")

            # append all file paths and names to list, then filter to the relevant jp2 files
            for (dirpath, dirnames, filenames) in os.walk(folder_path):
                file_paths += [os.path.join(dirpath, file) for file in filenames]
                file_names += [name for name in filenames]

            for path in fnmatch.filter(file_paths,"*.jp2"):
                    filtered_paths.append(path)

            for file in fnmatch.filter(file_names,"*.jp2"):
                filtered_names.append(file)


            #' check for existing containers
            existing_containers = self.block_blob_service.list_containers()

            existing_container_names = []

            for item in existing_containers:
                existing_container_names.append(item.name)

            if any(tile.lower() in p for p in existing_container_names):
                print('\n',' CONTAINER {} ALREADY EXISTS IN STORAGE ACCOUNT '.center(80, 'X').format(tile),
                      '\n', 'X'.center(80, 'X'))

                # add files to existing container matching tile name
                for i in np.arange(0,len(filtered_paths)):
                    print('\n',' UPLOADING FOLDERS TO EXISTING CONTAINER {} '.center(80, 'X').format(tile),
                          '\n', 'X'.center(80, 'X'))
                    source = str(filtered_paths[i])
                    destination = str(folder+'/' + filtered_names[i])

                    try:
                        self.block_blob_service.create_blob_from_path(container_name, destination, source)

                    except:
                        print("Uploading to blob failed")


            else:
                print('\n',
                      ' CONTAINER DOES NOT ALREADY EXIST. CREATING NEW CONTAINER {} '.center(80, 'X').format(tile),
                      '\n', 'X'.center(80, 'X'))

                # Create a container with the tile as filename, then add files.
                self.block_blob_service.create_container(container_name)

                print('\n', ' CONTAINER CREATED. UPLOADING FOLDERS TO NEW CONTAINER '.center(80, 'X'),
                      '\n', 'X'.center(80, 'X'))

                for i in np.arange(0, len(filtered_paths)):
                    source = str(filtered_paths[i])
                    destination = str(folder + '/' + filtered_names[i])

                    try:
                        self.block_blob_service.create_blob_from_path(container_name, destination, source)

                    except:
                        print("Uploading to blob failed")

        if check_blobs:

            blob_list = []
            print("Retrieving blobs in specified container...")

            try:
                content = self.block_blob_service.list_blobs(container_name)
                print("******Blobs currently in the container:**********")
                for blob in content:
                    blob_list.append(blob.name)

            except:
                print('\n', ' CHECKING BLOBS FAILED '.center(80, 'X'),
                      '\n', 'X'.center(80, 'X'))

            print('\n',' BLOBS CURRENTLY STORED IN CONTAINER {}: '.center(80, 'X'))
            print("\n",blob_list)

            if any(folder in p for p in blob_list):
                print('\n', ' UPLOAD SUCCESSFUL: FOLDERS IN BLOB MATCH THOSE IN SCRIPT '.center(80, 'X'),
                      '\n', 'X'.center(80, 'X'))
                upload_status = 1
            else:
                upload_status = 0

        else: upload_status = 1 # if check_blobs deselected, assume upload was successful (not recommended)

        return upload_status



    def download_imgs_by_date(tile, date, img_path):

        """
        This function downloads subsets of images stored remotely in Azure blobs. The blob name is identical to the
        ESA tile ID. Inside each blob are images from every overpass made in June, July and August of the given year.
        The files in blob storage are L2A files, meaning the L1C product has been downloaded from Sentinel-Hub and
        processed for atmospheric conditions, spatial resolution etc using the Sen2cor command line tool.
        This function searches for the blob associated with "tile" and then filteres out a subset according to the prescribed
        date.
        A flags can be raised in this function. The script checks that the correct number of image files have been
        downloaded and that one of them is the cloud mask. If not, the flag is printed to the console and the files
        associated with that particular date for that tile are discarded. The tile and date info are appended to a list of
        failed downloads.
        :param tile: tile ID
        :param date: date of overpass
        :param img_path: path to folder where images and other temp files will be stored
        :param blob_account_name: account name for azure storage account where blobs are stored
        :param blob account key: accesskey for blob storage account
        :return filtered_blob_list: list of files to download
        :return download_flag: Boolean, if True then problem with download, files skipped
        """


        # setup list
        bloblist = []
        download_flag = False
        QCflag = False

        # append names of all blobs to bloblist
        generator = self.block_blob_service.list_blobs(tile)
        for blob in generator:
            bloblist.append(blob.name)

        # filter total bloblist to just jp2s, then just for the specified date
        filtered_by_type = [string for string in bloblist if '_20m.jp2' in string]
        filtered_bloblist = [string for string in filtered_by_type if str("L2A_" + date) in string]


        # download the files in the filtered list
        for i in filtered_bloblist:
            print(i)
            try:
                self.block_blob_service.get_blob_to_path(tile,
                                             i, str(img_path+i[-38:-4]+'.jp2'))
            except:
                print("download failed {}".format(i))

            # index to -38 because this is the filename without paths to folders etc

        # Check downloaded files to make sure all bands plus the cloud mask are present in the wdir
        # Raises download flag (Boolean true) and reports to console if there is a problem

        if len(glob.glob(str(img_path + '*_B*_20m.jp2'))) < 9 or len(glob.glob(str(img_path + '*CLD*_20m.jp2'))) == 0:
            download_flag = True

            print("\n *** DOWNLOAD QC FLAG RAISED *** \n *** There may have been no overpass on this date, or there is a"
                  " band image or cloud layer missing from the downloaded directory ***")

        else:
            download_flag = False
            print("\n *** NO DOWNLOAD QC FLAG RAISED: ALL NECESSARY FILES AVAILABLE IN WDIR ***")

        # relevant files now downloaded from blob and stored in the savepath folder

        return filtered_bloblist, download_flag

