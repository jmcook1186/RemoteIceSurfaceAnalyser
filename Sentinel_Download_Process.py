"""
The user defines a list of relevant tiles and dates. This script then accesses the SentinelHub store and downloads the
relevant tiles as zipped L1C products and saves them locally. The local file is then converted to the L2A product using
the ESA Sen2Cor processor, overwriting the L1C product. The L2A product is then uploaded to blob storage and then erased
from the local disk.

In this script, two batches of dates are provided for each tile. This is to enable flushing of the local disk before too
many tiles are saved. The files are large, and more than one month's worth of images can lead to the local disk filling
up causing the script to crash. To avoid this, I iterate through month by month instead of providing a large date range.

Blob storage is organised into separate containers for each tile with a separate folder for each date saved inside.

tile
|--date
    |--individual band jp2s


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


#################################################################
################## DEFINE FUNCTIONS #############################
#################################################################

def download_L1C(L1Cpath, tile, dates):
    """
    This function uses the sentinelsat API to download L1C products for tiles defined by "tile" in the date range
    specified by "dates". Prints number of files and their names to console.

    :param L1Cpath:
    :param tile:
    :param dates:

    :return: L1Cfiles
    """

    # define keyword arguments
    query_kwargs = {
            'platformname': 'Sentinel-2',
            'producttype': 'S2MSI1C',
            'date': dates}

    products = OrderedDict()

    # loop through tiles and list all files in date range

    kw = query_kwargs.copy()
    kw['tileid'] = tile  # products after 2017-03-31
    pp = api.query(**kw)
    products.update(pp)

    # keep metadata in pandas dataframe
    metadata = api.to_dataframe(products)
    L1Cfiles = metadata.filename
    print('\n',' DOWNLOADING {} FILES '.center(80, 'X').format(len(L1Cfiles)),'\n', 'X'.center(80, 'X'))

    # download all files
    api.download_all(products, directory_path = L1Cpath)


    return L1Cfiles


def process_L1C_to_L2A(L1C_path, L1Cfiles, unzip_files = True):
    """
    This function is takes the downloaded L1C products from SentinelHub and converts them to L2A product using Sen2Cor.
    This is achieved as a batch job by iterating through the file names (L1Cfiles) stored in L1Cpath
    Running this script will save the L2A products to the working directory

    :return: None

    """
    for L1C in L1Cfiles:

        # unzip downloaded files in place, then delete zipped folders
        if unzip_files:

            unzip = str('unzip '+ L1C_path + L1C[0:-4]+'zip'+' -d ' + L1C_path)
            os.system(unzip)

        # process L1C to L2A
        sen2cor = str('/home/tothepoles/PycharmProjects/Sen2Cor-02.05.05-Linux64/bin/L2A_Process ' + L1C_path + '{}'.format(L1C) + ' --resolution=20')
        os.system(sen2cor)

    return


def remove_L1C(L1Cpath):
    """
    function removes L1C files from wdir after processing to L2A

    :param L1Cpath: path to wdir
    :return:

    """

    files = glob.glob(str(L1Cpath+'*L1C*')) # find files containing "L1C"

    for f in files:
        try:
            shutil.rmtree(f) # try removing using shutil.rmtree() (for unzipped files)
        except:
            os.remove(f) # if it can't be removed by shutil then use os.remove()

    return



def send_to_blob(blob_account_name, blob_account_key, tile, L1Cpath, check_blobs=False):

    """
    Function uploads processed L2A products to blob storage. Check if container matching tile name already exists -if so
    add files to that container, if not, create new container with name = tile.

    :param blob_account_name:
    :param blob_account_key:
    :param tile:
    :return:
    """

    block_blob_service = BlockBlobService(account_name=blob_account_name, account_key=blob_account_key)

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
        existing_containers = block_blob_service.list_containers()

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
                    block_blob_service.create_blob_from_path(container_name, destination, source)

                except:
                    print("Uploading to blob failed")


        else:
            print('\n',
                  ' CONTAINER DOES NOT ALREADY EXIST. CREATING NEW CONTAINER {} '.center(80, 'X').format(tile),
                  '\n', 'X'.center(80, 'X'))

            # Create a container with the tile as filename, then add files.
            block_blob_service.create_container(container_name)

            print('\n', ' CONTAINER CREATED. UPLOADING FOLDERS TO NEW CONTAINER '.center(80, 'X'),
                  '\n', 'X'.center(80, 'X'))

            for i in np.arange(0, len(filtered_paths)):
                source = str(filtered_paths[i])
                destination = str(folder + '/' + filtered_names[i])

                try:
                    block_blob_service.create_blob_from_path(container_name, destination, source)

                except:
                    print("Uploading to blob failed")

    if check_blobs:

        blob_list = []
        print("Retrieving blobs in specified container...")

        try:
            content = block_blob_service.list_blobs(container_name)
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



def remove_L2A(L1Cpath, upload_status): # delete unused L1C products

    """
    Once files have been sent to blob, remove them from local storage

    :param L1Cpath: path to wdir
    :return:
    """

    files = glob.glob(str(L1Cpath+'*L2A*')) # find files containing "L1C"

    if upload_status == 1:
        for f in files:
            try:
                shutil.rmtree(f) # try removing using shutil.rmtree() (for unzipped files)
            except:
                os.remove(f) # if it can't be removed by shutil it is likely zipped - use os.remove()

    else:
        print('\n', ' S2A FILES NOT UPLOADED TO BLOB: S2A FILES NOT DELETED FROM LOCAL STORAGE '.center(80, 'X'), '\n',
              'X'.center(80, 'X'))

    return


####################################################
################ DEFINE VARIABLES ##################
####################################################


# all tiles refers to all the individual tiles that cover the GRUS dark zone
tiles = ['22XDF', '21XWD', '21XWC', '22WEV', '22WEU', '22WEA', '22WEB', '22WED', '21XWB', '21XXA', '21WEC', '21XVD', '22WES', '22VER', '22WET']
year = 2017
months = [6,7,8] # specify months of the year to iterate through

# set up loop to iterate through tiles

for i in np.arange(0,len(tiles),1):

    # set up loop to iterate through batches of dates
    for ii in np.arange(0,len(months),1):

        month = months[ii]
        # set start and end dates
        startDate = date(year, month, 1)
        # adjust end date for number of days in month
        if month in [1,3,5,7,8,10,12]:
            endDate = date(year,month,31)
        elif month in [4,6,9,11]:
            endDate = date(year,month,30)
        elif month in [2]:
            endDate = date(year,month,28)

        # dates creates a tuple from the user-defined start and end dates
        dates = (startDate, endDate)

        # set path to save downloads
        L1Cpath = '/data/home/tothepoles/PycharmProjects/IceSurfClassifiers/Sentinel_BigStore/'

        # Define tiles, dates and save path
        api = SentinelAPI('jmcook1186', 'V66e6XAEeMqzaY', 'https://scihub.copernicus.eu/apihub')

        # define blob access
        blob_account_name = 'tothepoles'
        blob_account_key = 'HwYM3ZVtNv3j14/3iF57Zb9qIA7O5DTcB9Xx7pEoG1Ctw0fqJ7W5/JMSxfzKwp5tULqYVqH42dbKigvRg2QJqw=='

        tile = tiles[i]

        print("\n CURRENT TILE_ID = ", tile)

        ####################################################
        ################# RUN FUNCTIONS ####################
        ####################################################

        try: # use try/except so that any problems with individual tiles do not crash entire run
            L1Cfiles = download_L1C(L1Cpath, tile, dates)
            process_L1C_to_L2A(L1Cpath,L1Cfiles, unzip_files=True)
            remove_L1C(L1Cpath)
            upload_status = send_to_blob(blob_account_name, blob_account_key, tile, L1Cpath, check_blobs=True)
            remove_L2A(L1Cpath, upload_status)

            print("\n TILE ID {} FINISHED".format(tile))
            print('\n MOVING TO NEXT TILE '.center(80, 'X'), '\n', 'X'.center(80, 'X'))

        except:
            print("\n No images found for this tile on the specified dates")

print('X'.center(80,'X'), ' FINISHED ALL TILES '.center(80,'X'),'\n','X'.center(80,'X'))