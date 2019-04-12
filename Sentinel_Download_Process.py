"""
This function uses the sentinelsat python API to download batches of Sentinel 2 Level 1C images from SentinelHub,
process them into L2A products using Sen2Cor and then save them to file.

NB Each product is 600MB zipped, up to 800MB when converted to L2A.Recommend downloading batches of ~20 tiles, sending
to blob storage then deleting from hard drive, then repeating batch.

Takes about 1 minute to download each tile on Azure ES20_v3 Linux DSVM, then up to 15 minutes to process L1Cto L2A

"""

from collections import OrderedDict
from sentinelsat import SentinelAPI
from datetime import date
from azure.storage.blob import BlockBlobService, PublicAccess
import os
import shutil



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
    for tilename in tile:
        kw = query_kwargs.copy()
        kw['tileid'] = tilename  # products after 2017-03-31
        pp = api.query(**kw)
        products.update(pp)

    # keep metadata in pandas dataframe
    metadata = api.to_dataframe(products)
    L1Cfiles = metadata.filename
    print("\n***** DOWNLOADING {} FILES *****. \n \n*** FILENAMES *** "
          "\n****************** \n\n {}".format(len(L1Cfiles),L1Cfiles))

    # download all files
    api.download_all(products, directory_path = L1Cpath)


    return L1Cfiles


def process_L1C_to_L2A(L1C_path, L1Cfiles, unzip_files = True, removeL1C = True, cleanupL2A = True):
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
        sen2cor = str(
            '/home/tothepoles/PycharmProjects/Sen2Cor-02.05.05-Linux64/bin/L2A_Process ' + L1C_path + '{}'.format(L1C) + ' --resolution=20')
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
            os.remove(f) # if it can't be removed by shutil it is likely zipped - use os.remove()

    return



def send_to_blob(blob_account_name, blob_account_key, tile):

    """
    Function uploads processed L2A products to blob storage. Check if container matching tile name already exists -if so
    add files to that container, if not, create new container with name = tile.

    :param blob_account_name:
    :param blob_account_key:
    :param tile:
    :return:
    """
    block_blob_service = BlockBlobService(account_name=blob_account_name, account_key=blob_account_key)

    container_name = tile[0].lower() #convert string to lower case because Azure blobs named in lower case

    local_path = L1Cpath

    containers = block_blob_service.list_containers()

    if any(tile[0].lower() in filenames for filenames in containers):
        # add files to existing container matching tile name
        for file in os.listdir(local_path):
            block_blob_service.create_blob_from_path(container_name, file, os.path.join(local_path, file))

    else:
        # Create a container with the tile as filename, then add files.
        block_blob_service.create_container(container_name)
        for file in os.listdir(local_path):
            block_blob_service.create_blob_from_path(container_name,file,os.path.join(local_path,file))

    return



def remove_L2A(L1Cpath): # delete unused L1C products

    """
    Once files have been sent to blob, remove them from local storage

    :param L1Cpath: path to wdir
    :return:
    """

    files = glob.glob(str(L1Cpath+'*L2A*')) # find files containing "L1C"

    for f in files:
        try:
            shutil.rmtree(f) # try removing using shutil.rmtree() (for unzipped files)
        except:
            os.remove(f) # if it can't be removed by shutil it is likely zipped - use os.remove()

    return


######### DEFINE VARIABLES #########
####################################

# Define tiles, dates and save path
api = SentinelAPI('jmcook1186', 'V66e6XAEeMqzaY','https://scihub.copernicus.eu/apihub')

# define blob access
blob_account_name = 'tothepoles'
blob_account_key = 'HwYM3ZVtNv3j14/3iF57Zb9qIA7O5DTcB9Xx7pEoG1Ctw0fqJ7W5/JMSxfzKwp5tULqYVqH42dbKigvRg2QJqw=='

# all tiles refers to all the individual tiles that cover the GRUS dark zone
alltiles = ['22XDF', '21XWD', '21XWC', '22WEV', '22WEU', '22WEB', '22WEA', '22WED', '21XWB', '21XXA', '21WEC', '21XVD', '22WES', '22VER', '22WET']

# add tile name to 'completed tiles' when it has been downloaded and processed to L2A (manually, just to keep track)
completed_tiles = ['22XDF','21XWD', "21XWC", '22WEV']

#tiles refers to those tiles to download now
tile=['22WEV']

#set start and end dates
startDate = date(2017,6,1)
endDate = date(2017,6,30)

# dates creates a tuple from the user-defined start and end dates
dates= (startDate, endDate)

# set path to save downloads
L1Cpath = '/data/home/tothepoles/PycharmProjects/IceSurfClassifiers/Sentinel_BigStore/'



######### RUN FUNCTIONS #########
#################################

L1Cfiles = download_L1C(L1Cpath, tile, dates)
process_L1C_to_L2A(L1Cpath,L1Cfiles, unzip_files=True, removeL1C=True, cleanupL2A=True)
remove_L1C(L1Cpath)
send_to_blob(blob_account_name, blob_account_key, tile)
remove_L2A(L1Cpath)