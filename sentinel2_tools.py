"""
Functions for local-system management of Sentinel-2 files.

"""


def download_L1C(L1Cpath, tile, dates, cloudcoverthreshold):
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
            'date': dates,
            'cloudcoverpercentage': (0, cloudcoverthreshold)}

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
    #api.download_all(products, directory_path = L1Cpath)


    return L1Cfiles



def process_L1C_to_L2A(L1C_path, L1Cfiles, S2_resolution, unzip_files=True):
    """
    This function is takes the downloaded L1C products from SentinelHub and converts them to L2A product using Sen2Cor.
    This is achieved as a batch job by iterating through the file names (L1Cfiles) stored in L1Cpath
    Running this script will save the L2A products to the working directory

    Requires 'SEN2COR_BIN' environment variable to be set

    :return: None

    """

    sen2cor_bin = os.environ['SEN2COR_BIN']
    if len(sen2cor_bin) == 0:
        print('Error: sen2cor bin path environment variable is not set.')
        raise OSError

    for L1C in L1Cfiles:

        # unzip downloaded files in place, then delete zipped folders
        if unzip_files:

            unzip = str('unzip '+ L1C_path + L1C[0:-4]+'zip'+' -d ' + L1C_path)
            os.system(unzip)

        # process L1C to L2A
        try:
            sen2cor = str(sen2cor_bin + '/L2A_Process ' + L1C_path + '{}'.format(L1C) + ' --resolution={}'.format(S2_resolution))
            os.system(sen2cor)

        except:
            print("Processing {} L1C to L2A failed. Skipping to next tile".format(L1C))
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