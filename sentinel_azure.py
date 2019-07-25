from azure.storage.blob import BlockBlobService

def download_imgs_by_date(tile, date, img_path, blob_account_name, blob_account_key):

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

    # define blob access
    block_blob_service = BlockBlobService(account_name = blob_account_name, account_key= blob_account_key)

    # setup list
    bloblist = []
    download_flag = False
    QCflag = False

    # append names of all blobs to bloblist
    generator = block_blob_service.list_blobs(tile)
    for blob in generator:
        bloblist.append(blob.name)

    # filter total bloblist to just jp2s, then just for the specified date
    filtered_by_type = [string for string in bloblist if '_20m.jp2' in string]
    filtered_bloblist = [string for string in filtered_by_type if str("L2A_" + date) in string]


    # download the files in the filtered list
    for i in filtered_bloblist:
        print(i)
        try:
            block_blob_service.get_blob_to_path(tile,
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