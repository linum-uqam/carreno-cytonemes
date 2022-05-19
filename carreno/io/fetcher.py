# -*- coding: utf-8 -*-
import os
import gdown

def folders_id():
    return {
        'raw': '1wgd64deMQJ1G_tOwNuqb3I-F4N9ZNh4t',
        'psf': '1w9I63Dbby83eql_IwKUjFEm-gjBX6JAD'
    }

def fetch_file(url=None, id=None, md5=None, output='./', quiet=True):
    """
    Get file from Google Drive (url works for One Drive too)
    Parameters
    ----------
    url : str
        File URL
    id : str
        File id if we don't have the URL
    md5 : str
        key to validate download integrity (can be known via OAuth application)
    output : str
        Resulting file location, name and extension
    quiet : bool
        See `gdown` functions output
    Returns
    -------
    __ : str
        Path to downloaded file (relative path by default)
    """
    if not os.path.exists(output):
        os.makedirs(output)
        
    return gdown.download(url=url, id=id, md5=md5, output=output, quiet=quiet)


def fetch_folder(url=None, id=None, output='./', quiet=True):
    """
    Get folder from Google Drive (doesn't work with One Drive)
    Parameters
    ----------
    url : str
        Folder URL
    id : str
        Folder id if we don't have the URL
    md5 : str
        key to validate download integrity (can be known via OAuth application)
    output : str
        Resulting file location, name and extension
    quiet : bool
        See `gdown` functions output
    Returns
    -------
    __ : [str]
        Path for each downloaded files (relative path by default)
    """
    return gdown.download_folder(url=url, id=id, output=output, quiet=quiet)