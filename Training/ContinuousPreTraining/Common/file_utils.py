"""
Utilities for working with the local dataset cache.
"""

import traceback
import os
import re
import gzip
import logging
import shutil
import tempfile
import json
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional, Tuple, Union, IO, Callable, Set
from hashlib import sha256
from functools import wraps
from PIL import Image
from io import BytesIO
import sys
from pkgutil import iter_modules
from setuptools import find_packages

import boto3
import botocore
from botocore.exceptions import ClientError
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry  # pylint: disable=import-error

import tqdm as Tqdm

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
# logger.setLevel( logging.DEBUG )

CACHE_ROOT = Path(os.getenv('TURNINGTABLES_CACHE_ROOT', Path.home() / '.turningtables'))
CACHE_DIRECTORY = str(CACHE_ROOT / "cache")
DEPRECATED_CACHE_DIRECTORY = str(CACHE_ROOT / "datasets")

# This variable was deprecated in 0.7.2 since we use a single folder for caching
# all types of files (datasets, models, etc.)
DATASET_CACHE = CACHE_DIRECTORY

# Warn if the user is still using the deprecated cache directory.
if os.path.exists(DEPRECATED_CACHE_DIRECTORY):
    logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
    logger.warning(f"Deprecated cache directory found ({DEPRECATED_CACHE_DIRECTORY}).  "
                   f"Please remove this directory from your system to free up space.")


def url_to_filename(url: str, etag: str = None) -> str:
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode('utf-8')
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()

    return filename


def filename_to_url(filename: str, cache_dir: str = None) -> Tuple[str, str]:
    """
    Return the url and etag (which may be ``None``) stored for `filename`.
    Raise ``FileNotFoundError`` if `filename` or its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIRECTORY

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise FileNotFoundError("file {} not found".format(cache_path))

    meta_path = cache_path + '.json'
    if not os.path.exists(meta_path):
        raise FileNotFoundError("file {} not found".format(meta_path))

    with open(meta_path) as meta_file:
        metadata = json.load(meta_file)
    url = metadata['url']
    etag = metadata['etag']

    return url, etag


def cached_path(url_or_filename: Union[str, Path], cache_dir: str = None) -> str:
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIRECTORY
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)

    url_or_filename = os.path.expanduser(url_or_filename)
    parsed = urlparse(url_or_filename)

    if parsed.scheme in ('http', 'https', 's3'):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, cache_dir)
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == '':
        # File, but it doesn't exist.
        raise FileNotFoundError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))


def is_url_or_existing_file(url_or_filename: Union[str, Path, None]) -> bool:
    """
    Given something that might be a URL (or might be a local path),
    determine check if it's url or an existing file path.
    """
    if url_or_filename is None:
        return False
    url_or_filename = os.path.expanduser(str(url_or_filename))
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ('http', 'https', 's3') or os.path.exists(url_or_filename)


def split_s3_path(url: str) -> Tuple[str, str]:
    """Split a full s3 path into the bucket name and path."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError("bad s3 path {}".format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    # Remove '/' at beginning of path.
    if s3_path.startswith("/"):
        s3_path = s3_path[1:]
    return bucket_name, s3_path

def s3_request(func: Callable):
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """

    @wraps(func)
    def wrapper(url: str, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response["Error"]["Code"]) == 404:
                raise FileNotFoundError("file {} not found".format(url))
            else:
                raise

    return wrapper

def get_s3_resource():
    session = boto3.session.Session()
    if session.get_credentials() is None:
        # Use unsigned requests.
        s3_resource = session.resource("s3", config=botocore.client.Config(signature_version=botocore.UNSIGNED))
    else:
        s3_resource = session.resource("s3")
    return s3_resource

@s3_request
def s3_etag(url: str) -> Optional[str]:
    """Check ETag on S3 object."""
    s3_resource = get_s3_resource()
    bucket_name, s3_path = split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag


@s3_request
def s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = get_s3_resource()
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)


def session_with_backoff() -> requests.Session:
    """
    We ran into an issue where http requests to s3 were timing out,
    possibly because we were making too many requests too quickly.
    This helper function returns a requests session that has retry-with-backoff
    built in.
    see stackoverflow.com/questions/23267409/how-to-implement-retry-mechanism-into-python-requests-library
    """
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    return session


def http_get(url: str, temp_file: IO) -> None:
    with session_with_backoff() as session:
        req = session.get(url, stream=True)
        content_length = req.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None
        progress = Tqdm.tqdm(unit="B", total=total)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                temp_file.write(chunk)
        progress.close()


def get_from_cache(url: str, cache_dir: str = None) -> str:
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIRECTORY

    os.makedirs(cache_dir, exist_ok=True)

    # Get eTag to add to filename, if it exists.
    if url.startswith("s3://"):
        logger.info(f'''Cached path getting {url} from S3. Called by 
{traceback.extract_stack()[-4].filename.split('/')[-1]} {traceback.extract_stack()[-4].lineno}, 
{traceback.extract_stack()[-3].filename.split('/')[-1]} {traceback.extract_stack()[-3].lineno}''')
        etag = s3_etag(url)
    else:
        with session_with_backoff() as session:
            response = session.head(url, allow_redirects=True)
        if response.status_code != 200:
            raise IOError("HEAD request failed for url {} with status code {}"
                          .format(url, response.status_code))
        etag = response.headers.get("ETag")

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    if not os.path.exists(cache_path):
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with tempfile.NamedTemporaryFile() as temp_file:
            logger.info("%s not found in cache, downloading to %s", url, temp_file.name)

            # GET file object
            if url.startswith("s3://"):
                s3_get(url, temp_file)
            else:
                http_get(url, temp_file)

            # we are copying the file before closing it, so flush to avoid truncation
            temp_file.flush()
            # shutil.copyfileobj() starts at the current position, so go to the start
            temp_file.seek(0)

            logger.info("copying %s to cache at %s", temp_file.name, cache_path)
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            logger.info("creating metadata file for %s", cache_path)
            meta = {'url': url, 'etag': etag}
            meta_path = cache_path + '.json'
            with open(meta_path, 'w') as meta_file:
                json.dump(meta, meta_file)

            logger.info("removing temp file %s", temp_file.name)

    return cache_path


def read_set_from_file(filename: str) -> Set[str]:
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set()
    with open(filename, 'r') as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection


def get_file_extension(path: str, dot=True, lower: bool = True):
    ext = os.path.splitext(path)[1]
    ext = ext if dot else ext[1:]
    return ext.lower() if lower else ext


def get_json_from_s3_jsonl(s3_path, key_name="key", value_name="value"):
    j = {}

    if s3_path.endswith('.gz'):
        f = gzip.open(cached_path(s3_path), 'rb')
    else:
        f = open(cached_path(s3_path), 'rb', encoding='utf8')

    for l in f:
        l_j = json.loads(l.decode('utf-8'))
        j[l_j[key_name]] = l_j[value_name]

    f.close()
    return j


def is_path_creatable(pathname: str) -> bool:
    '''
    `True` if the current user has sufficient permissions to create the passed
    pathname; `False` otherwise.
    '''
    # Parent directory of the passed path. If empty, we substitute the current
    # working directory (CWD) instead.
    dirname = os.path.dirname(pathname) or os.getcwd()
    return os.access(dirname, os.W_OK)


def save_jsonl_to_local(local_filename, instances, header=None, sample_indent=False):
    if local_filename.endswith('.gz'):
        with gzip.open(local_filename, "wb") as f:
            # first JSON line is header
            if header is not None:
                f.write((json.dumps({'header': header}) + '\n').encode('utf-8'))
            for instance in instances:
                f.write((json.dumps(instance) + '\n').encode('utf-8'))
    else:
        with open(local_filename, "w") as f:
            # first JSON line is header
            if header is not None:
                f.write((json.dumps({'header': header}) + '\n'))
            for instance in instances:
                if sample_indent:
                    s = json.dumps(instance, indent=4)
                    # just making the answer starts in the sample no have a newline for every offset..
                    s = re.sub('\n\s*(\d+)', r'\1', s)
                    s = re.sub('(\d+)\n\s*]', r'\1]', s)
                    s = re.sub('(\d+)],\n\s*', r'\1],', s)
                    s = re.sub('\[\s*\n', r'[', s)
                    s = re.sub('\[\s*', r'[', s)
                    s = re.sub('",\s*\n', r'",', s)
                    s = re.sub('",\s*', r'", ', s)
                    s = re.sub('},\s*\n', r'},', s)
                    s = re.sub('},\s*', r'}, ', s)
                    s = re.sub('"\s*\n\s*}', r'"}', s)
                    f.write(s + '\n')
                else:
                    f.write((json.dumps(instance) + '\n'))


def verify_file_exists_on_s3(s3_bucket, file_path, s3_client=None):
    if s3_client == None:
        s3_client = boto3.client('s3')
    waiter = s3_client.get_waiter('object_exists')
    status = waiter.wait(
        Bucket=s3_bucket,
        Key=file_path, WaiterConfig={
            'Delay': 2,
            'MaxAttempts': 3
        })
    if status is not None:
        logger.error(f'{s3_bucket}/{file_path} not found on s3')
    else:
        logger.debug(f'{s3_bucket}/{file_path} correctly found on s3')

def list_contents_of_s3_directory(bucket, directory_prefix):
    directory_contents = []
    s3 = boto3.client('s3')
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=directory_prefix):
        directory_contents.extend([o['Key'] for o in page['Contents']])
    return directory_contents


def upper_to_lower_notation_name(dataset_reader_name):
    return ''.join(['_' + c.lower() if c.isupper() else c for c in dataset_reader_name])[1:]


def find_module(path, callange_to_find):
    """
    method to find a module from a specific path
    """
    modules = list()
    for pkg in [''] + find_packages(path):
        pkgpath = path + '/' + pkg.replace('.', '/')
        if sys.version_info.major == 2 or (sys.version_info.major == 3 and sys.version_info.minor < 6):
            for _, name, ispkg in iter_modules([pkgpath]):
                if not ispkg:
                    modules.append(pkg + '.' + name)
        else:
            for info in iter_modules([pkgpath]):
                if not info.ispkg:
                    modules.append(pkg + '.' + info.name)

    found_module = [module for module in modules if module.find(callange_to_find) > -1]
    if len(found_module) > 0:
        found_module = found_module[0]
        if found_module.startswith('.'):
            found_module = found_module[1:]
    else:
        found_module = None

    return found_module
