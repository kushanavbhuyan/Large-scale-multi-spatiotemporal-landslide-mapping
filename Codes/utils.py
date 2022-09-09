# -*- coding: utf-8 -*-

"""
Authors: Kushanav Bhuyan and Lorenzo Nava
"""

import os
from itertools import product
import rasterio as rio
from rasterio import windows
import skimage
import skimage.io
import numpy as np

from rasterio.plot import reshape_as_image
import rasterio.mask
from rasterio.features import rasterize

import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, Point, Polygon
from shapely.ops import cascaded_union

######################## load image tif ###############################
from typing import Any


def open_tif_image(input_path):
    # type: (function) -> np.array
    """Function to open tif images.

        Parameters:

        input_path (string) = path where the image file is located;

        return 
        
        np.array of the tif image"""

    # get the image_path.
    image_path = input_path
    # read image
    im = skimage.io.imread(image_path, plugin="tifffile")
    return im


def get_list_file_names(folder_path):
    
    """ Function to get the name of all files in folder.
    
    Parameters:
    
    folder_path (string) = path to the folder where all the .tif image files are located
    
    return 
    
    (list) with all the files within the folder_path """

    image_names = next(os.walk(folder_path))[2]
    if ".DS_Store" in image_names:
        image_names.remove(".DS_Store")
    print("Number of images: {}".format(len(image_names)))
    return image_names


def get_list_of_images_and_masks_with_same_dimension(image_names, images_path, masks_path, size):

    """Function to get the arrays of the .tif images and masks from a folder

        Parameters:

        image_names(list) = list of file names.

        images_path(string) = path to the folder where .tif image files are located.

        masks_path(string) = path to the folder where all the .tif mask files are located.

        size (int) = size of the height and width of the dimensions of the image.

        return 
        (array) of images, masks

        """
    images = []
    masks = []
    i = 0
    for image in sorted(image_names):
        current_image = open_tif_image(images_path + image)  # type: np.array
        current_mask = open_tif_image(masks_path + image)  # type: np.array
        i+= 1
        if current_image.shape[0] == size and current_image.shape[1] == size and current_mask.shape[0] == size and current_mask.shape[1] == size:
                images.append(current_image)
                masks.append(current_mask)
    print("Images shape: {}, Mask shape: {}".format(len(images), len(masks)))
    image = np.array(images, dtype="uint32")
    mask = np.expand_dims(np.array(masks, dtype="uint32"), axis=3)
    print("Images shape: {}, Mask shape: {}".format(image.shape, mask.shape))
    return image, mask


def save_array(image_array, output_file_name, outputfolder):
    """Function to save a numpy array to a specific folder and name.

    Parameters:

    image_array = np.array file.

    output_file_name = Name of the file that will be saved.

    outputfolder - path to the folder where the data will be saved.


    """

    np.save(outputfolder + output_file_name , image_array)
    print("Image saved on {}".format(outputfolder))



######################## Pre Processing ###############################
def patch_images(input_path,  output_path, size):
    
    """Function to patch the images in an specific size to a folder.
    
    Parameters:
    
    input_path(string) = path where the image file is located
    
    output_path(string) = path where the image tiles is located
    
    size (int) = crop size(width and height size will be the same during the crop)
    
    
    """

    size = size
    i = 0
    in_path = input_path


    out_path = output_path
    output_filename = 'tile_{}.tif'

    def get_tiles(ds, width=size, height=size):
        nols, nrows = ds.meta['width'], ds.meta['height']
        offsets = product(range(0, nols, width), range(0, nrows, height))
        big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
        for col_off, row_off in  offsets:
            window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
            transform = windows.transform(window, ds.transform)
            yield window, transform


    with rio.open(input_path) as inds:
        tile_width, tile_height = size, size

        meta = inds.meta.copy()

        for window, transform in get_tiles(inds):
            print(window)
            meta['transform'] = transform
            meta['width'], meta['height'] = window.width, window.height
            i += 1
            outpath = os.path.join(out_path,output_filename.format(i))
            with rio.open(outpath, 'w', **meta) as outds:
                outds.write(inds.read(window=window))





def get_nonzero_files(images_array,masks_array):

    """
    Function to evaluate all mask arrays and return just the files that have non zero masks.

    Parameters:

    images_array = array of images.

    mask_array = array of masks.

    :return images, masks
    """
    # 5. Delete all zeros

    # Delete files with just zeros in the mask
    all_zeros = []
    for i in range(masks_array.shape[0]):
        if masks_array[i].max() == 0:
            all_zeros.append(i)
    print("There are: {} arrays with just 0 values, and {} arrays with non zero values ".format(len(all_zeros), (
                images_array.shape[0] - len(all_zeros))))
    images = []
    masks = []
    for i in range(images_array.shape[0]):
        if i not in all_zeros:
            images.append(images_array[i])
            masks.append(masks_array[i])

    # Convert to array
    images = np.array(images, dtype="float32")
    masks = np.array(masks, dtype="float32")
    print("Image shape: {}, Mask shape: {}".format(images.shape, masks.shape))
    return images, masks


def generate_mask(raster_path, shape_path, output_path, file_name):
    """ Function to generate a mask from polygons with the same dimensions of an image. 

    raster_path = path to .tif image;

    shape_path = path to shapefile or geojson.

    output_path = path to save the binary mask.

    file_name = name of the saved file.

    """

    # Carregar o Raster

    with rio.open(raster_path, "r") as src:
        raster_img = src.read()
        raster_meta = src.meta

    # Carregar o shapefile ou GeoJson
    train_df = gpd.read_file(shape_path)

    # Verificar se o CRS é o mesmo
    if train_df.crs != src.crs:
        print(
            " Raster CRS : {}  Vetor CRS : {}.\n Convert to the same CRS!".format(
                src.crs, train_df.crs))

    # Função para gerar a máscara
    def poly_from_utm(polygon, transform):
        poly_pts = []

        poly = cascaded_union(polygon)
        for i in np.array(poly.exterior.coords):
            poly_pts.append(~transform * tuple(i))

        new_poly = Polygon(poly_pts)
        return new_poly

    poly_shp = []
    im_size = (src.meta['height'], src.meta['width'])
    for num, row in train_df.iterrows():
        if row['geometry'].geom_type == 'Polygon':
            poly = poly_from_utm(row['geometry'], src.meta['transform'])
            poly_shp.append(poly)
        else:
            for p in row['geometry']:
                poly = poly_from_utm(p, src.meta['transform'])
                poly_shp.append(poly)

    mask = rasterize(shapes=poly_shp,
                     out_shape=im_size)

    # Salvar
    mask = mask.astype("uint8")

    bin_mask_meta = src.meta.copy()
    bin_mask_meta.update({'count': 1})
    with rio.open(f"{output_path}/{file_name}" , 'w', **bin_mask_meta) as dst:
        dst.write(mask * 255, 1)






