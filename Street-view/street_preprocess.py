import os
import glob

import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt
import shutil

from ast import literal_eval
from osgeo import gdal
from osgeo import osr

key = 'AIzaSyD854_OK7Ak3_5plz1TkMnWK--AMEBUt9w'


def street_view_metadata_retrival(lat, long):
    response = requests.get(
        url='https://maps.googleapis.com/maps/api/streetview/metadata',
        params={
            'location': str(lat) + ',' + str(long),
            # 'size': '512x512',
            'key': key,
        }
    )
    ret = response.json()
    return ret


def street_view_location_transform():
    data_dir = '/home/xiaomou/Codes/koeln'
    metadata = []
    with open('available_street_view_metadata.txt', 'r') as f:
        for line in f:
            metadata.append(literal_eval(line))

    reliable_metadata = []
    img_ids = list(set([i['im_id'] for i in metadata]))
    for idx, img_id in enumerate(img_ids):
        print(idx+1, '/', len(img_ids))
        img_rgb = gdal.Open(os.path.join(data_dir, img_id+'_rgb.jp2'))
        x_origin, x_res, _, y_origin, _, y_res = img_rgb.GetGeoTransform()
        x_size, y_size = img_rgb.RasterXSize, img_rgb.RasterYSize

        img_show = img_rgb.ReadAsArray()[0:3, :, :].transpose(1, 2, 0)
        plt.imshow(img_show)

        street_info = [i for i in metadata if i['im_id'] == img_id]
        for s in street_info:
            lat, lng = s['location']['lat'], s['location']['lng']
            x, y = coord_transform(img_rgb, loc=(lat, lng), geo_to_proj=True)
            x_cur = round((x - x_origin) / x_res)
            y_cur = round((y - y_origin) / y_res)
            if x_cur > x_size or y_cur > y_size or x_cur < 0 or y_cur < 0:
                continue
            elif s['copyright'] == '© Google':
                s['coord'] = {'x': x_cur, 'y': y_cur}
                reliable_metadata.append(str(s)+'\n')
            plt.scatter(x_cur, y_cur, s=3, c='r')
        plt.savefig(os.path.join('street_view_locations', img_id + '.png'), dpi=300)
        plt.show()
        with open('reliable_metadata.txt', 'w') as f:
            f.writelines(reliable_metadata)


def street_view_image_retrival():
    metadata = []
    with open('reliable_metadata.txt', 'r') as f:
        for line in f:
            metadata.append(literal_eval(line))

    for idx, info in enumerate(metadata):
        for heading in [0, 90, 180, 270]:
            st_im_name = info['pano_id'] + '_' + str(heading) + '.jpeg'
            if os.path.exists(os.path.join('street_view_images', st_im_name)):
                print("exists.")
                continue
            response = requests.get(
                url='https://maps.googleapis.com/maps/api/streetview',
                params={
                    'pano': info['pano_id'],
                    'heading': heading,
                    'size': '512x512',
                    'return_error_code': 'true',
                    'key': key,
                }
            )
            with open(os.path.join('street_view_images', st_im_name), 'wb+') as f:
                f.write(response.content)
        print(idx+1, '/', len(metadata))


def coord_transform(gdal_obj, loc, geo_to_proj=False):
    EPSG = 4326  # WGS84 geographic reference system
    in_spatial_ref = osr.SpatialReference()
    out_spatial_ref = osr.SpatialReference()
    if geo_to_proj:
        in_spatial_ref.ImportFromEPSG(EPSG)
        out_spatial_ref.ImportFromWkt(gdal_obj.GetProjection())
    else:
        in_spatial_ref.ImportFromWkt(gdal_obj.GetProjection())
        out_spatial_ref.ImportFromEPSG(EPSG)

    coord_transform = osr.CoordinateTransformation(in_spatial_ref, out_spatial_ref)
    x, y, _ = coord_transform.TransformPoint(loc[0], loc[1])
    return x, y


def test():
    data_dir = '/home/xiaomou/Codes/koeln'
    img_id = '351_5645'
    img_rgb = gdal.Open(os.path.join(data_dir, img_id + '_rgb.jp2'))
    x_origin, x_res, _, y_origin, _, y_res = img_rgb.GetGeoTransform()
    x_size, y_size = img_rgb.RasterXSize, img_rgb.RasterYSize
    img_show = img_rgb.ReadAsArray()[0:3, :, :].transpose(1, 2, 0)
    plt.imshow(img_show)

    metadata = []
    with open('reliable_metadata.txt', 'r') as f:
        for line in f:
            metadata.append(literal_eval(line))
    street_info = [i for i in metadata if i['im_id'] == img_id]
    for s in street_info:
        if s['pano_id'] == 'RpAoWRlZfhd-6xJa_OYI_w':
            lat, lng = s['location']['lat'], s['location']['lng']
            x, y = coord_transform(img_rgb, loc=(lat, lng), geo_to_proj=True)
            x_cur = ((x-x_origin)/x_res)
            y_cur = ((y-y_origin)/y_res)
            plt.scatter(x_cur, y_cur, s=3, c='r')
            print(lat, lng, s['pano_id'])
            print(x_cur, y_cur)
            print(s['coord']['x'], s['coord']['y'])
    plt.show()


def main():
    data_dir = '/home/xiaomou/Codes/koeln'

    img_ids = sorted(glob.glob(os.path.join(data_dir, '*.jp2')))
    metadata = []
    for i in img_ids:
        # print(i, 'start')
        # read data

        im_id = os.path.split(i)[-1][:-8]

        img_rgb = gdal.Open(i)
        x_origin, x_res, _, y_origin, _, y_res = img_rgb.GetGeoTransform()
        x_size, y_size = img_rgb.RasterXSize, img_rgb.RasterYSize
        img_show = img_rgb.ReadAsArray()[0:3, :, :].transpose(1, 2, 0)
        plt.imshow(img_show)
        # Coordinate Transformation: x, y -> lat, long

        for x_offset in range(0, x_size, 100):
            for y_offset in range(0, y_size, 100):
                x_current = x_origin + (x_offset + 50) * x_res
                y_current = y_origin + (y_offset + 50) * y_res
                lat, lng = coord_transform(img_rgb, (x_current, y_current), geo_to_proj=False)
                # print(lat, lng, sep=',')
                ret = street_view_metadata_retrival(lat, lng)
                if ret.get('status') == 'OK':
                    ret['im_id'] = im_id
                    print(str(ret))
                    metadata.append(str(ret)+'\n')
                    print(type(ret))
                    c = input()
        plt.show()
    # with open('available_street_view_metadata.txt', 'w') as f:
    #     f.writelines(metadata)


def exclude_bad():
    metadata = []
    with open('reliable_metadata.txt', 'r') as f:
        for line in f:
            metadata.append(literal_eval(line))

    img_dir = './street_view_images'
    ex_cnt = 0
    for m in metadata:
        for heading in [0, 90, 180, 270]:
            name = m['pano_id']+'_'+str(heading)+'.jpeg'
            img_path = os.path.join(img_dir, name)
            if os.path.exists(img_path):
                img_store_size = os.path.getsize(img_path) / 1024
                if img_store_size < 24.8:
                    print(img_path, img_store_size)
                    ex_cnt += 1
                    shutil.move(img_path, os.path.join('./street_view_excluded', name))
    print(ex_cnt)


def get_street_sat():
    metadata = []
    with open('reliable_metadata.txt', 'r') as f:
        for line in f:
            metadata.append(literal_eval(line))

    sat_dir = '/home/xiaomou/Codes/height_estimation/Street-view/street_sat'

    for i, m in enumerate(metadata):
        if not os.path.exists(os.path.join(sat_dir, m['pano_id']+'.png')):
            continue
        rgb = cv2.imread(os.path.join(sat_dir, m['pano_id']+'.png'))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        height = cv2.imread(os.path.join(sat_dir, m['pano_id']+'.tif'), -1)
        plt.subplot(1,2,1)
        plt.imshow(rgb)
        plt.subplot(1,2,2)
        plt.imshow(height)
        plt.show()
        # print(i+1, '/', len(metadata))
        # sat_id, x, y = m['im_id'], m['coord']['x'], m['coord']['y']
        # n = 256
        # bottom, top, left, right = x - n // 2, x + n // 2, y - n // 2, y + n // 2
        # if bottom < 0 or top >= 1000 or left < 0 or right >= 1000:
        #     continue
        # sat_im = cv2.imread(os.path.join(sat_dir, sat_id+'_ndsm.tif'), -1)
        # sat_im_cropped = sat_im[left:right, bottom:top]
        # cv2.imwrite(os.path.join(out_dir, m['pano_id']+'.tif'), sat_im_cropped)
        c = input()


if __name__ == '__main__':
    # street_view_image_retrival()
    # exclude_bad()
    get_street_sat()
