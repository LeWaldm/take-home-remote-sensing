from torch.utils.data import Dataset
from copy import deepcopy
import torch
from osgeo import gdal
from rasterio.features import rasterize
import rasterio
import json
import urllib.request, json 
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class Observation_dataset(Dataset):
    def __init__(self, data, sample_heigth=96, sample_width=96, only_river=False):
        super().__init__()
        data = deepcopy(data)
        self.sample_height = sample_heigth
        self.sample_width = sample_width
        self.only_river = only_river

        # process data
        indices = [0]
        for d in tqdm(data):

            # load data into RAM (not feasible for large data volumes)
            s2 = rasterio.open(d['uri_to_s2'])
            # data['data_s2'] = torch.from_numpy(s2.read()) # s2.read() very slow ...
            d['data_s2'] = torch.from_numpy(gdal.Open(d['uri_to_s2']).ReadAsArray()).float()
            d['data_rgb'] = torch.from_numpy(gdal.Open(d['uri_to_rgb']).ReadAsArray())
            with urllib.request.urlopen(d['uri_to_annotations']) as url:
                d['data_annotations'] = json.load(url)
            with urllib.request.urlopen(d['uri_to_rivers']) as url:
                d['data_rivers'] = json.load(url)

            # check if sample_heigth and sample_width are valid
            if sample_heigth > s2.height:
                raise ValueError('Too large sample_height.')
            if sample_width > s2.width:
                raise ValueError('Too large sample_width.')
            
            # rasterize annotation classes 
            shapes = []
            for f in d['data_annotations']['features']:
                shapes.append( (f['geometry'],1))
            mask = rasterize(
                shapes,
                (s2.height, s2.width),
                transform = s2.transform)
            d['data_classes'] = torch.from_numpy(mask).float()

            # prepare uniform access
            if only_river:
                shapes = []
                for f in d['data_rivers']['features']:
                    shapes.append( (f['geometry'],1))
                river_mask = rasterize(
                    shapes,
                    (s2.height, s2.width),
                    transform = s2.transform)
                height, width = np.where(river_mask == 1)
                valid = np.logical_and(height <= s2.height - self.sample_height, 
                                       width <= s2.width - self.sample_width)
                d['locs_height'] = height[valid]
                d['locs_width'] = width[valid]
                n_samples = d['locs_height'].shape[0]
            else:
                n_samples = (s2.width - self.sample_width + 1) * \
                    (s2.height - self.sample_height + 1)
            indices.append(n_samples)

            # preprocess river coordinates to pixels
            shapes = []
            for f in d['data_rivers']['features']:
                coords = f['geometry']['coordinates'][0]
                coords_pixels = []
                for c in coords:
                    height, width = s2.index(c[0],c[1])
                    coords_pixels.append((height, width))
                shapes.append(coords_pixels)
            d['river_bounds'] = shapes

        self._length = int(torch.tensor(indices).sum())
        self.indices = torch.tensor(indices).cumsum(0)
        self.data = data

    def __len__(self):
        return self._length

    def __getitem__(self, index, return_rgb=False):
        # if index >= self._length:
        #     raise ValueError(f'Index {index} out of bounds (max is {self._length})')
        
        # get correct indices
        obs_idx, idx_height, idx_width = self._idx2obsheightwidth(index)
        data = self.data[obs_idx]

        # extract data
        s2 = data['data_s2'][:,
                            idx_height:idx_height+self.sample_height,
                            idx_width:idx_width+self.sample_width ]
        classes = data['data_classes'][
                            idx_height:idx_height+self.sample_height,
                            idx_width:idx_width+self.sample_width ]
        if return_rgb:
            rgb = data['data_rgb'][:,
                            idx_height:idx_height+self.sample_height,
                            idx_width:idx_width+self.sample_width ]
            return s2, classes, index, rgb
        return s2, classes, index

    def visualize_batch(self, idxs):

        obs_idxs, idx_heights, idx_widths = [], [], []
        for i in idxs:
            oi, hi, wi = self._idx2obsheightwidth(i)
            obs_idxs.append(int(oi))
            idx_heights.append(hi)
            idx_widths.append(wi)
        nrows = len(set(obs_idxs))
        obs_order = list(set(obs_idxs))
        obs_order.sort()
        obs_order_rev = {o:i for i,o in enumerate(obs_order)}

        # plot background
        plt.subplots(nrows=nrows, ncols=1, figsize=(7, 3*nrows))
        for i,obs in enumerate(obs_order):

            # plot RGB
            plt.subplot(nrows,1, i+1)
            plt.imshow(self.data[obs]['data_rgb'].permute(1,2,0))

            # plot river
            for shape in self.data[obs]['river_bounds']:
                for i in range(len(shape)-1):
                    plt.plot([shape[i][1], shape[i+1][1]], 
                             [shape[i][0], shape[i+1][0]],
                             '-.', c='blue')
                

        # plot boxes for each sample
        style = {'c': 'yellow'}
        for oi, ul_h, ul_w in zip(obs_idxs, idx_heights, idx_widths):
            plt.subplot(nrows,1, obs_order_rev[oi]+1)
            plt.plot([ul_w, ul_w+self.sample_width-1], [ul_h,ul_h], **style)
            plt.plot([ul_w, ul_w+self.sample_width-1], 
                     [ul_h+self.sample_height-1, ul_h+self.sample_height-1], **style)
            plt.plot([ul_w, ul_w], [ul_h, ul_h+self.sample_height-1], **style)
            plt.plot([ul_w+self.sample_width-1, ul_w+self.sample_width-1], 
                     [ul_h, ul_h+self.sample_height-1], **style)

    def _idx2obsheightwidth(self, index):

        obs_idx = (index >= self.indices[1:]).int().sum().int()
        loc_idx = index - self.indices[obs_idx]

        if self.only_river:
            idx_width = self.data[obs_idx]['locs_width'][loc_idx]
            idx_height = self.data[obs_idx]['locs_height'][loc_idx]
        else:
            img_width = self.data[obs_idx]['data_s2'].shape[2]
            width = img_width - self.sample_width + 1
            idx_width = loc_idx % width
            idx_height = loc_idx // width

        return obs_idx, idx_height, idx_width
        