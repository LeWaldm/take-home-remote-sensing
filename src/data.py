from torch.utils.data import Dataset
from copy import deepcopy
import torch
from osgeo import gdal
from rasterio.features import rasterize
import rasterio
import json
import urllib.request, json 
import matplotlib.pyplot as plt

class Observation_dataset(Dataset):
    def __init__(self, data, sample_heigth=96, sample_width=96) -> None:
        super().__init__()
        data = deepcopy(data)

        # load data into RAM (not feasible for large data volumes)
        s2 = rasterio.open(data['uri_to_s2'])
        # data['data_s2'] = torch.from_numpy(s2.read()) # s2.read() very slow ...
        data['data_s2'] = torch.from_numpy(gdal.Open(data['uri_to_s2']).ReadAsArray()).float()
        data['data_rgb'] = torch.from_numpy(gdal.Open(data['uri_to_rgb']).ReadAsArray())
        with urllib.request.urlopen(data['uri_to_annotations']) as url:
            data['data_annotations'] = json.load(url)
        with urllib.request.urlopen(data['uri_to_rivers']) as url:
            data['data_rivers'] = json.load(url)
        self.data = data

        # check if sample_heigth and sample_width are valid
        if sample_heigth > s2.height:
            raise ValueError('Too large sample_height.')
        if sample_width > s2.width:
            raise ValueError('Too large sample_width.')
        self.sample_height = sample_heigth
        self.sample_width = sample_width
        self.img_width = s2.width
        self.img_height = s2.height

        # calculate length for better performance
        self._length = (s2.width - self.sample_width + 1) * \
            (s2.height - self.sample_height + 1)
        
        # rasterize annotation classes 
        shapes = []
        for f in data['data_annotations']['features']:
            shapes.append( (f['geometry'],1))
        mask = rasterize(
            shapes,
            (s2.height, s2.width),
            transform = s2.transform)
        self.data['data_classes'] = torch.from_numpy(mask).long()

    def __len__(self):
        return self._length

    def __getitem__(self, index, return_rgb=False):
        # if index >= self._length:
        #     raise ValueError(f'Index {index} out of bounds (max is {self._length})')
        top_left_px_idx = torch.tensor(index)
        
        idx_height, idx_width = self._idx2heightwidth(index)
        s2 = self.data['data_s2'][:,
                                  idx_height:idx_height+self.sample_height,
                                  idx_width:idx_width+self.sample_width ]
        classes = self.data['data_classes'][
                                  idx_height:idx_height+self.sample_height,
                                  idx_width:idx_width+self.sample_width ]
        if return_rgb:
            rgb = self.data['data_rgb'][:,
                                  idx_height:idx_height+self.sample_height,
                                  idx_width:idx_width+self.sample_width ]
            return s2, classes, top_left_px_idx, rgb
        return s2, classes, top_left_px_idx

    def visualize_batch(self, top_left_idxs):
        fig,ax = plt.subplots()
        plt.imshow(self.data['data_rgb'].permute(1,2,0))

        # plot boxes for each sample
        style = {'c': 'yellow'}
        for idx in top_left_idxs:
            ul_h, ul_w = self._idx2heightwidth(idx)
            plt.plot([ul_w, ul_w+self.sample_width-1], [ul_h,ul_h], **style)
            plt.plot([ul_w, ul_w+self.sample_width-1], 
                     [ul_h+self.sample_height-1, ul_h+self.sample_height-1], **style)
            plt.plot([ul_w, ul_w], [ul_h, ul_h+self.sample_height-1], **style)
            plt.plot([ul_w+self.sample_width-1, ul_w+self.sample_width-1], 
                     [ul_h, ul_h+self.sample_height-1], **style)

    def _idx2heightwidth(self, idx):
        width = self.img_width - self.sample_width + 1
        idx_width = idx % width
        idx_height = idx // width
        return idx_height, idx_width
        