from .blender import BlenderDataset
from .llff import LLFFDataset,LLFF_Chromatic_Dataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'llff_chromatic':LLFF_Chromatic_Dataset
                }