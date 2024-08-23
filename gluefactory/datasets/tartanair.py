import logging

import pickle
import torch
import torchvision.transforms as transforms
import kornia
import numpy as np
from collections.abc import Iterable

from .base_dataset import BaseDataset
from ..geometry.wrappers import Camera, Pose
from ..models.cache_loader import CacheLoader
from ..settings import DATA_PATH
from ..utils.image import load_image
from ..utils.tools import fork_rng

test_split = [
    "abandonedfactory/abandonedfactory/Easy/P011",
    "abandonedfactory/abandonedfactory/Hard/P011",
    "abandonedfactory_night/abandonedfactory_night/Easy/P013",
    "abandonedfactory_night/abandonedfactory_night/Hard/P014",
    "amusement/amusement/Easy/P008",
    "amusement/amusement/Hard/P007",
    "carwelding/carwelding/Easy/P007",
    "endofworld/endofworld/Easy/P009",
    "gascola/gascola/Easy/P008",
    "gascola/gascola/Hard/P009",
    "hospital/hospital/Easy/P036",
    "hospital/hospital/Hard/P049",
    "japanesealley/japanesealley/Easy/P007",
    "japanesealley/japanesealley/Hard/P005",
    "neighborhood/neighborhood/Easy/P021",
    "neighborhood/neighborhood/Hard/P017",
    "ocean/ocean/Easy/P013",
    "ocean/ocean/Hard/P009",
    "office2/office2/Easy/P011",
    "office2/office2/Hard/P010",
    "office/office/Hard/P007",
    "oldtown/oldtown/Easy/P007",
    "oldtown/oldtown/Hard/P008",
    "seasidetown/seasidetown/Easy/P009",
    "seasonsforest/seasonsforest/Easy/P011",
    "seasonsforest/seasonsforest/Hard/P006",
    "seasonsforest_winter/seasonsforest_winter/Easy/P009",
    "seasonsforest_winter/seasonsforest_winter/Hard/P018",
    "soulcity/soulcity/Easy/P012",
    "soulcity/soulcity/Hard/P009",
    "westerndesert/westerndesert/Easy/P013",
    "westerndesert/westerndesert/Hard/P007",
] # official test split

def remove_duplicated_segment(namestr):

        parts = namestr.split('/')
        unique_parts = []
        previous_part = None
        for part in parts:
            if part != previous_part:
                unique_parts.append(part)
            previous_part = part
        return '/'.join(unique_parts)
def is_test_scene(scene):
    return any(x in scene for x in test_split)

logger = logging.getLogger(__name__)

def quat2mat_numpy(quat):
    # Extract quaternion components
    x, y, z, w = quat

    # Precompute products to avoid redundant calculations
    tx = 2 * x
    ty = 2 * y
    tz = 2 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z

    # Create the 3x3 rotation matrix
    mat = np.zeros((3, 3))

    mat[0, 0] = 1 - (tyy + tzz)
    mat[0, 1] = txy - twz
    mat[0, 2] = txz + twy
    mat[1, 0] = txy + twz
    mat[1, 1] = 1 - (txx + tzz)
    mat[1, 2] = tyz - twx
    mat[2, 0] = txz - twy
    mat[2, 1] = tyz + twx
    mat[2, 2] = 1 - (txx + tyy)

    return mat

class RGBDAugmentor:
    """ perform augmentation on RGB-D video """

    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.augcolor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2/3.14),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomInvert(p=0.1),
            transforms.ToTensor()])

        self.max_scale = 0.5

    def spatial_transform(self, image, depth, intrinsics):
        """ cropping and resizing """
        ht, wd = image.shape[-2:]

        # create camera
        K = torch.zeros(2, 3, device=intrinsics.device, dtype=intrinsics.dtype)
        K[0, 0] = intrinsics[0]
        K[1, 1] = intrinsics[1]
        K[0, 2] = intrinsics[2]
        K[1, 2] = intrinsics[3]
        camera = Camera.from_calibration_matrix(K).float()

        # sample scale
        max_scale = self.max_scale
        min_scale = np.log2(np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd)))

        scale = 1
        if np.random.rand() < 0.8:
            scale = 2 ** np.random.uniform(0.0, max_scale)

        ht1 = int(scale * ht)
        wd1 = int(scale * wd)
        size = (ht1, wd1)

        # image = F.interpolate(image, (ht1, wd1), mode='bicubic', align_corners=False)
        # depth = F.interpolate(depth, (ht1, wd1), recompute_scale_factor=False)
        image = kornia.geometry.transform.resize(
            image,
            size,
            side="long",
            antialias=True,
            align_corners=None,
            interpolation="bilinear",
        )

        depth = kornia.geometry.transform.resize(
            depth,
            size,
            side="long",
            antialias=True,
            align_corners=None,
            interpolation="nearest",
        )
        # intrinsics = scale * intrinsics
        camera = camera.scale(scale)

        # always perform center crop (TODO: try non-center crops)
        y0 = (image.shape[-2] - self.crop_size[0]) // 2
        x0 = (image.shape[-1] - self.crop_size[1]) // 2

        image = image[..., y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        depth = depth[..., y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        # intrinsics = intrinsics - torch.tensor([0.0, 0.0, x0, y0])
        camera = camera.crop([x0, y0], self.crop_size[::-1])

        data = {
            "scales": None,
            "image_size": np.array(self.crop_size[::-1]),
            "transform": None,
            "original_image_size": np.array([wd, ht]),
            "camera": camera,
            "image": image,
            "depth": depth.squeeze(0),
        }

        return data

    def color_transform(self, images):
        """ color jittering """
        num, ch, ht, wd = images.shape
        images = images.permute(1, 2, 3, 0).reshape(ch, ht, wd*num)
        images = 255 * self.augcolor(images[[2,1,0]] / 255.0)
        return images[[2,1,0]].reshape(ch, ht, wd, num).permute(3,0,1,2).contiguous()

    def __call__(self, images, depths, intrinsics):
        if np.random.rand() < 0.5:
            images = self.color_transform(images)

        return self.spatial_transform(images, depths, intrinsics)
    
class TartanAir(BaseDataset):
    
    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    default_conf = {
        # paths
        "data_dir": "TartanAir",
        "info_file": "TartanAir.pickle",
        # data sampling
        "views": 2,
        "fmin": 1,  # min distance in graph
        "fmax": 10000,  # max distance in graph
        "reseed": False,
        "seed": 0,
        # features from cache
        "load_features": {
            "do": False,
            **CacheLoader.default_conf,
            "collate": False,
        },
    }

    def _init(self, conf):
        if not (DATA_PATH / 'datasets' / conf.data_dir).exists():
            logger.info("Downloading the TartanAir dataset.")
            self.download()

    def download(self):
        raise NotImplementedError("TartanAir download not implemented yet.")
    
    def get_dataset(self, split):
        return _Dataset(self.conf, split)
        
class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split, load_sample=True):
        self.root = DATA_PATH / 'datasets' / conf.data_dir
        assert self.root.exists(), self.root
        self.split = split
        assert conf.views == 2, conf.views
        self.conf = conf

        self.aug = RGBDAugmentor(crop_size=[480,640])

        scene_info_file = DATA_PATH / 'datasets' / conf.info_file
        self.scene_info = \
            pickle.load(open(scene_info_file, 'rb'))[0]
        
        self.scene_index = {}
        for scene in self.scene_info:
            if not is_test_scene(scene):
                graph = self.scene_info[scene]['graph']
                inx_list = []
                for i in graph:
                    if i < len(graph) - 65:
                        inx_list.append(i) # same training split as dpvo
                self.scene_index[scene] = inx_list
            else:
                logger.info("Reserving {} for validation.".format(scene))

            graph = self.scene_info[scene]['graph']
            images_list = self.scene_info[scene]['images']
            depths_list = self.scene_info[scene]['depths']
            assert len(images_list) == len(graph)
            # remove repeated part in item in image_list and depth_list
            # images_list = [remove_duplicated_segment(item) for item in images_list]
            # depths_list = [remove_duplicated_segment(item) for item in depths_list]
            self.scene_info[scene]['images'] = images_list
            self.scene_info[scene]['depths'] = depths_list

        if conf.load_features.do:
            self.feature_loader = CacheLoader(conf.load_features)

        self.sample_new_items(conf.seed)
        assert len(self.items) > 0

    def _read_view(self, scene, idx):
        img_path = DATA_PATH / self.scene_info[scene]['images'][idx]
        depth_path = DATA_PATH / self.scene_info[scene]['depths'][idx]
        intrinsics = self.scene_info[scene]['intrinsics'][idx]
        intrinsics = torch.from_numpy(intrinsics.astype(np.float32))

        pose = self.scene_info[scene]['poses'][idx]
        trans, quat = pose[:3], pose[3:]
        pose = np.eye(4)
        pose[:3, :3] = quat2mat_numpy(quat)
        pose[:3, -1] = trans
        pose = torch.from_numpy(pose.astype(np.float32))
    
        img = load_image(img_path, grayscale=False)
        depth = torch.from_numpy((np.load(depth_path) / TartanAir.DEPTH_SCALE).astype(np.float32))[None]
        assert depth.shape[-2:] == img.shape[-2:]

        data = \
                self.aug(img, depth, intrinsics)

        name = img_path.name

        data = {
            "name": name,
            "scene": scene,
            "T_w2cam": Pose.from_4x4mat(pose).inv(),
            "depth": depth,
            **data,
        }

        if self.conf.load_features.do:
            features = self.feature_loader({k: [v] for k, v in data.items()})
            data = {"cache": features, **data}
        return data
    
    def sample_new_items(self, seed):
        logger.info("Sampling new %s data with seed %d.", self.split, seed)
        self.items = []
        split = self.split
        if split != 'train':
            raise NotImplementedError("Only train split is supported.")
        
        for scene_id in self.scene_index:
            frame_graph = self.scene_info[scene_id]['graph']

            pairs = []
            for ix in self.scene_index[scene_id]:
                inds = [ ix ]
                while len(inds) < self.conf.views:
                    # get other frames within flow threshold
                    k = (frame_graph[ix][1] > self.conf.fmin) & (frame_graph[ix][1] < self.conf.fmax)
                    frames = frame_graph[ix][0][k]

                    # prefer frames forward in time
                    if np.count_nonzero(frames[frames > ix]):
                        # ix = np.random.RandomState(seed + ix).choice(frames[frames > ix])
                        ix = np.random.choice(frames[frames > ix])

                    elif ix + 1 < len(frame_graph):
                        ix = ix + 1

                    elif np.count_nonzero(frames):
                        # ix = np.random.RandomState(seed + ix).choice(frames)
                        ix = np.random.choice(frames)
                    
                    inds += [ ix ]
                pairs += [(scene_id, *inds)]
            
            self.items += pairs

        np.random.RandomState(seed).shuffle(self.items)

    def __getitem__(self, idx):
        if self.conf.reseed:
            with fork_rng(self.conf.seed + idx, False):
                return self.getitem(idx)
        else:
            return self.getitem(idx)
                    
    def getitem(self, idx):
        if isinstance(idx, list):
            scene, idx0, idx1 = idx
        else:
            scene, idx0, idx1 = self.items[idx]

        data0 = self._read_view(scene, idx0)
        data1 = self._read_view(scene, idx1)
        data = {
            "view0": data0,
            "view1": data1,
        }
        data["T_0to1"] = data1["T_w2cam"] @ data0["T_w2cam"].inv()
        data["T_1to0"] = data0["T_w2cam"] @ data1["T_w2cam"].inv()
        data["name"] = f"{scene}/{data0['name']}_{data1['name']}"

        data["scene"] = scene
        data["idx"] = idx
        return data


    
    def __len__(self):
        return len(self.items)