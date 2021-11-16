import nvidia.dali.fn as fn
import nvidia.dali.math as math
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy


class BasicPipeline(Pipeline):
    def __init__(self, flags, seed, batch_size, input_shape, device_id=0):
        super().__init__(batch_size=batch_size, num_threads=flags.num_workers, device_id=device_id, seed=seed)
        self.flags = flags
        self.internal_seed = seed
        self.input_shape = flags.input_shape
        self.crop_shape = types.Constant(input_shape, dtype=types.INT64)
        self.layout = flags.layout.replace("N", "")
        self.axis_names = "DHW"
        self.bool = ops.Cast(dtype=types.DALIDataType.BOOL)
        # self.reshape = ops.Reshape(device="cpu", layout=flags.layout.replace("N", ""), bytes_per_sample_hint=313997472)
        self.reshape = ops.Reshape(device="cpu", layout="CDHW")
        self.transpose = ops.Transpose(device="gpu", perm=[1, 2, 3, 0])

        self.crop = ops.Crop(device="cpu", crop=input_shape, seed=seed, out_of_bounds_policy="pad")
        self.pos_x = ops.Uniform(range=(0, 1), seed=seed)
        self.pos_y = ops.Uniform(range=(0, 1), seed=seed)
        self.pos_z = ops.Uniform(range=(0, 1), seed=seed)

    @staticmethod
    def random_augmentation(probability, augmented, original):
        condition = fn.cast(fn.coin_flip(probability=probability), dtype=types.DALIDataType.BOOL)
        neg_condition = condition ^ True
        return condition * augmented + neg_condition * original

    def crop_fn(self, img, label):
        kwargs = {"crop_pos_x": self.pos_x(), "crop_pos_y": self.pos_y(), "crop_pos_z": self.pos_z()}
        img, label = self.crop(img, **kwargs), self.crop(label, **kwargs)
        return img, label

    def reshape_fn(self, img, label):
        img = self.reshape(img)
        label = self.reshape(label)
        return img, label

    def random_flips_fn(self, img, label):
        hflip, vflip, dflip = [fn.coin_flip(probability=0.33) for _ in range(3)]
        flips = {"horizontal": hflip, "vertical": vflip, "depthwise": dflip, 'bytes_per_sample_hint': 8388608}
        return fn.flip(img, **flips), fn.flip(label, **flips)

    def zoom_fn(self, img, lbl):
        resized_shape = self.crop_shape * self.random_augmentation(0.1, fn.uniform(range=(1.0, 1.2)), 1.0)
        img = fn.resize(img, interp_type=types.DALIInterpType.INTERP_CUBIC, size=resized_shape)
        lbl = fn.resize(lbl, interp_type=types.DALIInterpType.INTERP_NN, size=resized_shape)
        img = fn.crop(img, crop=self.flags.input_shape)
        lbl = fn.crop(lbl, crop=self.flags.input_shape)
        return img, lbl

    def gaussian_noise_fn(self, img):
        img_noised = img + fn.normal_distribution(img, stddev=0.1)
        return self.random_augmentation(0.1, img_noised, img)

    def brightness_fn(self, img):
        brightness_scale = self.random_augmentation(0.1, fn.uniform(range=(0.7, 1.3)), 1.0)
        return img * brightness_scale

    def gaussian_blur_fn(self, img):
        img_blured = fn.gaussian_blur(img, sigma=fn.uniform(range=(0.25, 1.5)))
        return self.random_augmentation(0.1, img_blured, img)

    @staticmethod
    def slice_fn(img, start_idx, length):
        return fn.slice(img, start_idx, length, axes=[0], out_of_bounds_policy="pad")

    def biased_crop_fn(self, img, label):
        roi_start, roi_end = fn.segmentation.random_object_bbox(label,
                                                                format='start_end',
                                                                foreground_prob=self.flags.oversampling,
                                                                classes=[1, 2],
                                                                background=0,
                                                                k_largest=2,
                                                                seed=self.internal_seed,
                                                                device='cpu',
                                                                cache_objects=True)

        anchor = fn.roi_random_crop(label, roi_start=roi_start, roi_end=roi_end, crop_shape=[1, *self.input_shape])
        anchor = fn.slice(anchor, 1, 3, axes=[0])  # drop channel from anchor
        slice_device = 'cpu'
        img, label = fn.slice([img, label], anchor, self.crop_shape, axis_names=self.axis_names,
                              out_of_bounds_policy="pad", device=slice_device)
        # img, label = fn.slice([img, label], anchor, self.crop_shape, axes=[1, 2, 3],
        #                       out_of_bounds_policy="pad")
        return img, label

    def move_to_gpu(self, img, label):
        return img.gpu(), label.gpu()


class TrainNumpyPipeline(BasicPipeline):
    def __init__(self, flags, seed, batch_size, image_list, label_list, num_shards=1, device_id=0):
        super().__init__(flags=flags, seed=seed, batch_size=batch_size, input_shape=flags.input_shape, device_id=device_id)
        self.oversampling = flags.oversampling
        self.flags = flags
        self.input_x = ops.readers.Numpy(files=image_list,
                                         shard_id=device_id,
                                         num_shards=num_shards,
                                         seed=seed,
                                         stick_to_shard=True,
                                         pad_last_batch=False,
                                         shuffle_after_epoch=False,
                                         prefetch_queue_depth=2,
                                         # read_ahead=True,
                                         bytes_per_sample_hint=313997472)
        self.input_y = ops.readers.Numpy(files=label_list,
                                         shard_id=device_id,
                                         num_shards=num_shards,
                                         seed=seed,
                                         stick_to_shard=True,
                                         pad_last_batch=False,
                                         shuffle_after_epoch=False,
                                         prefetch_queue_depth=2,
                                         # read_ahead=True,
                                         bytes_per_sample_hint=78499368)

    def define_graph(self):
        image = self.input_x(name="ReaderX")
        label = self.input_y(name="ReaderY")

        # Volumetric augmentations
        image, label = self.reshape_fn(image, label)
        image, label = self.biased_crop_fn(image, label)
        # image, label = self.fixed_biased_crop_fn(image, label)
        image, label = self.move_to_gpu(image, label)
        image, label = self.random_flips_fn(image, label)

        # Intensity augmentations
        image = self.brightness_fn(image)
        image = self.gaussian_noise_fn(image)

        return image, label


class ValNumpyPipeline(BasicPipeline):
    def __init__(self, flags, seed, batch_size, image_list, label_list, num_shards=1, device_id=0):
        super().__init__(flags=flags, seed=seed, batch_size=batch_size, input_shape=flags.val_input_shape, device_id=device_id)
        self.input_x = ops.readers.Numpy(files=image_list,
                                         shard_id=0,  # device_id,
                                         num_shards=1,  # num_shards,
                                         seed=seed,
                                         pad_last_batch=False,
                                         random_shuffle=False)
        self.input_y = ops.readers.Numpy(files=label_list,
                                         shard_id=0,  # device_id,
                                         num_shards=1,  # num_shards,
                                         seed=seed,
                                         pad_last_batch=False,
                                         random_shuffle=False)

    def define_graph(self):
        image = self.input_x(name="ReaderX")
        label = self.input_y(name="ReaderY")

        # Volumetric augmentations
        image, label = self.reshape_fn(image, label)
        image, label = self.move_to_gpu(image, label)

        return image, label


class DaliGenericIterator(DALIGenericIterator):
    def __init__(self, pipe: Pipeline, reader_name: str, mode: str = "train"):
        lbp = LastBatchPolicy.FILL if mode == "train" else LastBatchPolicy.PARTIAL
        super().__init__(pipelines=[pipe],
                         reader_name=reader_name,
                         output_map=["image", "label"],
                         auto_reset=True,
                         size=-1,
                         dynamic_shape=mode != "train",
                         last_batch_policy=lbp)

    def __next__(self):
        out = super().__next__()
        out = out[0]
        return out['image'], out['label']


def get_dali_loader(flags,
                    image_list,
                    label_list,
                    mode: str = "train",
                    num_shards: int = 1,
                    device_id: int = 0,
                    seed: int = None) -> DaliGenericIterator:
    assert seed is not None, "Seed is not set"
    if mode == "train":
        pipe = TrainNumpyPipeline(flags,
                                  seed=seed,
                                  batch_size=flags.batch_size,
                                  image_list=image_list,
                                  label_list=label_list,
                                  num_shards=num_shards,
                                  device_id=device_id)
    else:
        pipe = ValNumpyPipeline(flags,
                                seed=seed,
                                batch_size=1,  # For irregular shape it's required
                                image_list=image_list,
                                label_list=label_list,
                                num_shards=num_shards,
                                device_id=device_id)
    pipe.build()
    dali_iter = DaliGenericIterator(pipe, reader_name="ReaderX", mode=mode)
    return dali_iter
