"""The cached 224x224 image must be identical to the v1 preprocessing (main.py lines 113-140)."""
import sys
from pathlib import Path
import numpy as np
import cv2, pydicom
import torchvision.transforms as T
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xfi import data as D


def _dicom(tmp_path, bits):
    rng = np.random.default_rng(bits)
    meta = FileMetaDataset(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"; meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = Dataset(); ds.file_meta = meta; ds.Rows = ds.Columns = 1024; ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"; ds.PixelRepresentation = 0
    ds.BitsAllocated = 8 if bits == 8 else 16; ds.BitsStored = bits; ds.HighBit = bits - 1
    dtype = np.uint8 if bits == 8 else np.uint16
    yy, xx = np.mgrid[0:1024, 0:1024]
    img = (np.sin(xx / 37.0) * np.cos(yy / 53.0) * 0.4 + 0.5) * (2 ** bits - 1) + rng.normal(0, 3, (1024, 1024))
    ds.PixelData = np.clip(img, 0, 2 ** bits - 1).astype(dtype).tobytes()
    fp = tmp_path / f"x{bits}.dcm"; ds.save_as(fp, enforce_file_format=True)
    return fp


def _v1(fp):
    image = pydicom.dcmread(fp).pixel_array
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return T.Compose([T.ToPILImage(), T.Resize((224, 224)), T.ToTensor()])(image).numpy()


def test_cache_equals_v1(tmp_path):
    for bits in (8, 12):
        fp = _dicom(tmp_path, bits)
        g = D.preprocess_dicom(fp)
        ours = D.make_transform(train=False, imagenet_norm=False)(np.repeat(g[:, :, None], 3, axis=2)).numpy()
        assert ours.shape == (3, 224, 224)
        assert np.abs(ours - _v1(fp)).max() == 0.0
