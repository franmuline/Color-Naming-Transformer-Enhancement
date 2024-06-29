"""
File added by Francisco Antonio Molina Bakhos.
It implements the metrics deltaE and LPIPS for the evaluation of the models.
"""

from skimage import color
import numpy as np
import torch
import lpips
from basicsr.metrics.metric_util import img_preprocessing
import cv2


def calculate_deltaEab(img1,
                       img2,
                       crop_border,
                       input_order='HWC',
                       test_y_channel=False):
    """ Calculate deltaEab between img1 and img2.

    Args:
        img1 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        img2 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the deltaEab calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: deltaEab result.
    """

    img1, img2 = img_preprocessing(img1, img2, crop_border, input_order, test_y_channel)

    # If range is [0, 255], we normalize to [0, 1]
    if img1.max() > 1:
        img1 = img1 / 255.
        img2 = img2 / 255.

    deltaE = deltaEab()
    return deltaE(img1, img2)


def calculate_deltaE00(img1,
                       img2,
                       crop_border,
                       input_order='HWC',
                       test_y_channel=False):
    """ Calculate deltaE00 between img1 and img2.

    Args:
        img1 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        img2 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the deltaE00 calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: deltaE00 result.
    """

    img1, img2 = img_preprocessing(img1, img2, crop_border, input_order, test_y_channel)

    # If range is [0, 255], we normalize to [0, 1]
    if img1.max() > 1:
        img1 = img1 / 255.
        img2 = img2 / 255.

    deltaE = deltaE00()
    return deltaE(img1, img2)


def calculate_lpips(img1,
                    img2,
                    crop_border,
                    input_order='HWC',
                    test_y_channel=False):
    """ Calculate LPIPS between img1 and img2.

    Args:
        img1 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        img2 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the deltaE00 calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: deltaE00 result.
    """

    img1, img2 = img_preprocessing(img1, img2, crop_border, input_order, test_y_channel)

    lpips = LPIPS()
    return lpips(img1, img2).item()


class deltaEab():
    def __init__(self, color_chart_area=0):
        super().__init__()
        self.color_chart_area = color_chart_area

    def __call__(self, img1, img2):
        """ Compute the deltaE76 between two numpy RGB images
        From M. Afifi: https://github.com/mahmoudnafifi/WB_sRGB/blob/master/WB_sRGB_Python/evaluation/calc_deltaE.py
        :param img1: numpy RGB image or pytorch tensor
        :param img2: numpy RGB image or pytorch tensor
        :return: deltaE76
        """

        if type(img1) == torch.Tensor:
            assert img1.shape[0] == 1
            img1 = img1.squeeze().permute(1, 2, 0).cpu().numpy()

        if type(img2) == torch.Tensor:
            assert img2.shape[0] == 1
            img2 = img2.squeeze().permute(1, 2, 0).cpu().numpy()

        # Convert to Lab
        img1 = color.rgb2lab(img1)
        img2 = color.rgb2lab(img2)

        # reshape to 1D array
        img1 = img1.reshape(-1, 3).astype(np.float32)
        img2 = img2.reshape(-1, 3).astype(np.float32)

        # compute deltaE76
        de76 = np.sqrt(np.sum(np.power(img1 - img2, 2), 1))

        return sum(de76) / (np.shape(de76)[0] - self.color_chart_area)


class deltaE00():
    def __init__(self, color_chart_area=0):
        super().__init__()
        self.color_chart_area = color_chart_area
        self.kl = 1
        self.kc = 1
        self.kh = 1

    def __call__(self, img1, img2):
        """ Compute the deltaE00 between two numpy RGB images
        From M. Afifi: https://github.com/mahmoudnafifi/WB_sRGB/blob/master/WB_sRGB_Python/evaluation/calc_deltaE2000.py
        :param img1: numpy RGB image or pytorch tensor
        :param img2: numpy RGB image or pytorch tensor
        :return: deltaE00
        """

        if type(img1) == torch.Tensor:
            assert img1.shape[0] == 1
            img1 = img1.squeeze().permute(1, 2, 0).cpu().numpy()

        if type(img2) == torch.Tensor:
            assert img2.shape[0] == 1
            img2 = img2.squeeze().permute(1, 2, 0).cpu().numpy()

        # Convert to Lab
        img1 = color.rgb2lab(img1)
        img2 = color.rgb2lab(img2)

        # reshape to 1D array
        img1 = img1.reshape(-1, 3).astype(np.float32)
        img2 = img2.reshape(-1, 3).astype(np.float32)

        # compute deltaE00
        Lstd = np.transpose(img1[:, 0])
        astd = np.transpose(img1[:, 1])
        bstd = np.transpose(img1[:, 2])
        Cabstd = np.sqrt(np.power(astd, 2) + np.power(bstd, 2))
        Lsample = np.transpose(img2[:, 0])
        asample = np.transpose(img2[:, 1])
        bsample = np.transpose(img2[:, 2])
        Cabsample = np.sqrt(np.power(asample, 2) + np.power(bsample, 2))
        Cabarithmean = (Cabstd + Cabsample) / 2
        G = 0.5 * (1 - np.sqrt((np.power(Cabarithmean, 7)) / (np.power(
            Cabarithmean, 7) + np.power(25, 7))))
        apstd = (1 + G) * astd
        apsample = (1 + G) * asample
        Cpsample = np.sqrt(np.power(apsample, 2) + np.power(bsample, 2))
        Cpstd = np.sqrt(np.power(apstd, 2) + np.power(bstd, 2))
        Cpprod = (Cpsample * Cpstd)
        zcidx = np.argwhere(Cpprod == 0)
        hpstd = np.arctan2(bstd, apstd)
        hpstd[np.argwhere((np.abs(apstd) + np.abs(bstd)) == 0)] = 0
        hpsample = np.arctan2(bsample, apsample)
        hpsample = hpsample + 2 * np.pi * (hpsample < 0)
        hpsample[np.argwhere((np.abs(apsample) + np.abs(bsample)) == 0)] = 0
        dL = (Lsample - Lstd)
        dC = (Cpsample - Cpstd)
        dhp = (hpsample - hpstd)
        dhp = dhp - 2 * np.pi * (dhp > np.pi)
        dhp = dhp + 2 * np.pi * (dhp < (-np.pi))
        dhp[zcidx] = 0
        dH = 2 * np.sqrt(Cpprod) * np.sin(dhp / 2)
        Lp = (Lsample + Lstd) / 2
        Cp = (Cpstd + Cpsample) / 2
        hp = (hpstd + hpsample) / 2
        hp = hp - (np.abs(hpstd - hpsample) > np.pi) * np.pi
        hp = hp + (hp < 0) * 2 * np.pi
        hp[zcidx] = hpsample[zcidx] + hpstd[zcidx]
        Lpm502 = np.power((Lp - 50), 2)
        Sl = 1 + 0.015 * Lpm502 / np.sqrt(20 + Lpm502)
        Sc = 1 + 0.045 * Cp
        T = 1 - 0.17 * np.cos(hp - np.pi / 6) + 0.24 * np.cos(2 * hp) + \
            0.32 * np.cos(3 * hp + np.pi / 30) \
            - 0.20 * np.cos(4 * hp - 63 * np.pi / 180)
        Sh = 1 + 0.015 * Cp * T
        delthetarad = (30 * np.pi / 180) * np.exp(
            - np.power((180 / np.pi * hp - 275) / 25, 2))
        Rc = 2 * np.sqrt((np.power(Cp, 7)) / (np.power(Cp, 7) + np.power(25, 7)))
        RT = - np.sin(2 * delthetarad) * Rc
        klSl = self.kl * Sl
        kcSc = self.kc * Sc
        khSh = self.kh * Sh
        de00 = np.sqrt(np.power((dL / klSl), 2) + np.power((dC / kcSc), 2) +
                       np.power((dH / khSh), 2) + RT * (dC / kcSc) * (dH / khSh))

        return np.sum(de00) / (np.shape(de00)[0] - self.color_chart_area)


class LPIPS():
    def __init__(self):
        super().__init__()
        self.lpips = lpips.LPIPS(net='alex', verbose=False)

    def __call__(self, img1, img2):
        """ Compute the LPIPS between two numpy RGB images
        :param img1: numpy RGB image or pytorch tensor
        :param img2: numpy RGB image or pytorch tensor
        :return: LPIPS
        """

        if type(img1) == np.ndarray:
            img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.
            # We normalize the image to [-1, 1] as the LPIPS model was trained with this normalization
            img1 = img1 * 2 - 1

        if type(img2) == np.ndarray:
            img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.
            # We normalize the image to [-1, 1] as the LPIPS model was trained with this normalization
            img2 = img2 * 2 - 1

        return self.lpips(img1, img2)


if __name__ == '__main__':
    # Test the metrics
    img1 = "/home/franmuline/Master_Workspace/TFM/Datasets/FiveK/input/a0001-jmac_DSC1459.png"
    img2 = "/home/franmuline/Master_Workspace/TFM/Datasets/FiveK/input/a0001-jmac_DSC1459.png"

    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    # Convert to (H, W, C) format
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    print(f'deltaEab: {calculate_deltaEab(img1, img2, 0)}')
    print(f'deltaE00: {calculate_deltaE00(img1, img2, 0)}')
    print(f'LPIPS: {calculate_lpips(img1, img2, 0)}')