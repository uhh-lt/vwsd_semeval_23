from typing import Tuple
from pathlib import Path
import torchvision.transforms as T
from src.vwsd_dataset import VWSDGoldSample, VWSDTestSample
from PIL import ImageOps
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math


from src.visualsem.visualsem_dataset_nodes import VisualSemNodesDataset
from typing import List


def batch_iter(iterable, bs=1):
    length = len(iterable)
    for ndx in range(0, length, bs):
        yield iterable[ndx : min(ndx + bs, length)]


def plot_vwsd_sample(
    sample: VWSDGoldSample | VWSDTestSample,
    figsize: Tuple[int, int] | None = None,
    save_at: str | Path | None = None,
    fontsize: int | None = None,
    with_title: bool = True,
) -> None:
    f, axarr = plt.subplots(2, 5, figsize=figsize)
    if fontsize is not None:
        plt.rc("font", size=fontsize)
        plt.rc('axes', titlesize=fontsize)
        plt.rc('figure', titlesize=fontsize)
    if with_title:
        f.suptitle(f"target: '{sample.word}' context: '{sample.context}'")
    transform = T.ToPILImage()
    for i, img in enumerate(sample.images):
        img = transform(img)
        if isinstance(sample, VWSDGoldSample) and i == sample.gold_idx:
            img = ImageOps.expand(img, border=50, fill=f"rgb(255, 215, 0)")
        ax = axarr[i // 5, i % 5]  # type: ignore
        ax.imshow(img)
        ax.set_title(f"{i}")
        ax.axis("off")

        # set the spacing between subplots
    plt.subplots_adjust(hspace=-.4)

    if save_at is not None:
        plt.savefig(str(Path(save_at)), bbox_inches="tight")


def plot_vwsd_sample_predictions(sample: VWSDGoldSample | VWSDTestSample, ranks: List[int]) -> None:
    gradient = [
        (255, 215, 0),  # gold
        (24, 228, 2),
        (47, 201, 4),
        (71, 174, 6),
        (94, 147, 8),
        (118, 121, 10),
        (141, 94, 12),
        (165, 67, 14),
        (188, 40, 16),
        (212, 13, 18),
    ]
    f, axarr = plt.subplots(2, 5)
    f.suptitle(str(sample))
    transform = T.ToPILImage()
    for i, img in enumerate(sample.images):
        img = ImageOps.expand(transform(img), border=50, fill=f"rgb{gradient[ranks.index(i)]}")
        ax = axarr[i // 5, i % 5]  # type: ignore
        ax.imshow(img)
        ax.set_title(f"{i}")
        ax.axis("off")


def plot_visual_sem_node(vs: VisualSemNodesDataset, node_idx: int, max_imgs=10) -> None:
    node = vs[node_idx]
    bnid = node["bnid"]
    imgs = vs.get_node_images_by_bnid(bnid)[:max_imgs]

    if len(imgs) > 5:
        num_plts_per_row = math.ceil(len(imgs) / 2)
        f, axarr = plt.subplots(2, num_plts_per_row)
        f.suptitle(f'MainSense: {node["ms"]} -- BabelNetID: {bnid}')
        for i, img in enumerate(imgs):
            ax = axarr[i // num_plts_per_row, i % num_plts_per_row]  # type: ignore
            ax.imshow(mpimg.imread(img))
            ax.set_title(f"{i}")
            ax.axis("off")
    elif len(imgs) == 1:
        f, ax = plt.subplots()
        f.suptitle(f'MainSense: {node["ms"]} -- BabelNetID: {bnid}')
        ax.imshow(mpimg.imread(imgs[0]))
        ax.set_title(f"{0}")
        ax.axis("off")
    else:
        f, axarr = plt.subplots(1, len(imgs))
        f.suptitle(f'MainSense: {node["ms"]} -- BabelNetID: {bnid}')
        for i, ax in enumerate(axarr):
            ax.imshow(mpimg.imread(imgs[i]))
            ax.set_title(f"{i}")
            ax.axis("off")
