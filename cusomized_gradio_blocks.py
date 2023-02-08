from __future__ import annotations

import ast
import csv
import inspect
import os
import subprocess
import tempfile
import threading
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import PIL.Image

import gradio
from gradio import components, processing_utils, routes, utils
from gradio.context import Context
from gradio.documentation import document, set_documentation_group
from gradio.flagging import CSVLogger

if TYPE_CHECKING:  # Only import for type checking (to avoid circular imports).
    from gradio.components import IOComponent

CACHED_FOLDER = "gradio_cached_examples"
LOG_FILE = "log.csv"

def create_myexamples(
    examples: List[Any] | List[List[Any]] | str,
    inputs: IOComponent | List[IOComponent],
    outputs: IOComponent | List[IOComponent] | None = None,
    fn: Callable | None = None,
    cache_examples: bool = False,
    examples_per_page: int = 10,
    _api_mode: bool = False,
    label: str | None = None,
    elem_id: str | None = None,
    run_on_click: bool = False,
    preprocess: bool = True,
    postprocess: bool = True,
    batch: bool = False,):
    """Top-level synchronous function that creates Examples. Provided for backwards compatibility, i.e. so that gr.Examples(...) can be used to create the Examples component."""
    examples_obj = MyExamples(
        examples=examples,
        inputs=inputs,
        outputs=outputs,
        fn=fn,
        cache_examples=cache_examples,
        examples_per_page=examples_per_page,
        _api_mode=_api_mode,
        label=label,
        elem_id=elem_id,
        run_on_click=run_on_click,
        preprocess=preprocess,
        postprocess=postprocess,
        batch=batch,
        _initiated_directly=False,
    )
    utils.synchronize_async(examples_obj.create)
    return examples_obj

class MyExamples(gradio.helpers.Examples):
    def __init__(
            self,
            examples: List[Any] | List[List[Any]] | str,
            inputs: IOComponent | List[IOComponent],
            outputs: IOComponent | List[IOComponent] | None = None,
            fn: Callable | None = None,
            cache_examples: bool = False,
            examples_per_page: int = 10,
            _api_mode: bool = False,
            label: str | None = "Examples",
            elem_id: str | None = None,
            run_on_click: bool = False,
            preprocess: bool = True,
            postprocess: bool = True,
            batch: bool = False,
            _initiated_directly: bool = True,):

        if _initiated_directly:
            warnings.warn(
                "Please use gr.Examples(...) instead of gr.examples.Examples(...) to create the Examples.",
            )

        if cache_examples and (fn is None or outputs is None):
            raise ValueError("If caching examples, `fn` and `outputs` must be provided")

        if not isinstance(inputs, list):
            inputs = [inputs]
        if outputs and not isinstance(outputs, list):
            outputs = [outputs]

        working_directory = Path().absolute()

        if examples is None:
            raise ValueError("The parameter `examples` cannot be None")
        elif isinstance(examples, list) and (
            len(examples) == 0 or isinstance(examples[0], list)
        ):
            pass
        elif (
            isinstance(examples, list) and len(inputs) == 1
        ):  # If there is only one input component, examples can be provided as a regular list instead of a list of lists
            examples = [[e] for e in examples]
        elif isinstance(examples, str):
            if not Path(examples).exists():
                raise FileNotFoundError(
                    "Could not find examples directory: " + examples
                )
            working_directory = examples
            if not (Path(examples) / LOG_FILE).exists():
                if len(inputs) == 1:
                    examples = [[e] for e in os.listdir(examples)]
                else:
                    raise FileNotFoundError(
                        "Could not find log file (required for multiple inputs): "
                        + LOG_FILE
                    )
            else:
                with open(Path(examples) / LOG_FILE) as logs:
                    examples = list(csv.reader(logs))
                    examples = [
                        examples[i][: len(inputs)] for i in range(1, len(examples))
                    ]  # remove header and unnecessary columns

        else:
            raise ValueError(
                "The parameter `examples` must either be a string directory or a list"
                "(if there is only 1 input component) or (more generally), a nested "
                "list, where each sublist represents a set of inputs."
            )

        input_has_examples = [False] * len(inputs)
        for example in examples:
            for idx, example_for_input in enumerate(example):
                # if not (example_for_input is None):
                if True:
                    try:
                        input_has_examples[idx] = True
                    except IndexError:
                        pass  # If there are more example components than inputs, ignore. This can sometimes be intentional (e.g. loading from a log file where outputs and timestamps are also logged)

        inputs_with_examples = [
            inp for (inp, keep) in zip(inputs, input_has_examples) if keep
        ]
        non_none_examples = [
            [ex for (ex, keep) in zip(example, input_has_examples) if keep]
            for example in examples
        ]

        self.examples = examples
        self.non_none_examples = non_none_examples
        self.inputs = inputs
        self.inputs_with_examples = inputs_with_examples
        self.outputs = outputs
        self.fn = fn
        self.cache_examples = cache_examples
        self._api_mode = _api_mode
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.batch = batch

        with utils.set_directory(working_directory):
            self.processed_examples = [
                [
                    component.postprocess(sample)
                    for component, sample in zip(inputs, example)
                ]
                for example in examples
            ]
        self.non_none_processed_examples = [
            [ex for (ex, keep) in zip(example, input_has_examples) if keep]
            for example in self.processed_examples
        ]
        if cache_examples:
            for example in self.examples:
                if len([ex for ex in example if ex is not None]) != len(self.inputs):
                    warnings.warn(
                        "Examples are being cached but not all input components have "
                        "example values. This may result in an exception being thrown by "
                        "your function. If you do get an error while caching examples, make "
                        "sure all of your inputs have example values for all of your examples "
                        "or you provide default values for those particular parameters in your function."
                    )
                    break

        with utils.set_directory(working_directory):
            self.dataset = components.Dataset(
                components=inputs_with_examples,
                samples=non_none_examples,
                type="index",
                label=label,
                samples_per_page=examples_per_page,
                elem_id=elem_id,
            )

        self.cached_folder = Path(CACHED_FOLDER) / str(self.dataset._id)
        self.cached_file = Path(self.cached_folder) / "log.csv"
        self.cache_examples = cache_examples
        self.run_on_click = run_on_click

from gradio import utils, processing_utils
from PIL import Image as _Image
from pathlib import Path
import numpy as np

def customized_postprocess(self, y):
    if y is None:
        return None

    if isinstance(y, dict):
        if self.tool == "sketch" and self.source in ["upload", "webcam"]:
            y, mask = y["image"], y["mask"]
            if y is None:
                return None
            elif isinstance(y, np.ndarray):
                im = processing_utils.encode_array_to_base64(y)
            elif isinstance(y, _Image.Image):
                im = processing_utils.encode_pil_to_base64(y)
            elif isinstance(y, (str, Path)):
                im = processing_utils.encode_url_or_file_to_base64(y)
            else:
                raise ValueError("Cannot process this value as an Image")
            im = self._format_image(im)

            if mask is None:
                return im
            elif isinstance(y, np.ndarray):
                mask_im = processing_utils.encode_array_to_base64(mask)
            elif isinstance(y, _Image.Image):
                mask_im = processing_utils.encode_pil_to_base64(mask)
            elif isinstance(y, (str, Path)):
                mask_im = processing_utils.encode_url_or_file_to_base64(mask)
            else:
                raise ValueError("Cannot process this value as an Image")

            return {"image": im, "mask" : mask_im,}

    elif isinstance(y, np.ndarray):
        return processing_utils.encode_array_to_base64(y)
    elif isinstance(y, _Image.Image):
        return processing_utils.encode_pil_to_base64(y)
    elif isinstance(y, (str, Path)):
        return processing_utils.encode_url_or_file_to_base64(y)
    else:
        raise ValueError("Cannot process this value as an Image")

# def customized_as_example(self, input_data=None):
#     if input_data is None:
#         return str('assets/demo/misc/noimage.jpg')
#     elif isinstance(input_data, dict):
#         im = np.array(PIL.Image.open(input_data["image"])).astype(float)
#         mask = np.array(PIL.Image.open(input_data["mask"])).astype(float)/255
#         imm = (im * (1-mask)).astype(np.uint8)
#         import time
#         ctime = int(time.time()*100)
#         impath = 'assets/demo/temp/temp_{}.png'.format(ctime)
#         PIL.Image.fromarray(imm).save(impath)
#         return str(utils.abspath(impath))
#     else:
#         return str(utils.abspath(input_data))

def customized_as_example(self, input_data=None):
    if input_data is None:
        return str('assets/demo/misc/noimage.jpg')
    else:
        return str(utils.abspath(input_data))
