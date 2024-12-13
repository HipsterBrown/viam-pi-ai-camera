import asyncio
import time
from functools import lru_cache
from typing import ClassVar, List, Mapping, Optional, Sequence

from typing_extensions import Self
from viam.logging import getLogger
from viam.media.video import ViamImage, CameraMimeType
from viam.media.utils.pil import pil_to_viam_image
from viam.module.module import Module
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.vision import Vision, CaptureAllResult
from viam.utils import ValueTypes, struct_to_dict

from picamera2 import CompletedRequest, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection
from picamera2.devices.imx500.imx500 import FW_NETWORK_STAGE
from picamera2.devices.imx500.postprocess import softmax, scale_boxes

import numpy as np

LOGGER = getLogger(__name__)


class PiAiCamera(Vision, EasyResource):
    MODEL: ClassVar[Model] = Model(
        ModelFamily("hipsterbrown", "vision"), "pi-ai-camera"
    )

    intrinsics: NetworkIntrinsics
    picam: Picamera2 = None

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """This method creates a new instance of this Vision service.
        The default implementation sets the name from the `config` parameter and then calls `reconfigure`.

        Args:
            config (ComponentConfig): The configuration for this resource
            dependencies (Mapping[ResourceName, ResourceBase]): The dependencies (both implicit and explicit)

        Returns:
            Self: The resource
        """
        return super().new(config, dependencies)

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Sequence[str]:
        """This method allows you to validate the configuration object received from the machine,
        as well as to return any implicit dependencies based on that `config`.

        Args:
            config (ComponentConfig): The configuration for this resource

        Returns:
            Sequence[str]: A list of implicit dependencies
        """
        return []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """This method allows you to dynamically update your service when it receives a new `config` object.

        Args:
            config (ComponentConfig): The new configuration
            dependencies (Mapping[ResourceName, ResourceBase]): Any dependencies (both implicit and explicit)
        """
        attrs = struct_to_dict(config.attributes)
        task = str(attrs.get("task", "classification"))
        model = attrs.get(
            "model_path",
            f"/usr/share/imx500-models/{'imx500_network_mobilenet_v2.rpk' if task == 'classification' else 'imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk'}",
        )
        inference_rate = attrs.get("inference_rate", 30)
        postprocess = attrs.get("postprocess", None)
        self.labels_path = str(
            attrs.get(
                "labels_path",
                f"assets/{'imagenet_labels' if task == 'classification' else 'coco_labels'}.txt",
            )
        )

        if self.picam is not None:
            LOGGER.info("Stopping existing camera")
            self.picam.close()

        LOGGER.info("Creating IMX500 instance")
        self.imx500 = IMX500(model)
        self.intrinsics = self.imx500.network_intrinsics or NetworkIntrinsics()
        self.intrinsics.task = task
        if postprocess == "softmax":
            self.intrinsics.softmax = True
        if postprocess == "nanodet":
            self.intrinsics.postprocess = postprocess
        self.intrinsics.inference_rate = inference_rate

        with open(self.labels_path, "r") as f:
            self.intrinsics.labels = f.read().splitlines()

        self.intrinsics.update_with_defaults()

        LOGGER.info("Creating Picamera2 instance")
        self.picam = Picamera2(self.imx500.camera_num)
        cam_config = self.picam.create_still_configuration(buffer_count=4)
        self.picam.start(cam_config)

        if self.intrinsics.preserve_aspect_ratio:
            LOGGER.info("Preserving aspect ratio")
            self.imx500.set_auto_aspect_ratio()

        LOGGER.info("Waiting on camera firmware upload to complete")
        while True:
            current, total = self.imx500.get_fw_upload_progress(FW_NETWORK_STAGE)
            if total:
                LOGGER.info(f"{current} out of {total} bytes uploaded to camera")
                if current > 0.95 * total:
                    LOGGER.info("Firmware upload complete!")
                    break
            time.sleep(0.5)

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        LOGGER.info(
            f"Processing capture request for {camera_name} with arguments: return_image ({return_image}), return_classifications ({return_classifications}), return_detections ({ return_detections})"
        )
        properties = await self.get_properties()
        result = CaptureAllResult()
        with self.picam.captured_request() as capture_request:
            if return_image:
                result.image = pil_to_viam_image(
                    capture_request.make_image("main"), CameraMimeType.JPEG
                )

            if return_detections and properties.detections_supported:
                result.detections = self._parse_detections_from_request(capture_request)

            if return_classifications and properties.classifications_supported:
                result.classifications = self._parse_classifications_from_request(
                    capture_request, 5
                )

            return result

    def _parse_detections_from_request(
        self, request: CompletedRequest
    ) -> List[Detection]:
        metadata = request.get_metadata()
        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
        input_w, input_h = self.imx500.get_input_size()
        if np_outputs is None:
            return []

        if self.intrinsics.postprocess == "nanodet":
            boxes, scores, classes = postprocess_nanodet_detection(
                outputs=np_outputs[0], max_out=10
            )[0]
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
        else:
            boxes, scores, classes = (
                np_outputs[0][0],
                np_outputs[1][0],
                np_outputs[2][0],
            )
            if self.intrinsics.bbox_normalization:
                boxes = boxes / input_h

            if self.intrinsics.bbox_order == "xy":
                boxes = boxes[:, [1, 0, 3, 2]]

            boxes = np.array_split(boxes, 4, axis=1)
            boxes = zip(*boxes)

        boxes = [
            self.imx500.convert_inference_coords(box, metadata, self.picam)
            for box in boxes
        ]
        return [
            Detection(
                x_min=box[0],
                y_min=box[1],
                x_max=box[2],
                y_max=box[3],
                confidence=score,
                class_name=self._get_label_for_index(int(class_idx)),
            )
            for box, score, class_idx in zip(boxes, scores, classes)
        ]

    def _parse_classifications_from_request(
        self, request: CompletedRequest, count: int = 3
    ) -> List[Classification]:
        np_outputs = self.imx500.get_outputs(request.get_metadata())
        if np_outputs is None:
            return []

        np_output = np_outputs[0]
        if self.intrinsics.softmax:
            np_output = softmax(np_output)

        top_indices = np.argpartition(-np_output, count)[:count]
        top_indices = top_indices[np.argsort(-np_output[top_indices])]
        return [
            Classification(
                class_name=self._get_label_for_index(int(index)),
                confidence=np_output[index],
            )
            for index in top_indices
        ]

    def _get_label_for_index(self, index: int) -> str:
        labels = self._get_labels(self.labels_path)
        return labels[index]

    @lru_cache
    def _get_labels(self, _labels_path: str) -> List[str]:
        """labels_path is used to break the cache if the configuration changes"""
        labels = self.intrinsics.labels

        if self.intrinsics.ignore_dash_labels:
            labels = [label for label in labels if label and label != "-"]

        return labels

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        if self.intrinsics.task == "object detection":
            with self.picam.captured_request() as capture_request:
                detections = self._parse_detections_from_request(capture_request)
                return detections
        raise NotImplementedError()

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        raise NotImplementedError()

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        if self.intrinsics.task == "classification":
            with self.picam.captured_request() as capture_request:
                classifications = self._parse_classifications_from_request(
                    capture_request, count
                )
                return classifications
        raise NotImplementedError()

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        raise NotImplementedError()

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[PointCloudObject]:
        raise NotImplementedError()

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Vision.Properties:
        properties = Vision.Properties()
        properties.object_point_clouds_supported = False
        properties.detections_supported = self.intrinsics.task == "object detection"
        properties.classifications_supported = self.intrinsics.task == "classification"
        return properties

    async def close(self):
        LOGGER.info("Closing camera instance")
        self.picam.close()


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
