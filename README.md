# `pi-ai-camera` module

This [module](https://docs.viam.com/registry/modular-resources/) implements the [`rdk:service:vision` API](https://docs.viam.com/appendix/apis/services/vision/) for the [Raspberry Pi AI Camera](https://www.raspberrypi.com/products/ai-camera/).

With this model, you can perform real-time image classification and object detection while using minimum power and CPU resources compared to inferencing directly on the CPU.

## Requirements

This module assumes you are using the [AI Camera](https://www.raspberrypi.com/documentation/accessories/ai-camera.html#about) attached to one of the following boards:
- Raspberry Pi Zero 2 W
- Raspberry Pi 3 Model B+
- Raspberry Pi 4 Model B
- Raspberry Pi 5 

It also assumes you are using at least version Bookworm of the Raspberry Pi OS.

## Configure your pi-ai-camera vision service

Navigate to the [**CONFIGURE** tab](https://docs.viam.com/configure/) of your [machine](https://docs.viam.com/fleet/machines/) in the [Viam app](https://app.viam.com/).
[Add `hipsterbrown:pi-ai-camera` to your machine](https://docs.viam.com/configure/#services).

### Attributes

The following attributes are available for `hipsterbrown:vision:pi-ai-camera` vision service:

| Name    | Type   | Required?    | Default | Description |
| ------- | ------ | ------------ | ------- | ----------- |
| `task` | "classification" or "object detection" | Optional | "classification"  | Which computer vision task is supported by the selected model |
| `model` | string * | Optional     | 'mobilenet_v2' (classification) or 'ssd_mobilenetv2_fpnlite_320x320_pp' (object detection) ** | Which pre-compiled ML model to use with the selected task |
| `labels_path` | string | Optional | 'assets/imagenet_labels.txt' (classification) or 'assets/coco_labels.txt' (object detection) ** | Path to plain text file with the list of associated image labels for the model |
| `postprocess` | "softmax" or "nanodet" | Optional | - | What kind of post processing step should be taken on the ML output from the camera: [softmax](https://en.wikipedia.org/wiki/Softmax_function) is used with classification tasks |
| `inference_rate` | number | Optional | 30 | The number of frames per second to process images against the configured ML model |
| `default_minimum_confidence` | number | Optional | 0.55 | Number between 0 and 1 as a minimum percentage of confidence for the returned outputs from the model |
| `buffer_count` | number | Optional | 4 | The amount of queued images in the camera to prevent blocking processing new images. |

_* This value must be one of the included pre-compiled models, see more below._

_** The default value depends on the configured task._

**Model Selection:**

While this module tries to select the best general model for each task, feel free to experiment with the various included models to see if another works better for your use case.

The following models can be used for image classification:

- efficientnet_bo
- efficientnet_lite0
- efficientnetv2_b0
- efficientnetv2_b1
- efficientnetv2_b2
- higherhrnet_coco
- levit_128s
- mnasnet1.0
- mobilenet_v2
- mobilevit_xs
- mobilevit_xxs
- regnetx_002
- regnety_002
- regnety_004
- resnet18
- shufflenet_v2_x1_5
- squeezenet1.0

The following models can be used for object detection:

- efficientdet_lite0_pp
- nanodet_plus_416x416_pp
- nanodet_plus_416x416
- ssd_mobilenetv2_fpnlite_320x320_pp


### Example configuration

```json
{
    "task": "object detection",
    "model": "nanodet_plus_416x416_pp"
}
```

### Next steps

Compile other ML models you've trained or found from a model zoo like [HuggingFace](https://huggingface.co/models?pipeline_tag=image-classification&sort=trending). https://www.raspberrypi.com/documentation/accessories/ai-camera.html#model-deployment

_Guide coming soon._

## Troubleshooting

The board may require a reboot after the initial setup of this module due to it installing the `imx500-all` Linux package for the camera drivers and included models.

After configuring (or reconfiguring) the module, it may take a minute for changes to take effect while the neural network firmware is uploaded to the camera. See the [system architecture diagram](https://www.raspberrypi.com/documentation/accessories/ai-camera.html#system-architecture) for more info.
