# `pi-ai-camera` module

This [module](https://docs.viam.com/registry/modular-resources/) implements the [`rdk:service:vision` API](https://docs.viam.com/appendix/apis/services/vision/) for the [Raspberry Pi AI Camera](https://www.raspberrypi.com/products/ai-camera/).
With this model, you can...

## Requirements

TBD

```bash

```

## Configure your pi-ai-camera vision service

Navigate to the [**CONFIGURE** tab](https://docs.viam.com/configure/) of your [machine](https://docs.viam.com/fleet/machines/) in the [Viam app](https://app.viam.com/).
[Add `hipsterbrown:pi-ai-camera` to your machine](https://docs.viam.com/configure/#services).

On the new service panel, copy and paste the following attribute template into your pi-ai-camera’s attributes field:

```json
{
  <INSERT SAMPLE ATTRIBUTES>
}
```

### Attributes

The following attributes are available for `hipsterbrown:vision:pi-ai-camera` vision service:

| Name    | Type   | Required?    | Description |
| ------- | ------ | ------------ | ----------- |
| `todo1` | string | **Required** | TODO        |
| `todo2` | string | Optional     | TODO        |

### Example configuration

```json
{
  <INSERT SAMPLE CONFIGURATION(S)>
}
```

### Next steps

_Add any additional information you want readers to know and direct them towards what to do next with this module._
_For example:_

- To test your...
- To write code against your...

## Troubleshooting

_Add troubleshooting notes here._
