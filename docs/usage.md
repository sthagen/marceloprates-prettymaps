# Usage

## Installation

Install Prettymaps and its dependencies:

```sh
pip install prettymaps
```

## Basic Usage

Generate a map for a location:

```python
import prettymaps

prettymaps.plot("Paris, France")
```

## Customizing Layers and Styles

You can customize which map layers to show and their appearance:

```python
layers = {
    "perimeter": {},
    "streets": {"width": 8},
    "buildings": {},
    "water": {},
}
style = {
    "perimeter": {"fc": "#f2efe9", "ec": "#333"},
    "streets": {"fc": "#cccccc"},
    "buildings": {"fc": "#b0b0b0"},
    "water": {"fc": "#aadaff"},
}
prettymaps.plot("Berlin, Germany", layers=layers, style=style)
```

## Saving Maps

You can save the generated map to a file:

```python
prettymaps.plot("Tokyo, Japan", save_as="tokyo_map.png")
```

## Advanced Features

- **Keypoints:** Highlight specific places or features on the map.
- **Presets:** Save and reuse your favorite map styles.
- **Hillshade:** Add elevation shading for a 3D effect.

See the [API Reference](api.md) for all available options. 