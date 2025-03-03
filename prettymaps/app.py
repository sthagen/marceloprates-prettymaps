import streamlit as st
import logging
from matplotlib import pyplot as plt
import subprocess

try:
    import prettymaps
except:
    subprocess.run(["pip", "install", "prettymaps"])
    import prettymaps

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

presets = prettymaps.presets().to_dict()

# Set the title of the app
st.title("prettymaps")

cols = st.columns([1, 2])
with cols[0]:
    query = st.text_area(
        "Location", value="Stad van de Zon, Heerhugowaard, Netherlands", height=86
    )
    radius = st.slider("Radius (km)", 0.5, 20.0, 0.1, step=0.5)
    circular = st.checkbox("Circular map", value=False)

    # Add input for number of colors
    num_colors = st.number_input("Number of colors", min_value=1, value=2, step=1)

    custom_palette = {}
    color_cols = st.columns(4)
    palette = ["#433633", "#FF5E5B"]
    for i in range((num_colors + 3) // 4):  # Calculate the number of rows needed
        for j, col in enumerate(color_cols):
            idx = i * 4 + j
            if idx < num_colors:
                color = col.color_picker(
                    f"Color {idx + 1:02d}", palette[idx % len(palette)]
                )
                custom_palette[idx] = color

    # Add page size options
    page_size = st.selectbox(
        "Page Size", ["A4", "A5", "A3", "A2", "A1", "Custom"], index=0
    )

    # Preset selector
    preset_options = list(presets["preset"].values())
    selected_preset = st.selectbox(
        "Select a Preset", preset_options, index=preset_options.index("default")
    )

    if page_size == "Custom":
        width = st.number_input("Custom Width (inches)", min_value=1.0, value=8.27)
        height = st.number_input("Custom Height (inches)", min_value=1.0, value=11.69)
    else:
        page_sizes = {
            "A4": (8.27, 11.69),
            "A5": (5.83, 8.27),
            "A3": (11.69, 16.54),
            "A2": (16.54, 23.39),
            "A1": (23.39, 33.11),
        }
        width, height = page_sizes[page_size]

    # Layer selection
    st.subheader("Select Layers")

    layers = {
        "hillshade": st.checkbox("Hillshade", value=False),
        "buildings": st.checkbox("Buildings", value=True),
        "streets": st.checkbox("Streets", value=True),
        "waterway": st.checkbox("Waterway", value=True),
        "building": st.checkbox("Building", value=True),
        "water": st.checkbox("Water", value=True),
        "sea": st.checkbox("Sea", value=True),
        "forest": st.checkbox("Forest", value=True),
        "green": st.checkbox("Green", value=True),
        "rock": st.checkbox("Rock", value=True),
        "beach": st.checkbox("Beach", value=True),
        "parking": st.checkbox("Parking", value=True),
    }

    # Hillshade parameters
    if layers["hillshade"]:
        st.subheader("Hillshade Parameters")
        azdeg = st.number_input(
            "Azimuth (degrees)", min_value=0, max_value=360, value=315
        )
        altdeg = st.number_input(
            "Altitude (degrees)", min_value=0, max_value=90, value=45
        )
        vert_exag = st.number_input("Vertical Exaggeration", min_value=0.1, value=1.0)
        dx = st.number_input("dx", min_value=0.1, value=1.0)
        dy = st.number_input("dy", min_value=0.1, value=1.0)
        alpha = st.number_input("Alpha", min_value=0.0, max_value=1.0, value=0.75)

# Add a button in a new column to the right
with cols[1]:
    for i in range(0):
        st.write("")
    button = st.button(
        "Generate",
        key="generate_map",
        help="Click to generate the map",
        type="primary",
        icon=":material/map:",
        use_container_width=True,
    )
    if button:
        hillshade_params = (
            {
                "azdeg": azdeg,
                "altdeg": altdeg,
                "vert_exag": vert_exag,
                "dx": dx,
                "dy": dy,
                "alpha": alpha,
            }
            if layers["hillshade"]
            else {}
        )
        prettymaps.plot(
            query,
            radius=1000 * radius,
            circle=circular,
            layers={k: (False if v == False else {}) for k, v in layers.items()},
            style={"building": {"palette": list(custom_palette.values())}},
            figsize=(width, height),
        )
        st.pyplot(plt)
    else:
        fig_path = "https://github.com/marceloprates/prettymaps/raw/main/prints/heerhugowaard.png"
        st.image(fig_path, use_container_width=True)
