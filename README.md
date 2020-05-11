# spectrum
A music visualizer for my living room.

## Extra installation instructions
Gotta install a very specific version of this library.
```
cd ..
git clone https://github.com/gfx-rs/gfx.git
git checkout f999b5295ca30dc38f33b0a4770a31d50138687c
```

## Running the visualizer
Cargo isn't super good at specifying features in workspaces which is why you do it this way.
```
cd spectrum-viz
cargo run --release --features metal <filename>
```
`filename` refers to the MP3 file you want to visualize. TODO: Document configuration for the visualizer once that is more polished.

## Tools
TODO: Add the different tools that are available, once they are more polished.

## Science
The `science/` directory allows for exploring different aspects of spectrum, such as spectrograms, beat grids, etc, in Python. I also provides utilities for transforming and visualizing the data. More instructions on how to set that environment up are located there.
