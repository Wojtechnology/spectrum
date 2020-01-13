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
cargo run --features metal
```
