# spectrum
A music visualizer for my living room.

## Extra installation instructions
Gotta install a very specific version of this library.
```
cd ..
git clone https://github.com/gfx-rs/gfx.git
git checkout 3641183231f16877d4ea2fbdb2ff208ce736d6c4
```

## Running the visualizer
Cargo isn't super good at specifying features in workspaces which is why you do it this way.
```
cd spectrum-viz
cargo run --features metal
```
