use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::{env, fs};

// Code below builds `LD_LIBRARY_PATH` env variable for VSC debugging.
// The output file is used by `launch.json`.
fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join(".env");

    let mut f = File::create(dest_path).unwrap();

    let paths = glob::glob("../target/debug/build/torch-sys-*/out/libtorch/libtorch/lib").unwrap();

    let mut ld_library_path = String::new();

    for path in paths {
        let abs_path = fs::canonicalize(path.unwrap()).unwrap();
        ld_library_path.push_str(&abs_path.to_string_lossy());
        ld_library_path.push(':');
    }

    if ld_library_path.ends_with(':') {
        ld_library_path.pop();
    }

    f.write_all(b"LD_LIBRARY_PATH=").unwrap();
    f.write_all(ld_library_path.as_bytes()).unwrap();
    f.write_all(b"\n").unwrap();
}
