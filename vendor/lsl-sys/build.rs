use std::env;

fn main() {
    let target = env::var("TARGET").unwrap();

    if target.contains("apple") {
        // On macOS, link against the pre-built system liblsl framework.
        // The framework path is set by the nix devShell via LIBLSL_FRAMEWORK_PATH.
        let framework_path = env::var("LIBLSL_FRAMEWORK_PATH")
            .expect("LIBLSL_FRAMEWORK_PATH must be set (e.g. in nix devShell)");
        println!("cargo:rustc-link-search=framework={}", framework_path);
        println!("cargo:rustc-link-lib=framework=lsl");
    } else {
        build_liblsl();
    }
}

fn build_liblsl() {
    let target = env::var("TARGET").unwrap();

    let mut cfg = cmake::Config::new("liblsl");
    cfg.define("LSL_NO_FANCY_LIBNAME", "ON")
        .define("LSL_BUILD_STATIC", "ON");

    if target.contains("msvc") {
        let cxx_args = " /nologo /EHsc /MD /GR";
        cfg.define("WIN32", "1")
            .define("_WINDOWS", "1")
            .define("CMAKE_C_FLAGS", cxx_args)
            .define("CMAKE_CXX_FLAGS", cxx_args)
            .define("CMAKE_C_FLAGS_DEBUG", cxx_args)
            .define("CMAKE_CXX_FLAGS_DEBUG", cxx_args)
            .define("CMAKE_C_FLAGS_RELEASE", cxx_args)
            .define("CMAKE_CXX_FLAGS_RELEASE", cxx_args);
    }

    let install_dir = cfg.build();
    let libdir = install_dir.join("lib");
    let libname = "lsl-static";
    println!("cargo:rustc-link-search=native={}", libdir.to_str().unwrap());
    println!("cargo:rustc-link-lib=static={}", libname);

    if target.contains("linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    } else if target.contains("windows") {
        println!("cargo:rustc-link-lib=dylib=bcrypt");
    } else {
        println!("cargo:rustc-link-lib=dylib=c++");
    }
}
