{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    crane.url = "github:ipetkov/crane";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, crane, flake-utils, rust-overlay, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };

        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

        # System dependencies needed at build time
        buildInputs = with pkgs; [
          # Vulkan / GPU
          vulkan-loader
          vulkan-headers

          # Wayland
          wayland
          wayland-protocols
          libxkbcommon

          # X11
          libx11
          libxcursor
          libxrandr
          libxi
          libxcb

          # Mesa (EGL/GL)
          mesa

          # Font / text
          fontconfig
          freetype

          # Audio
          alsa-lib

          # TLS
          openssl

          # Compression
          zstd

          # LSL (system library, bypasses lsl-sys cmake build)
          liblsl

          # D-Bus (required by bluer for BlueZ Bluetooth)
          dbus
        ];

        # Tools needed at build time
        nativeBuildInputs = with pkgs; [
          pkg-config
          cmake
          clang
          perl
          rustPlatform.bindgenHook
        ];

        # Common env vars for the build
        commonArgs = {
          src = craneLib.cleanCargoSource ./.;
          strictDeps = true;
          inherit buildInputs nativeBuildInputs;

          ZSTD_SYS_USE_PKG_CONFIG = "1";
        };

        # Build dependencies first (for caching)
        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        # Full package build
        mind-daw = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
        });

        # Runtime library path for Vulkan/Wayland/etc
        runtimeLibPath = pkgs.lib.makeLibraryPath (with pkgs; [
          vulkan-loader
          wayland
          libxkbcommon
          mesa
          liblsl
          dbus
        ]);

      in {
        packages.default = mind-daw;

        checks = {
          inherit mind-daw;
          clippy = craneLib.cargoClippy (commonArgs // {
            inherit cargoArtifacts;
            cargoClippyExtraArgs = "--all-targets -- --deny warnings";
          });
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            rustToolchain
            rust-analyzer
          ] ++ buildInputs ++ nativeBuildInputs;

          LD_LIBRARY_PATH = runtimeLibPath;
          ZSTD_SYS_USE_PKG_CONFIG = "1";
        };
      }
    );
}
