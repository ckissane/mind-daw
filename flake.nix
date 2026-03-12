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

        isLinux = pkgs.stdenv.hostPlatform.isLinux;
        isDarwin = pkgs.stdenv.hostPlatform.isDarwin;

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };

        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

        # System dependencies needed at build time
        buildInputs = with pkgs; [
          # TLS
          openssl

          # Compression
          zstd

          # LSL (system library, bypasses lsl-sys cmake build)
          liblsl
        ]
        ++ pkgs.lib.optionals isLinux (with pkgs; [
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

          # Audio (ALSA)
          alsa-lib
          alsa-plugins

          # D-Bus (required by bluer for BlueZ Bluetooth)
          dbus
        ])
        ++ pkgs.lib.optionals isDarwin (with pkgs; [
          # Frameworks are provided automatically by the Darwin stdenv SDK.
          # libiconv is needed for some Rust crates on macOS.
          libiconv
        ]);

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
          # Patched lsl-sys uses system liblsl framework on macOS
          LIBLSL_FRAMEWORK_PATH = "${pkgs.liblsl}/Frameworks";
        };

        # Build dependencies first (for caching)
        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        # Full package build
        mind-daw = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
        });

        # Runtime library path (Linux only — macOS uses @rpath)
        runtimeLibPath = pkgs.lib.optionalString isLinux (
          pkgs.lib.makeLibraryPath (with pkgs; [
            vulkan-loader
            wayland
            libxkbcommon
            mesa
            liblsl
            dbus
            alsa-lib
          ])
        );

      in {
        packages.default = mind-daw;

        checks = {
          inherit mind-daw;
          clippy = craneLib.cargoClippy (commonArgs // {
            inherit cargoArtifacts;
            cargoClippyExtraArgs = "--all-targets -- --deny warnings";
          });
        };

        devShells.default = pkgs.mkShell ({
          packages = with pkgs; [
            rustToolchain
            rust-analyzer
          ] ++ buildInputs ++ nativeBuildInputs;

          ZSTD_SYS_USE_PKG_CONFIG = "1";
          # Patched lsl-sys uses the system liblsl framework on macOS
          LIBLSL_FRAMEWORK_PATH = "${pkgs.liblsl}/Frameworks";
        } // pkgs.lib.optionalAttrs isDarwin {
          # The nixpkgs liblsl framework has its install name pointing to 'lib/lsl.framework'
          # but the actual bundle is at 'Frameworks/lsl.framework'. DYLD_FRAMEWORK_PATH
          # makes dyld search the Frameworks/ dir by framework name, bypassing the stale path.
          DYLD_FRAMEWORK_PATH = "${pkgs.liblsl}/Frameworks";
          shellHook = ''
            XCODE_PATH="$(xcode-select -p 2>/dev/null || true)"
            if [ -n "$XCODE_PATH" ]; then
              export PATH="$XCODE_PATH/usr/bin:$PATH"
            fi
          '';
        } // pkgs.lib.optionalAttrs isLinux {
          LD_LIBRARY_PATH = runtimeLibPath;
          ALSA_PLUGIN_DIR = "${pkgs.alsa-plugins}/lib/alsa-lib";
          shellHook = ''
            export ALSA_PLUGIN_DIR="${pkgs.alsa-plugins}/lib/alsa-lib"
          '';
        });
      }
    );
}
