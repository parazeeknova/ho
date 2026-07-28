{ pkgs, lib, config, inputs, ... }:

{
  packages = [
    pkgs.git
    pkgs.uv
    pkgs.python314
    pkgs.stdenv.cc.cc.lib
  ];

  languages.python = {
    enable = true;
    package = pkgs.python314;
    uv.enable = true;
    uv.sync.enable = true;
  };

  dotenv.enable = true;

  env.LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";

  enterShell = ''
    echo "ho dev env — python $(python --version) | uv ready"
  '';
}
