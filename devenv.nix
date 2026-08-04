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

  languages.javascript = {
    enable = true;
    package = pkgs.nodejs_26;
  };

  dotenv.enable = true;

  env.LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";

  enterShell = ''
    echo "ho dev env — python $(python --version) | node $(node --version) | uv ready"
  '';
}
