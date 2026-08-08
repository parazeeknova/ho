{ pkgs, lib, config, inputs, ... }:

{
  packages = [
    pkgs.git
    pkgs.uv
    pkgs.python314
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
    # LaTeX compiler for the JD-tailored resume (resume.tex -> tailored PDF).
    pkgs.tectonic
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

  env.LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib";
  env.SSL_CERT_FILE = "/etc/ssl/certs/ca-bundle.crt";
  env.REQUESTS_CA_BUNDLE = "/etc/ssl/certs/ca-bundle.crt";

  enterShell = ''
    # Guard against a stale/non-devenv VIRTUAL_ENV (e.g. a leftover .venv) so
    # `uv run` never warns about an environment-path mismatch or targets the
    # wrong interpreter. uv is configured via UV_PROJECT_ENVIRONMENT to use
    # the devenv venv; any other VIRTUAL_ENV is ignored by unsetting it.
    if [ -n "$VIRTUAL_ENV" ] && [ "$VIRTUAL_ENV" != "$UV_PROJECT_ENVIRONMENT" ]; then
      unset VIRTUAL_ENV
    fi
    echo "ho dev env — python $(python --version) | node $(node --version) | uv ready"
  '';
}
