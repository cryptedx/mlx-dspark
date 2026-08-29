# Repository instructions

## After every implementation

- When an implementation is complete, build and package the Mac app with `./packaging/make_app.sh --debug` from `apps/MacApp/`.
- Start a fresh debug app instance with `MLXDSPARK_ENGINE_SOURCE="$(cd ../.. && pwd)" open -n "build.noindex/mlx-dspark (dev).app"` from `apps/MacApp/`. Restart the running instance first when needed so the current binary is used.
- Verify the finished change in the running app before handing it off; report any build, launch, or verification failure.
