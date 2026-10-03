import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    from gdrive_fsspec import GoogleDriveFileSystem

    fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True, auth_kwargs={"use_local_webserver": False})
    mo.output.clear_console()
    return (fs,)


@app.cell
def _():
    import os

    return


@app.cell
def _(fs):
    fs.ls("/")
    return


@app.cell
def _():
    import subprocess

    subprocess.run(["ls", "-sla"])
    return (subprocess,)


@app.cell
def _(subprocess):
    subprocess.run(["ls", "-sla", ".."])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
