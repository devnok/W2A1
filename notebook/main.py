# import pakcages

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation

# constants

ASSET_PATH = "../assets/"

file_name = {
    "iod": "iod.nc",
    "sst": "sst.anom.mon.mean.nc",
    "precip": "precip.anom.mon.mean.nc",
    "zwind": "uwnd.10m.anom.mon.mean.nc",
    "mwind": "vwnd.10m.anom.mon.mean.nc",
}

file_path = {}

for key, value in file_name.items():
    file_path[key] = ASSET_PATH + value

# load data from netCDF file

iod_data = nc.Dataset(file_path["iod"])
sst_data = nc.Dataset(file_path["sst"])
precip_data = nc.Dataset(file_path["precip"])


sst = sst_data.variables["sst"][:]
lon = sst_data["lon"][:]
lat = sst_data["lat"][:]

data_crs = ccrs.PlateCarree()  # 데이터가 위경도 좌표를 기준으로 정의되어 있으므로, PlateCarree를 사용한다.
projection = ccrs.Mollweide()

fig = plt.figure(figsize=(16, 9))
ax = plt.axes(projection=projection)
ax.set_global()


def draw(i):
    cont = ax.contourf(
        lon, lat, sst[i], levels=20, transform=data_crs, cmap="RdBu_r", vmin=-5, vmax=5
    )
    ax.coastlines()

    if i == 0:
        fig.colorbar(cont)

    return cont


def init():
    # 해안선 추가
    ax.coastlines()
    # 국경 추가
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    # return (ax,)


def update(i):
    ax.clear()
    draw(i)
    # return (draw(i),)


ani = FuncAnimation(fig, update, frames=np.arange(0, 499), init_func=init)

# plt.show()
ani.save("animation.mp4", writer="ffmpeg", fps=30)
