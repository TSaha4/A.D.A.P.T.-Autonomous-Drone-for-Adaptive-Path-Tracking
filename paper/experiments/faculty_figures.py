"""Generate additional faculty-requested figures from measured results only."""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio

import observed_flood_forecast as forecast

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'paper/experiments/results'
FIGURES = ROOT / 'paper/figures'
CACHE = ROOT / 'paper/experiments/cache/sen1floods11/data/flood_events/HandLabeled'
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 7, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})


def segmentation():
    chips = ('Sri-Lanka_713926', 'Ghana_319168')
    fig, axes = plt.subplots(2, 6, figsize=(7.1, 2.85))
    fields = ('rgb', 'truth', 'otsu', 'lab', 'ndwi', 'unet')
    names = ('S2 RGB', 'Hand label', 'S1 VH Otsu', 'S2 Lab k-means', 'S2 NDWI', 'S1 U-Net')
    for row, chip in enumerate(chips):
        data = np.load(RESULTS / 'sen1floods11' / f'qualitative_{chip}.npz')
        for col, field in enumerate(fields):
            ax = axes[row, col]
            if field == 'rgb':
                ax.imshow(data[field][:, :, ::-1])
            else:
                arr = data[field] == 1 if field == 'truth' else data[field]
                ax.imshow(arr, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
            ax.set(xticks=[], yticks=[])
            if row == 0: ax.set_title(names[col], fontsize=7)
            if col == 0: ax.set_ylabel(chip.replace('_', ' '), fontsize=7)
    fig.tight_layout(pad=0.4, w_pad=0.15, h_pad=0.2)
    fig.savefig(FIGURES / 'fig_segmentation_qual.pdf')
    plt.close(fig)


def flood():
    grid, transform, aois = forecast.setup_grid()
    masks = [forecast.raster(forecast.geometry(forecast.get_geojson(x, 'observedEventA')), grid, transform)
             for x, _ in forecast.PRODUCTS]
    valid_aoi = [forecast.raster(x, grid, transform) for x in aois]
    data = json.loads((RESULTS / 'forecast/emsr504_aoi01.json').read_text())
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.5))
    for row, result in enumerate(data['pairs']):
        valid = valid_aoi[row] & valid_aoi[row + 1]
        current, observed = masks[row], masks[row + 1]
        # The recorded weather leaves the published proxy unchanged in both pairs.
        predicted = current.copy()
        view = np.argwhere(valid)
        y0, x0 = view.min(0); y1, x1 = view.max(0) + 1
        pad = 4
        sl = np.s_[max(0, y0-pad):min(grid[0], y1+pad), max(0, x0-pad):min(grid[1], x1+pad)]
        layers = (current, predicted, observed, predicted ^ observed)
        labels = ('Initial observed', 'Project proxy', 'Next observed', 'Disagreement')
        for col, (layer, label) in enumerate(zip(layers, labels)):
            image = np.ma.masked_where(~valid[sl], layer[sl])
            ax = axes[row, col]
            ax.imshow(image, cmap='Blues' if col < 3 else 'Reds', vmin=0, vmax=1,
                      interpolation='nearest')
            ax.set(xticks=[], yticks=[])
            if row == 0: ax.set_title(label, fontsize=7)
            if col == 0: ax.set_ylabel(f"{result['start']} to\n{result['end']}", fontsize=7)
    fig.tight_layout(pad=0.35, w_pad=0.1, h_pad=0.25)
    fig.savefig(FIGURES / 'fig_forecast_observed.pdf')
    plt.close(fig)


def geography():
    chip = 'Ghana_319168'
    with rasterio.open(CACHE / 'S2Hand' / f'{chip}_S2Hand.tif') as src:
        s2 = src.read()
        bounds = src.bounds
    rgb = np.clip(np.stack([s2[3], s2[2], s2[1]], -1) / 3000 * 255, 0, 255).astype(np.uint8)
    wpl = RESULTS / 'georeference' / f'{chip}_geotiff.waypoints'
    rows = [r.split() for r in wpl.read_text().splitlines()[1:]]
    route = np.array([(float(r[9]), float(r[8])) for r in rows if r[3] == '16'])
    drops = np.array([(float(r[9]), float(r[8])) for r in rows if r[3] == '183'])
    error = json.loads((RESULTS / 'georeference/error_grid.json').read_text())
    one = next(x for x in error['per_chip'] if x['chip'] == chip)
    values = np.array(one['grid_error_m']).reshape(9, 9)
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.65))
    axes[0].imshow(rgb, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top))
    axes[0].plot(route[:, 0], route[:, 1], color='#FFDB58', lw=1.0, label='GeoTIFF WPL')
    axes[0].scatter(drops[:, 0], drops[:, 1], s=8, c='#E74C3C', zorder=3, label='Drops')
    axes[0].set(xlabel='Longitude (WGS84)', ylabel='Latitude (WGS84)', title='Mission on georeferenced S2 chip')
    axes[0].legend(fontsize=6, frameon=False)
    im = axes[1].imshow(values, origin='upper', cmap='magma', extent=(0, 512, 512, 0))
    axes[1].set(xlabel='Image x (px)', ylabel='Image y (px)', title='Scalar-scale coordinate error')
    fig.colorbar(im, ax=axes[1], label='Error (m)', fraction=.046, pad=.04)
    fig.tight_layout(pad=.45)
    fig.savefig(FIGURES / 'fig_georeferenced_mission.pdf')
    plt.close(fig)


def assignment():
    def bases(method):
        path = glob.glob(str(RESULTS / 'fleet_balancing' / method / 'sessions/kanpur_moderate_K4/*/run_summary.json'))[0]
        return json.loads(Path(path).read_text())['bases']
    nearest, balanced = bases('nearest'), bases('balanced')
    comp = json.loads((RESULTS / 'fleet_balancing/comparison.json').read_text())
    ort = json.loads((RESULTS / 'fleet_ortools/comparison.json').read_text())
    def row(method):
        if method == 'OR-Tools':return next(x for x in ort if x['map']=='kanpur' and x['weather']=='moderate')
        return next(x for x in comp if x['map']=='kanpur' and x['weather']=='moderate' and x['method']==method)
    groups = [row('nearest_feasible'), row('balanced_roundtrip_proxy'), row('OR-Tools')]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3), gridspec_kw={'width_ratios':[1.35,1,1]})
    xs = np.arange(4); width=.34
    axes[0].bar(xs-width/2, [b['route_length_m']/1000 for b in nearest], width,
                color='#315F89', label='Nearest')
    axes[0].bar(xs+width/2, [b['route_length_m']/1000 for b in balanced], width,
                color='#9ABAD1', label='Balanced')
    axes[0].set(xticks=xs, xticklabels=[f'UAV {i}' for i in range(1,5)], ylabel='Route per UAV (km)')
    axes[0].legend(frameon=False, fontsize=6)
    labels=['Nearest','Balanced','OR-Tools']; colors=['#315F89','#9ABAD1','#9C4E45']
    axes[1].bar(labels,[r['makespan_cruise_min'] for r in groups],color=colors)
    axes[1].set(ylabel='Longest-route cruise (min)',ylim=(0,33))
    axes[2].bar(labels,[r['visited'] for r in groups],color=colors)
    axes[2].set(ylabel='Visited drops of 41',ylim=(0,41))
    for ax in axes[1:]: ax.tick_params(axis='x',labelrotation=25)
    fig.tight_layout(pad=.5,w_pad=.9)
    fig.savefig(FIGURES / 'fig_workload_vrp.pdf')
    plt.close(fig)


def sensitivity():
    data = json.loads((RESULTS / 'fleet_sensitivity/sensitivity.json').read_text())
    robust = json.loads((RESULTS / 'operator_robustness/summary.json').read_text())['summary']
    fig, axes = plt.subplots(2, 2, figsize=(6.85, 4.7))
    names={'DRONE_LOADED_ENDURANCE_MIN':'Endurance','PAYLOAD_PER_DROP_KG':'Drop payload',
           'WIND_PENALTY_PER_MPS':'Wind penalty'}
    ax=axes[0,0]
    for map_name, color in (('varanasi','#315F89'),('kanpur','#A34840')):
        base=next(r['visited'] for r in data if r['map']==map_name and r['variable']=='scale' and r['meters_per_pixel']==2)
        for idx,(parameter,label) in enumerate(names.items()):
            rows=sorted((r for r in data if r['map']==map_name and r['variable']==parameter),key=lambda r:r['multiplier'])
            ax.plot([.8,1,1.2],[rows[0]['visited'],base,rows[1]['visited']],
                    color=color,ls=('-', '--', ':')[idx], marker='o',ms=2,label=f'{map_name}: {label}')
    ax.set(xlabel='Assumption multiplier',ylabel='Visited drops',xticks=[.8,1,1.2],title='Battery-model sensitivity')
    ax.legend(frameon=False,fontsize=5,ncol=2)
    ax=axes[0,1]
    for map_name,color in (('varanasi','#315F89'),('kanpur','#A34840')):
        rows=[r for r in data if r['map']==map_name and r['variable']=='scale']
        ax.plot([r['meters_per_pixel'] for r in rows],[r['visited'] for r in rows],
                'o-',ms=2,color=color,label=map_name.capitalize())
    ax.set(xscale='log',xlabel='Ground scale (m/px)',ylabel='Visited drops',title='Scale sensitivity')
    ax.legend(frameon=False,fontsize=6)
    ax=axes[1,0]
    for map_name,color in (('varanasi','#315F89'),('kanpur','#A34840')):
        rows=[r for r in data if r['map']==map_name and r['variable']=='scale']
        ax.plot([r['meters_per_pixel'] for r in rows],[r['wall_s'] for r in rows],
                'o-',ms=2,color=color,label=map_name.capitalize())
    ax.set(xscale='log',xlabel='Ground scale (m/px)',ylabel='One-run wall time (s)',title='Measured runtime')
    ax.legend(frameon=False,fontsize=6)
    ax=axes[1,1]
    for map_name,color in (('varanasi','#315F89'),('kanpur','#A34840')):
        rows=[next(r for r in robust if r['map']==map_name and r['weather']==w)
              for w in ('calm','moderate','heavy')]
        xx=np.arange(3)
        ax.errorbar(xx,[r['route_length_px_mean'] for r in rows],
                    yerr=[r['route_length_px_sd'] for r in rows],color=color,marker='o',ms=2,
                    capsize=2,label=map_name.capitalize())
    ax.set(xticks=[0,1,2],xticklabels=['Calm','Moderate','Heavy'],ylabel='Route length (display px)',
           title='Operator and weather variation')
    ax.legend(frameon=False,fontsize=6)
    for ax in axes.flat:ax.grid(alpha=.15)
    fig.tight_layout(pad=.55,w_pad=.75,h_pad=.6)
    fig.savefig(FIGURES / 'fig_sensitivity_suite.pdf')
    plt.close(fig)


def assam():
    arr=np.load(RESULTS/'e2e/assam.npz')
    image, mask=arr['display_image'][:,:,::-1], arr['mask']
    occupancy=float(np.count_nonzero(mask))/mask.size
    fig,axes=plt.subplots(1,3,figsize=(6.85,2.1))
    axes[0].imshow(image);axes[0].set_title('Thematic input')
    axes[1].imshow(mask,cmap='Reds',vmin=0,vmax=255);axes[1].set_title(f'HSV mask: {occupancy:.1%} flood')
    axes[2].bar(['Varanasi','Kanpur','Assam'],[.081,.151,occupancy],color=['#315F89','#315F89','#A34840'])
    axes[2].axhline(.8,color='#A34840',ls='--',label='80% review gate')
    axes[2].set(ylabel='Mask occupancy',ylim=(0,1),title='Proposed input rejection')
    axes[2].legend(frameon=False,fontsize=6)
    for ax in axes[:2]:ax.set(xticks=[],yticks=[])
    fig.tight_layout(pad=.45)
    fig.savefig(FIGURES/'fig_assam_failure.pdf')
    plt.close(fig)


def mavproxy():
    transcript=(RESULTS/'sitl/mavproxy_transcript.txt').read_text().splitlines()
    chosen=[]
    for line in transcript:
        if any(t in line for t in ('Loaded 364 waypoints in','Sent all 364','Got MISSION_ACK: TYPE_MISSION: ACCEPTED',
                                  'Have 364 of 364 waypoints')):
            chosen.append(line.strip())
    fig,ax=plt.subplots(figsize=(6.85,1.25))
    ax.set_facecolor('#15202B');fig.patch.set_facecolor('white')
    ax.text(.02,.93,'MAVProxy ground-station transcript (actual SITL session)',
            transform=ax.transAxes,va='top',color='#DDEEFF',fontfamily='monospace',fontsize=7)
    for i,line in enumerate(chosen[:5]):
        ax.text(.02,.77-i*.14,line[:105],transform=ax.transAxes,va='top',color='#A7D6A6',
                fontfamily='monospace',fontsize=6.3)
    ax.set(xticks=[],yticks=[])
    for spine in ax.spines.values():spine.set_visible(False)
    fig.tight_layout(pad=.25)
    fig.savefig(FIGURES/'fig_mavproxy.pdf')
    plt.close(fig)


if __name__=='__main__':
    FIGURES.mkdir(parents=True,exist_ok=True)
    segmentation();flood();geography();assignment();sensitivity();assam();mavproxy()
