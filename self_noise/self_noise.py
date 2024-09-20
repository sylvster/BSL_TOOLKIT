# Sensor Self-Noise
# Based on Adam Ringler (USGS) code

# Import Modules

import obspy as ob
from obspy.core import UTCDateTime, read, Stream
from obspy.signal.spectral_estimation import get_nlnm, get_nhnm
from obspy import read_inventory

import numpy as np

import scipy as sp
from scipy import signal

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas as pd

import sys
import os
import yaml

print("# obspy version = ",ob.__version__)
print("# numpy version = ",np.__version__)
print("# scipy version = ",sp.__version__)
print("# matplotlib version = ",mpl.__version__)

# Set figure and font sizes

plt.rcParams['figure.figsize'] = 16, 10
plt.rcParams['figure.figsize'] = 16, 9

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Import parameters
file_path = os.path.realpath(__file__)
directory = os.path.dirname(file_path)

with open(f"{directory}/config.yml") as f:
    config = yaml.safe_load(f)

com1 = config["sensor_type"]+"Z"
com2 = config["sensor_type"]+"N"
com3 = config["sensor_type"]+"E"

data_arr = config["datapara"].split(".")
start_time = UTCDateTime(f"{data_arr[0]}{data_arr[1]}T{data_arr[2]}") # year, date, time

sensor_ref = {"q8": {"gain": 4.194300E+05, 
                     "decifactor_1": 1,
                     "decifactor_2": 1,
                     "decifactor_3": 1,
                    },
              "q330": {"gain": 4.194300E+05, 
                     "decifactor_1": 1,
                     "decifactor_2": 1,
                     "decifactor_3": 1,
                      },
              "ct": {"gain": 400000, 
                     "decifactor_1": 1,
                     "decifactor_2": 1,
                     "decifactor_3": 1,
                    },
              "q330hr": {"gain": 1677720 if config['sensor_type'] == "HH" else 4.194300E+05,
                         "decifactor_1": 1,
                         "decifactor_2": 1,
                         "decifactor_3": 1,
                        },
              "q330hrs": {"gain": 1677720 if config['sensor_type'] == "HH" else 1677720,
                         "decifactor_1": 1,
                         "decifactor_2": 1,
                         "decifactor_3": 1,
                        }
             }

# if config["sesnor_name"] not in sensor_ref:
#     sys.exit("error message")

gain = sensor_ref[config['sensor_name']]['gain']
decifactor_1 = sensor_ref[config['sensor_name']]['decifactor_1']
decifactor_2 = sensor_ref[config['sensor_name']]['decifactor_2']
decifactor_3 = sensor_ref[config['sensor_name']]['decifactor_3']


data1 = f"{config['datadir']}/{config['sta1']}.{config['net1']}.{com1}.{config['loc1']}.D.{config['datapara']}"
data2 = f"{config['datadir']}/{config['sta2']}.{config['net2']}.{com2}.{config['loc2']}.D.{config['datapara']}"
data3 = f"{config['datadir']}/{config['sta3']}.{config['net3']}.{com3}.{config['loc3']}.D.{config['datapara']}"

print("# data1 = ",data1)
print(config["resp1"])

pre_filt = [0.0003, 0.0005, 45, 50] # filtering range when instrment reponse is corrected

tw_trim = eval(config["tw_trim"])
print("# tw_trim = ", tw_trim)

data_points = eval(config["data_points"])
print(data_points)

overlap = (7.0/8.0)

# frequency range of figure
xmin = 0.0005 # Hz
xmin = 0.00005 # H

xmax = 10 # Hz
xmax = 50 # Hz

if config["sensor_name"] in sensor_ref:
    # frequency range of figure
    xmin = 0.0005 # H
    xmin = 0.00005 # H

    xmax = 50 # Hz
    #xmax = 250 # Hz

if config["sensor_name"] in sensor_ref:
    pngOUT_fi = f"{config['sta1']}{config['version']}.{config['sensor_name']}_incoherent_noise_{config['sensor_type']}.png"
    print("# pngOUT_fi = ", pngOUT_fi)

    pngOUT_fi_psd = f"{config['sta1']}{config['version']}.{config['sensor_name']}_psd_{config['sensor_type']}.png"
    print("# pngOUT_fi_psd = ", pngOUT_fi_psd)

    pngOUT_fi_coeff = f"{config['sta1']}{config['version']}.{config['sensor_name']}_coeff_{config['sensor_type']}.png"
    print("# pngOUT_fi_coeff = ", pngOUT_fi_coeff)

    pngOUT_fi_psd_eachPSD = f"{config['sta1']}{config['version']}.{config['sensor_name']}_psd_{config['sensor_type']}_eachPSD.png"
    print("# pngOUT_fi_psd_eachPSD = ", pngOUT_fi_psd_eachPSD)

    pngOUT_fi_coh = f"{config['sensor_name']}_coh.png"
    print("# pngOUT_fi_coh = ", pngOUT_fi_coh)
else:
    pngOUT_fi = f"{config['sensor_name']}_incoherent_noise_{config['sensor_type']}.png"
    print("# pngOUT_fi = ", pngOUT_fi)
    pngOUT_fi_psd = f"{config['sensor_name']}_psd_{config['sensor_type']}.png"
    print("# pngOUT_fi_psd = ", pngOUT_fi_psd)
    pngOUT_fi_coh = f"{config['sensor_name']}_coh_{config['sensor_type']}.png"
    print("# pngOUT_fi_coh = ", pngOUT_fi_coh)
    pngOUT_fi_coeff = f"{config['sensor_name']}_coeff_{config['sensor_type']}.png"
    print("# pngOUT_fi_coeff = ", pngOUT_fi_coeff)

bp_para = f"{pre_filt[1]}-{pre_filt[2]}Hz"
print("# bp_para = ", bp_para)
data1_sac = f"{config['sta1']}.{config['net1']}.{com1}.{config['loc1']}.D.{config['datapara']}_{bp_para}_acc.sac"
print("# data1_sac = ", data1_sac)
data2_sac = f"{config['sta2']}.{config['net2']}.{com2}.{config['loc2']}.D.{config['datapara']}_{bp_para}_acc.sac"
print("# data2_sac = ", data2_sac)
data3_sac = f"{config['sta3']}.{config['net3']}.{com3}.{config['loc3']}.D.{config['datapara']}_{bp_para}_acc.sac"
print("# data3_sac = ", data3_sac)

# Color codes for three seismic data
#C B_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
#                  '#f781bf', '#a65628', '#984ea3',
#                  '#999999', '#e41a1c', '#dede00']
color3="#377eb8"
color2="#a65628"
color1="#e41a1c"

# Read seismic data

st1 = read(data1)
st2 = read(data2)
st3 = read(data3)

if config["plotOPT"]:
    _plot = st1.plot(size=(1000,300))
    _plot = st2.plot(size=(1000,300))
    _plot = st3.plot(size=(1000,300))
    st_all_raw = st1 + st2 + st3
    _plot = st_all_raw.plot(size=(1000,500))

# Read response file

## Read seismic data
inv1 =read_inventory(config['resp1'])
inv2 =read_inventory(config['resp2'])
inv3 =read_inventory(config['resp3'])

print(st1)

st1 = st1.merge()
st2 = st2.merge()
st3 = st3.merge()

print(st1)
print(st2)
print(st3)

st1.detrend('linear') # remove linear trend
st1.detrend("demean") # demean
st1.taper(0.05) # cosin taper

if config['sensor_name'] in sensor_ref:
    st1[0].data = st1[0].data/gain
else:
    st1 = st1.remove_response( output="ACC", pre_filt=pre_filt, water_level=None, inventory=inv1)

# decimation
if decifactor_1 == 1:
    print("# no decimatoin")
else:
    st1.decimate(factor=decifactor_1, strict_length=False)

if config['plotOPT']:
    _plot = st1.plot(size=(1000,300))


st2.detrend('linear') # remove linear trend
st2.detrend("demean") # demean
st2.taper(0.05) # cosin taper

if config['sensor_name'] in sensor_ref:
    st2[0].data = st2[0].data/gain
else:
    st2 = st2.remove_response( output="ACC", pre_filt=pre_filt, water_level=None, inventory=inv2)

# decimation
if decifactor_2 == 1:
    print("# no decimation 2")
else:
    st2.decimate(factor=decifactor_2, strict_length=False)

if config['plotOPT']:
    _plot = st2.plot(size=(1000,300))


st3.detrend('linear') # remove linear trend
st3.detrend("demean") # demean
st3.taper(0.05) # cosin taper

if config['sensor_name'] in sensor_ref:
    st3[0].data = st3[0].data/gain
else:
    st3 = st3.remove_response( output="ACC", pre_filt=pre_filt, water_level=None, inventory=inv3)

# decimation
if decifactor_3 == 1:
    print("# no decimation 3")
else:
    st3.decimate(factor=decifactor_3, strict_length=False)

if config['plotOPT']:
    _plot = st3.plot(size=(1000,300))

if config['savesacOPT']:
    st1.write(data1_sac, format="SAC")
    st2.write(data2_sac, format="SAC")
    st3.write(data3_sac, format="SAC")

st_notrim = st1.copy() + st2.copy() + st3.copy()
if config['plotOPT']:
    _plot = st_notrim.copy().taper(0.001).filter("bandpass", freqmin=0.01, freqmax=0.025).plot(size=(1000,400))
print(st_notrim)

st = st1.copy() + st2.copy() + st3.copy()
st.trim(start_time, start_time + tw_trim)

st.detrend('linear') # remove linear trend
st.detrend("demean") # demean
st.taper(0.05) # cosin taper

if config['plotOPT']:
    _plot = st.plot(size=(1000,600))
    _plot = st.copy().taper(0.001).filter("bandpass", freqmin=0.01, freqmax=0.025).plot(size=(1000,400))
    _plot = st.copy().taper(0.001).filter("bandpass", freqmin=0.07, freqmax=0.12).plot(size=(1000,600))
    _plot = st.copy().taper(0.001).filter("bandpass", freqmin=0.02, freqmax=0.05).plot(size=(1000,600))
    _plot = st.copy().taper(0.001).filter("bandpass", freqmin=0.005, freqmax=0.01).plot(size=(1000,600))

if config['waveformplotOPT']:
    fig = plt.figure(figsize=(28, 11.5))

    wf_fl = 0.8
    wf_fh = 1.2

    wd_bp_para = str(wf_fl)+"-"+str(wf_fh)+"Hz"

    t=fig.text(0.13, 0.85, str(wf_fl)+"-"+str(wf_fh)+" Hz BP filter")
    t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white'))

    color3="forestgreen"
    color2="dodgerblue"
    color1="darkorange"

    st1_sncl = st[0].stats.station + '.' + st[0].stats.network + '.' + st[0].stats.location + "." + st[0].stats.channel
    st2_sncl = st[1].stats.station + '.' + st[1].stats.network + '.' + st[1].stats.location + "." + st[1].stats.channel
    st3_sncl = st[2].stats.station + '.' + st[2].stats.network + '.' + st[2].stats.location + "." + st[2].stats.channel

    sncl1 = st1_sncl
    sncl2 = st2_sncl
    sncl3 = st3_sncl

    st1_wf_fil = st[0].copy().taper(0.001).filter("bandpass", freqmin=wf_fl, freqmax=wf_fh)
    st2_wf_fil = st[1].copy().taper(0.001).filter("bandpass", freqmin=wf_fl, freqmax=wf_fh)
    st3_wf_fil = st[2].copy().taper(0.001).filter("bandpass", freqmin=wf_fl, freqmax=wf_fh)

    data_all = np.concatenate((st1_wf_fil.data, st2_wf_fil.data) )

    wf_amp_max = max(data_all.min(), data_all.max(), key=abs)
    wf_amp_max2 = wf_amp_max * 1.1
    wf_amp_min2 = wf_amp_max2 * -1

    plt.plot_date(st[1].times("matplotlib"), st2_wf_fil.data*1, label=sncl2, color=color2, linewidth=1.75, linestyle='solid',  alpha=config['alpha'])
    plt.plot_date(st[0].times("matplotlib"), st1_wf_fil.data*1, label=sncl1, color=color1, linewidth=1.75, linestyle='solid',  alpha=config['alpha'])
    plt.plot_date(st[2].times("matplotlib"), st3_wf_fil.data*1, label=sncl3, color=color3, linewidth=1.75, linestyle='solid',  alpha=config['alpha'])

    myFmt = mdates.DateFormatter("%D/%H:%M")
    myFmt = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")
    plt.gca().xaxis.set_major_formatter(myFmt)


    plt.xlim(st[0].times("matplotlib")[0], st[0].times("matplotlib")[-1])
    plt.ylim(wf_amp_min2 , wf_amp_max2 )
    plt.grid()

    if config['sensor_name'] in sensor_ref:
        plt.ylabel("Amplitude (volt)")
    else:
        plt.ylabel("Amplitude (m/s)")
    plt.legend(loc = "upper right")
    plot_fi =  sncl2+"_"+sncl3+"_"+wd_bp_para+".png"
    plt.savefig(plot_fi, bbox_inches="tight")
    plt.clf()

print(st)

# Compute incoherent−noise with Sleeman method (three stations)
delta = st[0].stats.delta
print("# delta = ", delta)
print(st[0])

nperseg = data_points
noverlap = int(data_points*overlap)
print("# nperseg = ", nperseg)
delta = st[0].stats.delta
print("# delta = ", delta)

fs = 1 / delta
print("# fs = ", fs)

#https://github.com/MideTechnology/endaq-python/blob/main/endaq/calc/psd.py
#nperseg = int(fs / bin_width)
#:param bin_width: the bin width or resolution in Hz for the PSD, defaults to 1,
bin_width = 1  # Hz
bin_width = 0.00166 # 1.66mHz
nperseg_bin_width = int(fs / bin_width)
print("# nperseg_bin_width = ", nperseg_bin_width)
print("# nperseg = ", nperseg)
print("# nperseg = ", nperseg, " noverlap = ", noverlap)

fre1_sp, p11_sp = signal.welch(st[0], 1/delta, nperseg=nperseg, noverlap=noverlap)
fre1_sp, p22_sp = signal.welch(st[1], 1/delta, nperseg=nperseg, noverlap=noverlap)
fre1_sp, p33_sp = signal.welch(st[2], 1/delta, nperseg=nperseg, noverlap=noverlap)

fre1_sp, p21_sp = signal.csd(st[1], st[0], 1/delta, nperseg=nperseg, noverlap=noverlap)
fre1_sp, p13_sp = signal.csd(st[0], st[2], 1/delta, nperseg=nperseg, noverlap=noverlap)
fre1_sp, p23_sp = signal.csd(st[1], st[2], 1/delta, nperseg=nperseg, noverlap=noverlap)

n11_coeff = p21_sp*p13_sp/p23_sp
n22_coeff = np.conjugate(p23_sp)*p21_sp/np.conjugate(p13_sp)
n33_coeff = p23_sp*np.conjugate(p13_sp)/p21_sp

# original
# use coeff
n11_sp = (p11_sp - n11_coeff)
n22_sp = (p22_sp - n22_coeff)
n33_sp = (p33_sp - n33_coeff)

# def
if config['sensor_name'] in sensor_ref:
    print("# no coh")
else:
    #f, Cxy = signal.coherence(x, y, fs, nperseg=1024)
    fre1_coh, c21_coh = signal.coherence(st[1], st[0], 1/delta, nperseg=nperseg, noverlap=noverlap)
    fre1_coh, c13_coh = signal.coherence(st[0], st[2], 1/delta, nperseg=nperseg, noverlap=noverlap)
    fre1_coh, c23_coh = signal.coherence(st[1], st[2], 1/delta, nperseg=nperseg, noverlap=noverlap)

print(p22_sp)
print(p33_sp)

q8_noise_model_fi ="q8_selfnoise.csv"

q8_noise_model = pd.read_csv(q8_noise_model_fi ,
                       sep=",", header=0)
print(q8_noise_model)

# Plotting incoherent−noise

if config['sensor_name'] in sensor_ref:
    # frequency range of figure
    xmin = 0.00001 # H
    xmax = 50 # Hz

plt.rcParams['figure.figsize'] = 18, 9

color3="forestgreen"
color2="dodgerblue"
color1="darkorange"

xout_sp = fre1_sp

st1_sncl = st[0].stats.station + '.' + st[0].stats.network + '.' + st[0].stats.location + "." + st[0].stats.channel
st2_sncl = st[1].stats.station + '.' + st[1].stats.network + '.' + st[1].stats.location + "." + st[1].stats.channel
st3_sncl = st[2].stats.station + '.' + st[2].stats.network + '.' + st[2].stats.location + "." + st[2].stats.channel

# only PSD
plt.plot(xout_sp,10*np.log10(p11_sp),color=color1,linestyle='solid',linewidth = 2.5,label='PSD ' + st1_sncl,  alpha=config['alpha'])
plt.plot(xout_sp,10*np.log10(p22_sp),color=color2,linestyle='solid',linewidth = 2.5,label='PSD ' + st2_sncl,  alpha=config['alpha'])
plt.plot(xout_sp,10*np.log10(p33_sp),color=color3,linestyle='solid',linewidth = 2.5,label='PSD ' + st3_sncl,  alpha=config['alpha'])

plt.grid(True)
plt.xscale("log")
plt.xlim(xmin,xmax)

if config['sensor_name'] in sensor_ref:
    plt.ylim(-150,-70)
else:
    plt.ylim(-210,-70)
    if config["sensor_type"] == "HN":
        plt.ylim(-140,-40)

plt.grid(which='major',color='black',linestyle='-',linewidth = 0.5)
plt.grid(which='minor',color='black',linestyle='-',linewidth = 0.25)

plt.tick_params(labelsize=14)

if config['sensor_name'] == "q8":
    print("# no LNM and HNM")
    plt.ylabel("PSD Power [10log(volt**2/Hz)] (dB)", fontsize=16)
    # psd only
    plt.title('PSD : ' + str(st[0].stats.starttime.year) + ' ' + str(st[0].stats.starttime.julday) + ' ' +      str(st[0].stats.starttime.hour) + ':' + str(st[0].stats.starttime.minute) + ':' + str(st[0].stats.starttime.second) + ' totalTW ' +      str(st[0].stats.npts*delta) + ' seconds TW '+ str(nperseg*delta) + ' seconds')

    plt.plot(q8_noise_model['freq (Hz)'], q8_noise_model['model-upper'], color='black',linestyle='dashed',linewidth = 1.0,label='Q8 NHNM')
    plt.plot(q8_noise_model['freq (Hz)'], q8_noise_model['model-lower'], color='black',linestyle='solid',linewidth = 1.0,label='Q8 NLNM')
elif config['sensor_name'] in sensor_ref:
    print("# no LNM and HNM")
    plt.ylabel("PSD Power [10log(volt**2/Hz)] (dB)", fontsize=16)
    # psd only
    plt.title('PSD : ' + str(st[0].stats.starttime.year) + ' ' + str(st[0].stats.starttime.julday) + ' ' +      str(st[0].stats.starttime.hour) + ':' + str(st[0].stats.starttime.minute) + ':' + str(st[0].stats.starttime.second) + ' totalTW ' +      str(st[0].stats.npts*delta) + ' seconds TW '+ str(nperseg*delta) + ' seconds')
else:
    plt.ylabel("PSD Power [10log(m**2/sec**4/Hz)] (dB)", fontsize=16)

    # psd only
    plt.title('PSD '+config["sensor_type"]+': ' + str(st[0].stats.starttime.year) + ' ' + str(st[0].stats.starttime.julday) + ' ' +      str(st[0].stats.starttime.hour) + ':' + str(st[0].stats.starttime.minute) + ':' + str(st[0].stats.starttime.second) + ' totalTW ' +      str(st[0].stats.npts*delta) + ' seconds TW ' + str(nperseg*delta) + ' seconds')

    model_periods, high_noise = get_nhnm()
    plt.plot(1/model_periods, high_noise, color='black',linestyle='dashed',linewidth = 1.0,label='NHNM')

    model_periods, low_noise = get_nlnm()
    plt.plot(1/model_periods, low_noise, color='black',linestyle='solid',linewidth = 1.0,label='NLNM')

plt.xlabel("Frequency (Hz)", fontsize=16)

plt.legend(loc="upper left", fontsize=14)
plt.legend(loc="upper right", fontsize=14)

if config['pngOUTOPT']:
    plt.savefig(pngOUT_fi_psd)
plt.clf()

# def
if config['sensor_name'] in ["q8", "ct", "q330hr", "q330hrs"]:
    print("# no coh plot")
else:
    plt.rcParams['figure.figsize'] = 18, 9

    color3="forestgreen"
    color2="dodgerblue"
    color1="darkorange"

    xout_sp = fre1_sp

    st1_sncl = st[0].stats.station + '.' + st[0].stats.network + '.' + st[0].stats.location + "." + st[0].stats.channel
    st2_sncl = st[1].stats.station + '.' + st[1].stats.network + '.' + st[1].stats.location + "." + st[1].stats.channel
    st3_sncl = st[2].stats.station + '.' + st[2].stats.network + '.' + st[2].stats.location + "." + st[2].stats.channel

    plt.plot(xout_sp,10*np.log10(n11_sp),color=color1,linewidth = 2.5,label='Noise '+st1_sncl, alpha=config['alpha'])
    plt.plot(xout_sp,10*np.log10(n22_sp),color=color2,linewidth = 2.5,label='Noise '+st2_sncl, alpha=config['alpha'])
    plt.plot(xout_sp,10*np.log10(n33_sp),color=color3,linewidth = 2.5,label='Noise '+st3_sncl, alpha=config['alpha'])

    plt.grid(True)
    plt.xscale("log")
    plt.xlim(xmin,xmax)

    if config['sensor_name'] in sensor_ref:
        plt.ylim(-150,-70)
    else:
        plt.ylim(-210,-70)


    plt.grid(which='major',color='black',linestyle='-',linewidth = 0.5)
    plt.grid(which='minor',color='black',linestyle='-',linewidth = 0.25)

    plt.tick_params(labelsize=14)


    if config['sensor_name'] == 'q8':
        print("# no LNM and HNM")
        plt.ylabel("PSD Power [10log(volt**2/Hz)] (dB)", fontsize=16)
        # psd only
        plt.title('Incoherent-noise PSD : ' + str(st[0].stats.starttime.year) + ' ' + str(st[0].stats.starttime.julday) + ' ' +      str(st[0].stats.starttime.hour) + ':' + str(st[0].stats.starttime.minute) + ':' + str(st[0].stats.starttime.second) + ' totalTW ' +      str(st[0].stats.npts*delta) + ' seconds TW '+ str(nperseg*delta) + ' seconds')

        plt.plot(q8_noise_model['freq (Hz)'], q8_noise_model['model-upper'], color='black',linestyle='dashed',linewidth = 1.0,label='Q8 NHNM')
        plt.plot(q8_noise_model['freq (Hz)'], q8_noise_model['model-lower'], color='black',linestyle='solid',linewidth = 1.0,label='Q8 NLNM')
    elif config['sensor_name'] in sensor_ref:
        print("# no LNM and HNM")
        plt.ylabel("PSD Power [10log(volt**2/Hz)] (dB)", fontsize=16)
        # psd only
        plt.title('Incoherent-noise PSD : ' + str(st[0].stats.starttime.year) + ' ' + str(st[0].stats.starttime.julday) + ' ' +      str(st[0].stats.starttime.hour) + ':' + str(st[0].stats.starttime.minute) + ':' + str(st[0].stats.starttime.second) + ' totalTW ' +      str(st[0].stats.npts*delta) + ' seconds TW '+ str(nperseg*delta) + ' seconds')
        
    else:
        plt.ylabel("PSD Power [10log(m**2/sec**4/Hz)] (dB)", fontsize=16)
        plt.title('Incoherent-noise PSD : ' + str(st[0].stats.starttime.year) + ' ' + str(st[0].stats.starttime.julday) + ' ' +      str(st[0].stats.starttime.hour) + ':' + str(st[0].stats.starttime.minute) + ':' + str(st[0].stats.starttime.second) + ' totalTW ' +      str(st[0].stats.npts*delta) + ' seconds TW '+ str(nperseg*delta) + ' seconds')

        model_periods, high_noise = get_nhnm()
        plt.plot(1/model_periods, high_noise, color='black',linestyle='dashed',linewidth = 1.0,label='NHNM')

        model_periods, low_noise = get_nlnm()
        plt.plot(1/model_periods, low_noise, color='black',linestyle='solid',linewidth = 1.0,label='NLNM')

    plt.xlabel("Frequency (Hz)", fontsize=16)



    plt.legend(loc="upper left", fontsize=14)
    plt.legend(loc="upper right", fontsize=14)

    if config['pngOUTOPT']:
        plt.savefig(pngOUT_fi)

    plt.clf()



# test plot coeff
if config['nocoeffplotOPT']:
    print("# no coeff plot")
else:
    plt.rcParams['figure.figsize'] = 18, 9

    color3="forestgreen"
    color2="dodgerblue"
    color1="darkorange"

    xout_sp = fre1_sp

    st1_sncl = st[0].stats.station + '.' + st[0].stats.network + '.' + st[0].stats.location + "." + st[0].stats.channel
    st2_sncl = st[1].stats.station + '.' + st[1].stats.network + '.' + st[1].stats.location + "." + st[1].stats.channel
    st3_sncl = st[2].stats.station + '.' + st[2].stats.network + '.' + st[2].stats.location + "." + st[2].stats.channel

    plt.plot(xout_sp,10*np.log10(n11_coeff),color=color1,linewidth = 2.5,label='Coeff '+st1_sncl, alpha=config['alpha'])
    plt.plot(xout_sp,10*np.log10(n22_coeff),color=color2,linewidth = 2.5,label='Coeff '+st2_sncl, alpha=config['alpha'])
    plt.plot(xout_sp,10*np.log10(n33_coeff),color=color3,linewidth = 2.5,label='Coeff '+st3_sncl, alpha=config['alpha'])

    plt.grid(True)
    plt.xscale("log")
    plt.xlim(xmin,xmax)

    if config['sensor_name'] in sensor_ref:
        plt.ylim(-150,-70)
    else:
        plt.ylim(-210,-70)


    plt.grid(which='major',color='black',linestyle='-',linewidth = 0.5)
    plt.grid(which='minor',color='black',linestyle='-',linewidth = 0.25)

    plt.tick_params(labelsize=14)


    if config['sensor_name'] == 'q8':
        print("# no LNM and HNM")
        plt.ylabel("PSD Power [10log(volt**2/Hz)] (dB)", fontsize=16)
        # psd only
        plt.title('Coeff PSD : ' + str(st[0].stats.starttime.year) + ' ' + str(st[0].stats.starttime.julday) + ' ' +      str(st[0].stats.starttime.hour) + ':' + str(st[0].stats.starttime.minute) + ':' + str(st[0].stats.starttime.second) + ' totalTW ' +      str(st[0].stats.npts*delta) + ' seconds TW '+ str(nperseg*delta) + ' seconds')

        plt.plot(q8_noise_model['freq (Hz)'], q8_noise_model['model-upper'], color='black',linestyle='dashed',linewidth = 1.0,label='Q8 NHNM')
        plt.plot(q8_noise_model['freq (Hz)'], q8_noise_model['model-lower'], color='black',linestyle='solid',linewidth = 1.0,label='Q8 NLNM')
    elif config['sensor_name'] in sensor_ref:
        print("# no LNM and HNM")
        plt.ylabel("PSD Power [10log(volt**2/Hz)] (dB)", fontsize=16)
        # psd only
        plt.title('Coeff PSD : ' + str(st[0].stats.starttime.year) + ' ' + str(st[0].stats.starttime.julday) + ' ' +      str(st[0].stats.starttime.hour) + ':' + str(st[0].stats.starttime.minute) + ':' + str(st[0].stats.starttime.second) + ' totalTW ' +      str(st[0].stats.npts*delta) + ' seconds TW '+ str(nperseg*delta) + ' seconds')
    else:
        plt.ylabel("PSD Power [10log(m**2/sec**4/Hz)] (dB)", fontsize=16)
        plt.title('Coeff PSD : ' + str(st[0].stats.starttime.year) + ' ' + str(st[0].stats.starttime.julday) + ' ' +      str(st[0].stats.starttime.hour) + ':' + str(st[0].stats.starttime.minute) + ':' + str(st[0].stats.starttime.second) + ' totalTW ' +      str(st[0].stats.npts*delta) + ' seconds TW '+ str(nperseg*delta) + ' seconds')

        model_periods, high_noise = get_nhnm()
        plt.plot(1/model_periods, high_noise, color='black',linestyle='dashed',linewidth = 1.0,label='NHNM')

        model_periods, low_noise = get_nlnm()
        plt.plot(1/model_periods, low_noise, color='black',linestyle='solid',linewidth = 1.0,label='NLNM')

    plt.xlabel("Frequency (Hz)", fontsize=16)

    plt.legend(loc="upper left", fontsize=14)
    plt.legend(loc="upper right", fontsize=14)

    if config['pngOUTOPT']:
        plt.savefig(pngOUT_fi_coeff)

    plt.clf()
