import obspy as ob
from obspy import read
from obspy import UTCDateTime
from obspy import read, read_inventory

from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import URL_MAPPINGS

from obspy.geodetics.base import gps2dist_azimuth

from obspy.signal.rotate import rotate_ne_rt
from obspy.signal.rotate import rotate_rt_ne
from obspy.signal.rotate import rotate2zne

import numpy as np

import scipy as sp
from scipy.fftpack import fft, ifft
from scipy.linalg import norm
from scipy import ndimage
from scipy import signal
from scipy.stats import pearsonr

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates 

import sys
import os
import yaml
import re
import argparse


print("Waveform Similarity")
print(f"Versions: obspy {ob.__version__}, numpy {np.__version__}, scipy {sp.__version__}, matplotlib {mpl.__version__}\n")

# Get file directory
file_path = os.path.realpath(__file__)
directory = os.path.dirname(file_path)

flags = {"config": ("-c", str),
         "sta1": ("-s1", str), "net1": ("-n1", str), "com1": ("-c1", str), "loc1": ("-l1", str), 
         "sta2": ("-s2", str), "net2": ("-n2", str), "com2": ("-c2", str), "loc2": ("-l2", str), 
         "st": ("-st", str), "minmag": ("-mm", float), "localEQOPT": ("-eq", bool)}

parser = argparse.ArgumentParser(description="Flags for waveform_similarity.py")

for key in flags.keys():
    var = key
    flag, vartype = flags[key]
    parser.add_argument(flag, f"--{var}", metavar='', type=vartype)

args, unknown = parser.parse_known_args()
args_dict = args.__dict__

config_name = "waveform_similarity.yml"

if args_dict["config"]:
    config_name = args_dict["config"]

with open(f"{directory}/{config_name}") as c:
    config = yaml.safe_load(c)

for var in flags.keys():
    if args_dict[var]:
        config[var] = args_dict[var]

# Set up font
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

### Functions

# Extract seed_id from obspy stream
def get_seedid(tr):
    return f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}"

# Correct instrument response
def st_remove_resp_inv(st, deciopt, decifactor, pre_filt, output, inv):
    st.detrend("linear") # remove linear trend
    st.detrend("demean") # demean
    st.taper(0.05) # cosin taper

    if deciopt == 1:
        #decimate to 100Hz
        if decifactor == 100:
            st.decimate(5, strict_length=False)
            st.decimate(2, strict_length=False)

            st.decimate(5, strict_length=False)
            st.decimate(2, strict_length=False)
        else:
            st.decimate(factor=decifactor, strict_length=False)
    
    print(inv)
    st = st.remove_response(pre_filt=pre_filt,output=output,water_level=None, inventory=inv) # get velocity data (m/s)
    return st

def st_remove_resp (st, deciopt, decifactor, pre_filt, output):
    st.detrend("linear") # remove linear trend
    st.detrend("demean") # demean
    st.taper(0.05) # cosin taper

    if deciopt == 1:
        # decimate to 100Hz
        if decifactor == 100:     
            st.decimate(5, strict_length=False)
            st.decimate(2, strict_length=False)

            st.decimate(5, strict_length=False)
            st.decimate(2, strict_length=False)

        elif decifactor == 200:     
            st.decimate(2, strict_length=False)

            st.decimate(5, strict_length=False)
            st.decimate(2, strict_length=False)

            st.decimate(5, strict_length=False)
            st.decimate(2, strict_length=False)

        elif decifactor == 500:     
            st.decimate(5, strict_length=False)

            st.decimate(5, strict_length=False)
            st.decimate(2, strict_length=False)

            st.decimate(5, strict_length=False)
            st.decimate(2, strict_length=False)
        else:
            st.decimate(factor=decifactor, strict_length=False)
    
    st = st.remove_response(pre_filt=pre_filt,output=output,water_level=None) # get velocity data (m/s)

    return st

# Extract station coordinate from obspy stream
def get_sta_coord(seedid, inv, starttime):
    return inv.get_coordinates(seedid, starttime)

# Extract sensor orientation from obspy stream
def get_sta_orientation(seedid, inv, starttime):
    return inv.get_orientation(seedid, starttime)

# Rotate 3-com data into ZNE coordinate
def get_zne_data (st, inv, starttime):
    if len(st) != 3:
        sys.exit(f"The length of st must be 3.")
        
    sta_coordinate = []
    sta_orientation = []

    for i, tr in enumerate(st, 1):
        seedid=get_seedid(tr)

        sta_coordinate.append(get_sta_coord(seedid, inv, starttime))
        sta_orientation.append(get_sta_orientation(seedid, inv, starttime))

        print("# seedid = ",seedid, sta_coordinate)
        print("# seedid = ",seedid, sta_orientation)

    print("# st2 = ",st[2])
    print("# st1 = ",st[1])
    print("# st0 = ",st[0])
    ztmp = st[2]
    AzZ = sta_orientation[2]['azimuth']
    DipZ = sta_orientation[2]['dip']
    
    ntmp = st[1]
    AzN = sta_orientation[1]['azimuth']
    DipN = sta_orientation[1]['dip']

    etmp = st[0]
    AzE = sta_orientation[0]['azimuth']
    DipE = sta_orientation[0]['dip']
    
    print("# AzZ = ", AzZ, " DipZ = ", DipZ)
    print("# AzN = ", AzN, " DipN = ", DipN)
    print("# AzE = ", AzE, " DipE = ", DipE)

    # input is UVW case
    # ENZ output U -> E, V -> N, W -> Z 
    # XYZ -> UVW
    staout = seedid.split(".")[1]
    print("# staout = ", staout)
    if(staout == "BK02"):
        # U
        AzZ = 210.0
        DipZ = 54.7-90.0

        # V
        AzN = 330.0
        DipN = 54.7-90.0
        
        # W
        
        AzE = 90.0
        DipE = 54.7-90.0

        print("# AzZ (U) = ", AzZ, " DipZ (U) = ", DipZ)
        print("# AzN (V) = ", AzN, " DipN (V) = ", DipN)
        print("# AzE (W) = ", AzE, " DipE (W) = ", DipE)

    t1z , t1n, t1e = rotate2zne(ztmp,AzZ,DipZ,ntmp,AzN,DipN,etmp,AzE,DipE)
    st[0].data = t1e
    st[1].data = t1n
    st[2].data = t1z


    st[0].stats.channel = st[0].stats.channel[:-1] + "E"
    st[1].stats.channel = st[0].stats.channel[:-1] + "N"
    st[2].stats.channel = st[0].stats.channel[:-1] + "Z"

    return st

# To compute back-azimuth
def get_baz(st, inv, evla, evlo):
    seedid=get_seedid(st[0])
    sta_coord = get_sta_coord(seedid,inv,starttime)
    
    stla = sta_coord['latitude']
    stlo = sta_coord['longitude']
    
    source_latitude = evla
    source_longitude = evlo
    
    station_latitude = stla
    station_longitude = stlo
    
    # theoretical backazimuth and distance
    baz = gps2dist_azimuth(source_latitude, source_longitude, station_latitude, station_longitude)

    print('Epicentral distance [m]: ', baz[0])
    print('Epicentral distance [km]: ', baz[0]/1000.0)
    print('Theoretical azimuth [deg]: ', baz[1])
    print('Theoretical backazimuth [deg]: ', baz[2])
    
    return baz

# Will add coherency later
def waveform_comparison (st, plotOPT, slideOPT, logOPT, manualOPT, fl=1, fh=10):    
    # # font size
    # SMALL_SIZE = 16
    # MEDIUM_SIZE = 18
    # BIGGER_SIZE = 20

    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    sncl1 = get_seedid(st[0])
    sncl2 = get_seedid(st[1])

    plot_fi = f"{plot_dir}/{sncl2}_{sncl1}_{event_para}.pdf"

    T1=st[0].stats.delta
    N1=st[0].stats.npts
    xf1 = np.linspace(0.0, 1.0/(2.0*T1), N1//2)
    yf1 = fft(st[0].data)

    T2=st[1].stats.delta
    N2=st[1].stats.npts
    xf2 = np.linspace(0.0, 1.0/(2.0*T2), N2//2)
    yf2 = fft(st[1].data)

    # correlation
    cc_pearsonr, p_pearsonr = pearsonr(st[0].data, st[1].data)
    cc_pearsonr_out = "{:.3f}".format(cc_pearsonr)
    print("#RESULT cc_pearsonr_out = ",cc_pearsonr_out," ",sncl1," ",sncl2)

    distkm1_out = "{:.3f}".format(baz1[0]/1000.0)
    distkm2_out = "{:.3f}".format(baz2[0]/1000.0)
    print("#RESULT distkm1_out = ",distkm1_out," ",sncl1," ",sncl2)
    print("#RESULT distkm2_out = ",distkm2_out," ",sncl1," ",sncl2)
    
    # no div N1 or N2 match sac results....
    yf1_amp =  2.0/1.0 * np.abs(yf1)
    yf2_amp =  2.0/1.0 * np.abs(yf2)
    yf_amp_ratio = yf2_amp / yf1_amp
    
    index_select = np.where(  (fl <= xf2) & (xf2 <= fh) )

    yf_amp_max =  np.amax(np.concatenate((yf1_amp[index_select], yf2_amp[index_select]) ))
    yf_amp_min =  np.amin(np.concatenate((yf1_amp[index_select], yf2_amp[index_select]) ))

    yf_amp_max2 = yf_amp_max * 5
    yf_amp_min2 = yf_amp_min * 0.5

    index_select_num = yf_amp_ratio[index_select].shape[0]

    amp_median = ndimage.median(yf_amp_ratio[index_select])
    amp_l1 = norm(yf_amp_ratio[index_select]-amp_median, 1)/index_select_num

    amp_l1_out = "{:.3f}".format(amp_l1)
    amp_median_out = "{:.3f}".format(amp_median)

    print("#RESULT amp_median_out = ",amp_median_out," amp_l1_out = ",amp_l1_out," ",sncl1," ",sncl2)

    yf1_ph = np.angle(yf1)
    yf2_ph = np.angle(yf2)
    
    yf1_ph = np.where(yf1_ph <= -np.pi,  yf1_ph + 2*np.pi,  yf1_ph)
    yf1_ph = np.where(np.pi <= yf1_ph,  yf1_ph - 2*np.pi,  yf1_ph)

    yf2_ph = np.where(yf2_ph <= -np.pi,  yf2_ph + 2*np.pi,  yf2_ph)
    yf2_ph = np.where(np.pi <= yf2_ph,  yf2_ph - 2*np.pi,  yf2_ph)

    yf_ph_diff = yf2_ph - yf1_ph

    # phase diff also. need to check it later
    yf_ph_diff = np.where(yf_ph_diff <= -np.pi,  yf_ph_diff + 2*np.pi,  yf_ph_diff)
    yf_ph_diff = np.where(np.pi <= yf_ph_diff,  yf_ph_diff - 2*np.pi,  yf_ph_diff)
    
    ph_median = ndimage.median(yf_ph_diff[index_select])
    ph_l1 = norm(yf_ph_diff[index_select]-ph_median, 1)/index_select_num
    
    ph_l1_out = "{:.3f}".format(ph_l1)
    ph_median_out = "{:.3f}".format(ph_median)
    print("#RESULT ph_median_out = ",ph_median_out," ph_l1_out = ", ph_l1_out, " ",sncl1, " ", sncl2)
    print("#RESULT2 ",sncl1," ",sncl2," ",amp_median_out," ",amp_l1_out," ",ph_median_out," ",ph_l1_out," ",cc_pearsonr_out)

    yf_amp_max =  np.amax(np.concatenate((yf1_amp[index_select], yf2_amp[index_select]) ))

    data_all = np.concatenate((st[0].data, st[1].data) )
    wf_amp_max = max(data_all.min(), data_all.max(), key=abs)

    wf_amp_max2 = wf_amp_max * 1.1
    wf_amp_min2 = wf_amp_max2 * -1
    
    if plotOPT: 
        gs = gridspec.GridSpec(3,2)

        # slide
        if slideOPT:
            fig = plt.figure(figsize=(28, 11.5))
            plt.subplots_adjust(wspace=0.15, hspace=0.45)
        else:
            # def
            fig = plt.figure(figsize=(16, 16))
            plt.subplots_adjust(wspace=0.4, hspace=0.4)

        t=fig.text(0.13, 0.85, str(fl)+"-"+str(fh)+" Hz BP filter")
        t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white'))
        t=fig.text(0.13, 0.70, "CC = "+str(cc_pearsonr_out))
        t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white'))

        if slideOPT:
            t=fig.text(0.48, 0.295, "Median: "+str(amp_median_out), ha='right')
            t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white'))
            t=fig.text(0.48, 0.275, "L1 norm: "+str(amp_l1_out), ha='right')
            t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white'))
        else:
            t=fig.text(0.44, 0.295, "Median: "+str(amp_median_out), ha='right')
            t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white'))
            t=fig.text(0.44, 0.275, "L1 norm: "+str(amp_l1_out), ha='right')
            t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white'))

        t=fig.text(0.89, 0.160, "Median: "+str(ph_median_out), ha='right')
        t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white'))
        t=fig.text(0.89, 0.140, "L1 norm: "+str(ph_l1_out), ha='right')
        t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white'))

        # waveforms
        plt.subplot(gs[0, :])
        plt.plot_date(st[0].times("matplotlib"), st[0].data*1, fmt='-', label=sncl1, color="red",  linewidth=0.75, linestyle='solid')
        plt.plot_date(st[1].times("matplotlib"), st[1].data*1, fmt='-', label=sncl2, color="blue", linewidth=0.75, linestyle='solid')

        myFmt = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S") 
        plt.gca().xaxis.set_major_formatter(myFmt) 

        plt.xlim(st[0].times("matplotlib")[0], st[0].times("matplotlib")[-1])
        plt.ylim(wf_amp_min2 , wf_amp_max2 )
        plt.grid()

        if manualOPT:
            print("# no title")
        else:
            plt.title(event_para+" \n Origin Time:"+str(origin_time)+" USGS event-id:"+evid+" \n M"+str(evmag)+" "+event_region)

        plt.ylabel("Amplitude (m/s)")
        plt.legend(loc = "upper right")
        
        # Amplitude spectra
        plt.subplot(gs[1,0])

        plt.plot(xf1, (yf1_amp[0:N1//2]), label=sncl1, color="red", linewidth=0.75)
        plt.plot(xf2, (yf2_amp[0:N2//2]), label=sncl2, color="blue", linewidth=0.75)
        plt.xlim(fl,fh)
        plt.ylim(yf_amp_min2, yf_amp_max2)
        plt.grid()
        
        plt.yscale("log")
        if logOPT:
            plt.xscale("log")
        
        plt.title("Amplitude spectra")
        plt.ylabel("Amplitude (m$^2$/s$^2$/Hz)")
        plt.xlabel("Frequency (Hz)")
        plt.legend(loc = "upper right")

        # Phase spectra
        plt.subplot(gs[1,1])

        plt.plot(xf1, (yf1_ph[0:N1//2]), label=sncl1, color="red", linewidth=0.75)
        plt.plot(xf2, (yf2_ph[0:N2//2]), label=sncl2, color="blue", linewidth=0.75)
        
        plt.xlim(fl,fh)
        plt.ylim(-4,4)
        plt.grid()

        if logOPT:
            plt.xscale("log")
            
        plt.title("Phase spectra")
        plt.ylabel("Phase (rad)")
        plt.xlabel("Frequency (Hz)")
        plt.legend(loc = "upper right")     
        
        # Amplitude ratio
        plt.subplot(gs[2,0])

        plt.plot(xf1, (yf_amp_ratio[0:N1//2]), color="black", linewidth=0.75)
        plt.plot([fl, fh], [amp_median, amp_median], color="black", linewidth=1.25, linestyle="dashed")

        plt.xlim(fl,fh)
        plt.ylim(0.5,1.5)
        plt.grid()

        if logOPT:
            plt.xscale("log")

        plt.title("Spectral amplitude ratio")
        plt.ylabel("Amp. ("+sncl2+"/\n"+sncl1+")")       
        plt.xlabel("Frequency (Hz)")
        
        # Phase diff
        plt.subplot(gs[2,1])

        plt.plot(xf1, (yf_ph_diff[0:N1//2]), color="black", linewidth=0.75)
        delay_sec = (yf_ph_diff[0:N1//2])/(2*np.pi*xf1)

        st[0].delay_sec = delay_sec
        st[0].xf1 = xf1
        plt.plot([fl, fh], [ph_median, ph_median], color="black", linewidth=1.25, linestyle="dashed")

        plt.xlim(fl,fh)
        plt.ylim(-1.0,1.0)
        plt.grid()

        if logOPT:
            plt.xscale("log")


        plt.title("Difference in spectral phase")
        plt.ylabel("Phase ("+sncl2+"-\n"+sncl1+")")

        plt.xlabel("Frequency (Hz)")
        
        plt.savefig(plot_fi, bbox_inches="tight")

clientEQ = Client(config["client"])
print(clientEQ)

### Event search

st = UTCDateTime(config["st"])
minmag = config["minmag"]
localEQOPT = config["localEQOPT"]

et = st + 120
maxmag = minmag + 0.2

print("# st = {0} et = {1}".format(st, et))
print("# minmag = {0} maxmag = {1}".format(minmag, maxmag))

catalog = clientEQ.get_events(starttime=st , endtime=et, minmagnitude=minmag, maxmagnitude=maxmag)
print(catalog)

plotOPT = config["plotOPT"]

if plotOPT:
    catalog.plot()

### Extract event information

# event info. origin time, location, magnitude
event = catalog[0]
origin = event.origins[0]
origin_time = origin.time
evla = origin.latitude
evlo = origin.longitude
evdp_km = origin.depth / 1000
evmag = event.magnitudes[0].mag

evyearOUT = origin_time.year
evjdayOUT = origin_time.julday
evhourOUT = origin_time.hour
evminOUT = origin_time.minute
evsecOUT = origin_time.second

evid = event.origins[0]['extra']['dataid']['value']
try:
    event_region = event.event_descriptions[0]['text']
except:
    event_region = ""
print("# evid = ", evid)
print("# event_region = ", event_region)

auth = event.origins[0]['creation_info']['agency_id']
print("# auth = ",auth)

# need for file name
evyearOUT2 = (str)(evyearOUT)
evjdayOUT2 = (str)(evjdayOUT)
if evjdayOUT < 100:
    evjdayOUT2 = "0"+(str)(evjdayOUT)
elif evjdayOUT < 10:
    evjdayOUT2 = "00"+(str)(evjdayOUT)

evhourOUT2 = (str)(evhourOUT)
if evhourOUT < 10:
    evhourOUT2 = "0"+(str)(evhourOUT)
        

evminOUT2 = (str)(evminOUT)
if evminOUT < 10:
    evminOUT2 = "0"+(str)(evminOUT)


evsecOUT2 = (str)(evsecOUT)
if evsecOUT < 10:
    evsecOUT2 = "0"+(str)(evsecOUT)
        
evmseedid = f"{evyearOUT2}.{evjdayOUT2}.{evhourOUT2}.{evminOUT2}{evsecOUT2}"
event_para = f"{evmseedid}_M{evmag}_{evid}"

### Directory for waveform plot
# This example will create directory "2020.204.061244_M7.8_us7000asvb" where all plots will be saved.

pwd_dir = os.getcwd()
plot_dir= pwd_dir +"/"+ event_para

print("# plot_dir = ",plot_dir)

# create output directory
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


pre_tw = config["pre_tw"]
stw = eval(config["stw"])
if localEQOPT:
    stw = 1.5*60 # 1.5 min from the starting time (oriting_time + pre_tw)

starttime = origin_time + pre_tw  
endtime = starttime + stw

manualOPT = config["manualOPT"]

if manualOPT:
    starttime =UTCDateTime("2021-07-31T02:22:00")
    endtime = starttime + 20

    starttime =UTCDateTime("2023-07-27T06:50:57")
    endtime = starttime + 40

print("# starttime = ",starttime)
print("# endtime   = ",endtime)

sta1 = config["sta1"]
net1 = config["net1"]
com1 = config["com1"]
loc1 = config["loc1"]

if sta1 == "BRIB":
    loc1 = "01" # location
if sta1 == "TRIN":
    loc1 = "01" # location    
if sta1 == "SCOT":
    loc1 = "01" # location  
if sta1 == "SUTB":
    loc1 = "01" # location 
if sta1 == "DCMP":
    loc1 = "01" # location     
if sta1 == "SHEP":
    loc1 = "01" # location 

deciopt_1 = 0
decifactor_1 = 100 # 100sps -> 1sps

if sta1 == "TCAS" or sta1 == "TCHL" or sta1 == "THIS" or sta1 == "TRAM" or sta1 == "TRAY" or sta1 == "TSCN" or sta1 == "TSCS" or sta1 == "MBARI":
    deciopt_1 = 1
    decifactor_1 = 2

client1 = Client("SCEDC") # data from NCEDC
client1.base_url = 'https://service.ncedc.org'
print(client1)

print("# sta1 = ", sta1)
print("# net1 = ", net1)
print("# com1 = ", com1)
print("# loc1 = ", loc1)

sta2 = config["sta2"] 
net2 = config["net2"] 
com2 = config["com2"] 
loc2 = config["loc2"] 

if sta2 == "BRIB":
    loc2 = "01" # location
if sta2 == "TRIN":
    loc2 = "01" # location
if sta2 == "SCOT":
    loc2 = "01" # location
if sta2 == "SUTB":
    loc2 = "01" # location
if sta2 == "DCMP":
    loc2 = "01" # location
if sta2 == "SHEP":
    loc2 = "01" # location

deciopt_2 = 0
decifactor_2 = 100

if sta2 == "TCAS" or sta2 == "TCHL" or sta2 == "THIS" or sta2 == "TRAM" or sta2 == "TRAY" or sta2 == "TSCN" or sta2 == "TSCS" or sta2 == "MBARI":
    deciopt_2 = 1
    decifactor_2 = 2

client2 = Client("SCEDC") # data from NCEDC
client2.base_url = 'https://service.ncedc.org'
print(client1)

print("# sta2 = ", sta2)
print("# net2 = ", net2)
print("# com2 = ", com2)
print("# loc2 = ", loc2)

bv1OPT = 0
bv2OPT = 0

if re.findall('BK+[0-9]', sta1) != []:
    bv1OPT = 1
if re.findall('BK+[0-9]', sta2) != []:
    bv2OPT = 1
if sta1 == "MBARI":
    bv1OPT = 1
if sta2 == "MBARI":
    bv2OPT = 1
print("# bv1OPT = ", bv1OPT)
print("# bv2OPT = ", bv2OPT)

transformOPT = config["transformOPT"]

client1.base_url = 'https://service.ncedc.org'
client2.base_url = 'https://service.ncedc.org'
if transformOPT:
    client1.base_url = 'http://transform.geo.berkeley.edu:8080'
    client2.base_url = 'http://transform.geo.berkeley.edu:8080'

waveformcheckOPT = config["waveformcheckOPT"]
if waveformcheckOPT:
    if bv1OPT == 1:
        st1 = client1.get_waveforms_testdata(network=net1, station=sta1, location=loc1, channel=com1,
                        starttime=starttime, endtime=endtime, 
                         attach_response=False)
    else:
        st1 = client1.get_waveforms(network=net1, station=sta1, location=loc1, channel=com1,
                        starttime=starttime, endtime=endtime, 
                         attach_response=False)
    if bv2OPT == 1:  
        st2 = client2.get_waveforms_testdata(network=net2, station=sta2, location=loc2, channel=com2,
                         starttime=starttime, endtime=endtime, 
                         attach_response=False)
    else:
        st2 = client2.get_waveforms(network=net2, station=sta2, location=loc2, channel=com2,
                         starttime=starttime, endtime=endtime, 
                         attach_response=False)

if waveformcheckOPT:
    if plotOPT:
        st_tmp = st1.copy() + st2.copy()

        plot_st = UTCDateTime("2021-12-20T20:10:50")  # 
        plot_et = UTCDateTime("2021-12-20T20:11:00")  #
        st_tmp.sort(keys=['channel'])   
        print(st_tmp)
        _plot = st_tmp.copy().detrend("linear").plot(equal_scale=False, size=(1200,600))
        _plot = st_tmp.copy().detrend("linear").taper(0.001).filter("bandpass", freqmin=2, freqmax=8, 
                                                                    zerophase=True).interpolate(sampling_rate=100.0).plot(equal_scale=False, size=(1200,600))

        _plot = st_tmp.copy().detrend("linear").taper(0.001).filter("bandpass", freqmin=1, freqmax=10, 
                                                                    zerophase=True).interpolate(sampling_rate=100.0).plot(equal_scale=False, size=(1200,600))
        _plot = st_tmp.copy().detrend("linear").taper(0.001).filter("bandpass", freqmin=5, freqmax=10, 
                                                                    zerophase=True).interpolate(sampling_rate=100.0).plot(equal_scale=False, size=(1200,600))
        _plot = st_tmp.copy().detrend("linear").taper(0.001).filter("bandpass", freqmin=0.5, freqmax=2.0, 
                                                                    zerophase=True).interpolate(sampling_rate=100.0).plot(equal_scale=False, size=(1200,600))

        _plot = st_tmp.copy().detrend("linear").taper(0.001).filter("highpass", freq=0.5, 
                                                                    zerophase=True).interpolate(sampling_rate=100.0).plot(equal_scale=False, size=(1200,600))
    
        _plot = st_tmp.copy().detrend("linear").taper(0.01).filter("bandpass", freqmin=0.02, freqmax=0.05, zerophase=True).plot(equal_scale=False, size=(1200,600))

        _plot = st_tmp.copy().detrend("linear").taper(0.01).filter("bandpass", freqmin=0.05, freqmax=0.1, zerophase=True).plot(equal_scale=False, size=(1200,600))
        _plot = st_tmp.copy().detrend("linear").taper(0.01).filter("bandpass", freqmin=0.1, freqmax=1.0, zerophase=True).plot(equal_scale=False, size=(1200,600))

locact_config = config["localTrue"] if localEQOPT else config["localFalse"]

### Set time window for waveform similarity
Rvel = locact_config["Rvel"]
tw_pre_tw = locact_config["tw_pre_tw"]
tw_trim = locact_config["tw_trim"]

fl2 = locact_config["fl2"]
fl = locact_config["fl"]
fh = locact_config["fh"]
fh2 = locact_config["fh2"]
logOPT = locact_config["logOPT"]

pre_filt = [fl2, fl, fh, fh2]
print("# pre_filt = ", pre_filt)

### Downloading seismic data
# Use get_waveforms to download data and do st.plot() for plotting.
# Also use get_stations to obtain station inventory file.

if sta1 == "MBARIX":
    mseedid = "2022.351.113942"
    st1 = read("/Users/taira/work/python_work/MBARI/mseed/"+sta1+".BK.HHE.00.D."+mseedid)
    st1 += read("/Users/taira/work/python_work/MBARI/mseed/"+sta1+".BK.HHN.00.D."+mseedid)
    st1 += read("/Users/taira/work/python_work/MBARI/mseed/"+sta1+".BK.HHZ.00.D."+mseedid)
    st1.trim(starttime, endtime)  
    inv1 = read_inventory("station_MBARI.xml")
elif sta1 == "BK90x" or sta1 == "BK91x":
    mseedid = "2023.355.230102"
    st1 = read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta1+".BK.HHE.00.D."+mseedid)
    st1 += read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta1+".BK.HHN.00.D."+mseedid)
    st1 += read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta1+".BK.HHZ.00.D."+mseedid)
    st1.trim(starttime, endtime)  
    inv1 = read_inventory("station_BVtest.xml")
elif sta1 == "CCORX":
    print("# read")
    st1 = client1.get_waveforms(network=net1, station=sta1, location=loc1, channel=com1,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    sis_dir = "https://files.anss-sis.scsn.org/preview"
    sis_key = "prod"
    inv_fi = sis_dir+"/preview_"+sis_key+"_"+net1+"_"+sta1+".xml"
    inv1 = read_inventory(inv_fi)
elif sta1 == "VAK":
    print("# read")
    st1 = client1.get_waveforms(network=net1, station=sta1, location=loc1, channel=com1,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    sis_dir = "https://files.anss-sis.scsn.org/preview"
    sis_key = "prod"
    inv_fi = sis_dir+"/preview_"+sis_key+"_"+net1+"_"+sta1+".xml"
    inv1 = read_inventory(inv_fi)
    
elif sta1 == "PABX":
    mseedid = "2021.225.054555"
    st1 = read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta1+".BK.HHE.00.D."+mseedid)
    st1 += read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta1+".BK.HHN.00.D."+mseedid)
    st1 += read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta1+".BK.HHZ.00.D."+mseedid)
    st1.trim(starttime, endtime)  
    inv1 = read_inventory("station_BVtest.xml")
elif sta1 == "MERX":
    st1 = read("http://ncedc.org/ftp/outgoing/taira/MERC.BK.HHE.00.D.2020.275.003127")
    st1 += read("http://ncedc.org/ftp/outgoing/taira/MERC.BK.HHN.00.D.2020.275.003127")
    st1 += read("http://ncedc.org/ftp/outgoing/taira/MERC.BK.HHZ.00.D.2020.275.003127")
    st1.trim(starttime, endtime)  
    inv1 = read_inventory("station_BVtest.xml")
elif bv1OPT == 1:
    oldOPT = 0
    if oldOPT:
        mseedid = "2022.261.064414"
        chname = "HH"
        if sta1 == "BK17":
            chname = "DP"
            loc1 = "40"
            inv1 = read_inventory("http://ncedc.org/ftp/outgoing/taira/WQC/BK.BK17.xml")

            st1 = read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta1+".BK."+chname+"3."+loc1+".D."+mseedid)
            st1 += read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta1+".BK."+chname+"2."+loc1+".D."+mseedid)
            st1 += read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta1+".BK."+chname+"1."+loc1+".D."+mseedid)
        else:
            inv1 = read_inventory("station_BVtest.xml")

            st1 = read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta1+".BK."+chname+"E."+loc1+".D."+mseedid)
            st1 += read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta1+".BK."+chname+"N."+loc1+".D."+mseedid)
            st1 += read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta1+".BK."+chname+"Z."+loc1+".D."+mseedid)

        st1.trim(starttime, endtime)  
    st1 = client1.get_waveforms_testdata(network=net1, station=sta1, location=loc1, channel=com1,
                    starttime=starttime, endtime=endtime, 
                     attach_response=False)
    inv1 = read_inventory("station_BVtest.xml")
    if sta1 == "MBARI":
        inv1 = read_inventory("station_MBARI.xml")
elif sta1 == "BK83":
    mseedid = "2021.275.062918"
    chname = "LH"
    chname = "HH"
    st1 = read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta1+".BK."+chname+"E.00.D."+mseedid)
    st1 += read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta1+".BK."+chname+"N.00.D."+mseedid)
    st1 += read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta1+".BK."+chname+"Z.00.D."+mseedid)
    st1.trim(starttime, endtime)  
    inv1 = read_inventory("BK.BK83.xml")
elif sta1 == "BK81":
    mseedid = "2021.275.062918"
    chname = "LH"
    chname = "HH"
    st1 = read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta1+".BK."+chname+"E.01.D."+mseedid)
    st1 += read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta1+".BK."+chname+"N.01.D."+mseedid)
    st1 += read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta1+".BK."+chname+"Z.01.D."+mseedid)    
    st1.trim(starttime, endtime)  
    inv1 = read_inventory("BK.BK81.xml")
elif sta1 == "TOLX":
    st1 = read("http://ncedc.org/ftp/outgoing/taira/TOLH.BK.HHE.00.D.2020.276.123247")
    st1 += read("http://ncedc.org/ftp/outgoing/taira/TOLH.BK.HHN.00.D.2020.276.123247")
    st1 += read("http://ncedc.org/ftp/outgoing/taira/TOLH.BK.HHZ.00.D.2020.276.123247")   
    st1.trim(starttime, endtime)  
    inv1 = read_inventory("station_BVtest.xml")
elif sta1 == "BCCX":
    st1 = read("http://ncedc.org/ftp/outgoing/taira/BCCR.BK.HHE.00.D.2020.284.012411")
    st1 += read("http://ncedc.org/ftp/outgoing/taira/BCCR.BK.HHN.00.D.2020.284.012411")
    st1 += read("http://ncedc.org/ftp/outgoing/taira/BCCR.BK.HHZ.00.D.2020.284.012411")   
    inv1 = read_inventory("station_BVtest.xml")
elif sta1 == "PORX":
    print("# read")
    st1 = client1.get_waveforms(network=net1, station=sta1, location=loc1, channel=com1,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    inv1 = read_inventory("station_BVtest.xml")    
elif sta1 == "CLRX":
    print("# read")
    st1 = client1.get_waveforms(network=net1, station=sta1, location=loc1, channel=com1,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    inv1 = read_inventory("station_BVtest.xml")    
elif sta1 == "WROX":
    print("# read")
    st1 = client1.get_waveforms(network=net1, station=sta1, location=loc1, channel=com1,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    inv1 = read_inventory("station_BVtest.xml")
elif sta1 == "GRPX":
    print("# read")
    st1 = client1.get_waveforms(network=net1, station=sta1, location=loc1, channel=com1,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    inv1 = read_inventory("station_BVtest.xml")
elif sta1 == "DRDX":
    print("# read")
    st1 = client1.get_waveforms(network=net1, station=sta1, location=loc1, channel=com1,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    inv1 = read_inventory("station_BVtest.xml")
elif sta1 == "SANX":
    print("# read")
    st1 = client1.get_waveforms(network=net1, station=sta1, location=loc1, channel=com1,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    inv1 = read_inventory("station_BVtest.xml")
elif sta1 == "GUMX":
    print("# read")
    st1 = client1.get_waveforms(network=net1, station=sta1, location=loc1, channel=com1,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    inv1 = read_inventory("station_BVtest.xml")
elif sta1 == "MILX":
    print("# read")
    st1 = client1.get_waveforms(network=net1, station=sta1, location=loc1, channel=com1,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    inv1 = read_inventory("station_BVtest.xml")    
elif sta1 == "YBX":
    print("# read")
    st1 = client1.get_waveforms(network=net1, station=sta1, location=loc1, channel=com1,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    print("# read resp")
    inv1 = read_inventory("http://ncedc.org/ftp/outgoing/taira/WQC/BK.YBH.xml")
elif sta1 == "MHX":
    print("# read")
    st1 = client1.get_waveforms(network=net1, station=sta1, location=loc1, channel=com1,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    print("# read resp")
    inv1 = read_inventory("http://ncedc.org/ftp/outgoing/taira/WQC/BK.MHC.xml")
elif sta1 == "JEPS":
    print("# read")
    st1 = client1.get_waveforms(network=net1, station=sta1, location=loc1, channel=com1,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    inv1 = read_inventory("station_BVtest.xml")
elif sta1 == "CMBX":
    print("# read")
    st1 = client1.get_waveforms(network=net1, station=sta1, location=loc1, channel=com1,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    inv1 = read_inventory("station_BVtest.xml")
    inv1 = read_inventory("http://ncedc.org/ftp/outgoing/taira/WQC/resp/BK.CMB.xml")
elif sta1 == "TBRX":
    print("# starttime = ", starttime)
    print("# endtime = ", endtime)
    print("# sta1 = ", sta1)
    print("# net1 = ", net1)
    st1 = client1.get_waveforms(network=net1, station=sta1, location=loc1, channel=com1,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    inv1 = client1.get_stations(network=net1, station=sta1, location=loc1, channel=com1,
                     starttime=starttime, endtime=endtime, 
                     level="response")
else:
    print("# starttime = ", starttime)
    print("# endtime = ", endtime)
    print("# sta1 = ", sta1)
    print("# net1 = ", net1)
    print("# com1 = ", com1)
    print("# loc1 = ", loc1)
    st1 = client1.get_waveforms(network=net1, station=sta1, location=loc1, channel=com1,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    print("# st1 done")
    print("# move to inv1")
    inv1 = client1.get_stations(network=net1, station=sta1, location=loc1, channel=com1,
                     starttime=starttime, endtime=endtime, 
                     level="response")

SISOPT1 = config["SISOPT1"]
if SISOPT1:
    sis_dir = "https://files.anss-sis.scsn.org/preview"

    SISTestOPT1 = 0
    sis_key = "prod"
    if SISTestOPT1:
        sis_key = "test"

    inv_fi1 = sis_dir+"/preview_"+sis_key+"_"+net1+"_"+sta1+".xml"
    print("# SIS inv_fi1 = ", inv_fi1)
    inv1 = read_inventory(inv_fi1)
    
if plotOPT:
    _plot = st1.plot(size=(1000,600))



if bv2OPT == 1:
    oldOPT = 0
    if oldOPT:
        mseedid = "2022.261.153101"
        chname = "LN"
        chname = "HN"
        st2 = read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta2+".BK."+chname+"E.00.D."+mseedid)
        st2 += read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta2+".BK."+chname+"N.00.D."+mseedid)
        st2 += read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta2+".BK."+chname+"Z.00.D."+mseedid)
        st2.trim(starttime, endtime)  
    else:
        st2 = client2.get_waveforms_testdata(network=net2, station=sta2, location=loc2, channel=com2,
                         starttime=starttime, endtime=endtime, 
                         attach_response=False)
    
    inv2 = read_inventory("station_BVtest.xml")
    if sta2 == "MBARI":
        inv2 = read_inventory("station_MBARI.xml")
    elif sta2 == "BK90x" or sta2 == "BK91x":
        print("BK90")
        mseedid = "2023.355.230102"
        st2 = read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta2+".BK.HNE.00.D."+mseedid)
        st2 += read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta2+".BK.HNN.00.D."+mseedid)
        st2 += read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta2+".BK.HNZ.00.D."+mseedid)
        st2.trim(starttime, endtime)  
        inv2 = read_inventory("station_BVtest.xml")
elif sta2 == "CCORX":
    print("# read")
    st2 = client1.get_waveforms(network=net2, station=sta2, location=loc2, channel=com2,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    sis_dir = "https://files.anss-sis.scsn.org/preview"
    sis_key = "prod"
    inv_fi = sis_dir+"/preview_"+sis_key+"_"+net2+"_"+sta2+".xml"
    inv2 = read_inventory(inv_fi)
elif sta2 == "BKSx":
    print("# sta2 = ", sta2)
    print("# net2 = ", net2)
    print("# com2 = ", com2)
    print("# loc2 = ", loc2)

    mseedid = "2024.164.022556"
    st2 =  read("/work/dc6/ftp/outgoing/taira/WQC/bktest_data/"+sta2+".BK.HHU.00.D."+mseedid+".sac")
    st2 += read("/work/dc6/ftp/outgoing/taira/WQC/bktest_data/"+sta2+".BK.HHV.00.D."+mseedid+".sac")
    st2 += read("/work/dc6/ftp/outgoing/taira/WQC/bktest_data/"+sta2+".BK.HHW.00.D."+mseedid+".sac")

    inv2 = client2.get_stations(network=net2, station=sta2, location=loc2, channel=com2,
                     starttime=starttime, endtime=endtime, 
                     level="response")
elif sta2 == "MBARIX":
    mseedid = "2022.351.113942"
    st2 = read("/Users/taira/work/python_work/MBARI/mseed/"+sta2+".BK.HNE.00.D."+mseedid)
    st2 += read("/Users/taira/work/python_work/MBARI/mseed/"+sta2+".BK.HNN.00.D."+mseedid)
    st2 += read("/Users/taira/work/python_work/MBARI/mseed/"+sta2+".BK.HNZ.00.D."+mseedid)
    #st = read("/Users/taira/work/python_work/MBARI/mseed/MBARI.BK.HHZ.00.D.2022.315.104845")

    st2.trim(starttime, endtime)  

    inv2 = read_inventory("station_MBARI.xml")
elif sta2 == "MHX":
    print("# read")
    st2 = client2.get_waveforms(network=net2, station=sta2, location=loc2, channel=com2,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)

    print("# read resp")
    inv2 = read_inventory("http://ncedc.org/ftp/outgoing/taira/WQC/BK.MHC.xml")
elif sta2 == "PABX":
    mseedid = "2021.225.054555"
    st2 = read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta2+".BK.HNE.00.D."+mseedid)
    st2 += read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta2+".BK.HNN.00.D."+mseedid)
    st2 += read("http://ncedc.org/ftp/outgoing/taira/WQC/bktest_data/"+sta2+".BK.HNZ.00.D."+mseedid)
    st2.trim(starttime, endtime)  

    inv2 = read_inventory("station_BVtest.xml")
elif sta2 == "MERX":
    st2 = read("http://ncedc.org/ftp/outgoing/taira/MERC.BK.HNE.00.D.2020.275.003127")
    st2 += read("http://ncedc.org/ftp/outgoing/taira/MERC.BK.HNN.00.D.2020.275.003127")
    st2 += read("http://ncedc.org/ftp/outgoing/taira/MERC.BK.HNZ.00.D.2020.275.003127")
    inv2 = read_inventory("station_BVtest.xml")
elif sta2 == "TOLX":
    st2 = read("http://ncedc.org/ftp/outgoing/taira/TOLH.BK.HNE.00.D.2020.276.123247")
    st2 += read("http://ncedc.org/ftp/outgoing/taira/TOLH.BK.HNN.00.D.2020.276.123247")
    st2 += read("http://ncedc.org/ftp/outgoing/taira/TOLH.BK.HNZ.00.D.2020.276.123247")
    inv2 = read_inventory("station_BVtest.xml")
elif sta2 == "BCCX":
    st2 = read("http://ncedc.org/ftp/outgoing/taira/BCCR.BK.HNE.00.D.2020.284.012411")
    st2 += read("http://ncedc.org/ftp/outgoing/taira/BCCR.BK.HNN.00.D.2020.284.012411")
    st2 += read("http://ncedc.org/ftp/outgoing/taira/BCCR.BK.HNZ.00.D.2020.284.012411")   
    inv2 = read_inventory("station_BVtest.xml")
elif sta2 == "PETY":
    print("# read")
    st2 = client1.get_waveforms(network=net2, station=sta2, location=loc2, channel=com2,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    #inv2 = read_inventory("station_BVtest.xml")
    sis_dir = "https://files.anss-sis.scsn.org/preview"
    sis_key = "prod"
    inv_fi = sis_dir+"/preview_"+sis_key+"_"+net2+"_"+sta2+".xml"
    inv2 = read_inventory(inv_fi)
elif sta2 == "JEPS":
    print("# read")
    st2 = client1.get_waveforms(network=net2, station=sta2, location=loc2, channel=com2,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    inv2 = read_inventory("station_BVtest.xml")
elif sta2 == "MZTX":
    print("# read")
    st2 = client1.get_waveforms(network=net2, station=sta2, location=loc2, channel=com2,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    inv2 = read_inventory("station_BVtest.xml")
elif sta2 == "PORX":
    print("# read")
    st2 = client1.get_waveforms(network=net2, station=sta2, location=loc2, channel=com2,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    inv2 = read_inventory("station_BVtest.xml")
elif sta2 == "CLRX":
    print("# read")
    st2 = client1.get_waveforms(network=net2, station=sta2, location=loc2, channel=com2,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    inv2 = read_inventory("station_BVtest.xml")
elif sta2 == "GUMX":
    print("# read")
    st2 = client1.get_waveforms(network=net2, station=sta2, location=loc2, channel=com2,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    inv2 = read_inventory("station_BVtest.xml")
elif sta2 == "MNRX" and com2 == "HN?":
    print("# read")
    st2 = client2.get_waveforms(network=net2, station=sta2, location=loc2, channel=com2,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    inv2 = read_inventory("station_BVtest.xml")
elif sta2 == "WROX":
    print("# read")
    st2 = client2.get_waveforms(network=net2, station=sta2, location=loc2, channel=com2,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    inv2 = read_inventory("station_BVtest.xml")
elif sta2 == "GRPX":
    print("# read")
    st2 = client2.get_waveforms(network=net2, station=sta2, location=loc2, channel=com2,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    inv2 = read_inventory("station_BVtest.xml")
elif sta2 == "CMBX":
    print("# read")
    st2 = client2.get_waveforms(network=net2, station=sta2, location=loc2, channel=com2,
                    starttime=starttime, endtime=endtime, 
                     attach_response=True)
    inv2 = read_inventory("station_BVtest.xml")
elif sta2 == "TBRX":
    print("# sta2 = ", sta2)
    print("# net2 = ", net2)
    print("# com2 = ", com2)
    print("# loc2 = ", loc2)
    st2 = client2.get_waveforms(network=net2, station=sta2, location=loc2, channel=com2,
                     starttime=starttime, endtime=endtime, 
                     attach_response=True)
    # for station locations
    inv2 = client2.get_stations(network=net2, station=sta2, location=loc2, channel=com2,
                     starttime=starttime, endtime=endtime, 
                     level="response")
else:
    print("# sta2 = ", sta2)
    print("# net2 = ", net2)
    print("# com2 = ", com2)
    print("# loc2 = ", loc2)

    st2 = client2.get_waveforms(network=net2, station=sta2, location=loc2, channel=com2,
                     starttime=starttime, endtime=endtime, 
                     attach_response=True)
    # for station locations
    inv2 = client2.get_stations(network=net2, station=sta2, location=loc2, channel=com2,
                     starttime=starttime, endtime=endtime, 
                     level="response")
SISOPT2 = config["SISOPT2"]
if SISOPT2:
    sis_dir = "https://files.anss-sis.scsn.org/preview"

    SISTestOPT2 = 0
    sis_key = "prod"
    if SISTestOPT2:
        sis_key = "test"
    inv_fi2 = sis_dir+"/preview_"+sis_key+"_"+net2+"_"+sta2+".xml"
    print("# SIS inv_fi2 = ", inv_fi2)

    inv2 = read_inventory(inv_fi2)
    
if plotOPT:
    _plot = st2.plot(size=(1000,400))

st1_merge = st1.copy().merge()
print(st1_merge)
st1 = st1_merge.copy()

st2_merge = st2.copy().merge()
print(st2_merge)
st2 = st2_merge.copy()

if plotOPT:
    st_tmp = st1.copy() + st2.copy()
    _plot = st_tmp.copy().detrend("linear").taper(0.001).plot(equal_scale=False, 
                                                              size=(1200,600))
    
### Removing Instrument Response
# use st_remove_resp function uses obspy remove_response to remove instrument response.
# Example will provide ground velocity data (m/s)

st1_zne_raw = get_zne_data (st1.copy(), inv1, starttime)
st1_zne_raw.write("test_zne_output.sac", formate="SAC")

oldwayOPT = 0

if oldwayOPT:
    if bv1OPT == 1 or  \
        sta1 == "BK17" or  \
        sta1 == "MBARI" or sta1 == "WLKR" or sta1 == "JEPS" or sta1 == "SKGX" or sta1 == "CCORX":    

        print("# use inv1")
        st1 = st_remove_resp_inv(st1, deciopt_1, decifactor_1, pre_filt, "VEL", inv1)

    else:
        st1 = st_remove_resp(st1, deciopt_1, decifactor_1, pre_filt, "VEL")
        #st1 = st_remove_resp(st1, deciopt_1, decifactor_1, pre_filt_tremor, "VEL")

        
st1 = st_remove_resp_inv(st1, deciopt_1, decifactor_1, pre_filt, "VEL", inv1)

if plotOPT:
    _plot = st1.plot(size=(1000,600))

if oldwayOPT:
    if bv2OPT == 1 or  \
        sta2 == "CCORX" or sta2 == "WLKR" or  \
        sta2 == "MBARI":    

        print("# use inv2")
        st2 = st_remove_resp_inv(st2, deciopt_2, decifactor_2, pre_filt, "VEL", inv2)

    else:
        st2 = st_remove_resp(st2, deciopt_2, decifactor_2, pre_filt, "VEL")

st2 = st_remove_resp_inv(st2, deciopt_2, decifactor_2, pre_filt, "VEL", inv2)

if plotOPT:
    _plot = st2.plot(size=(1000,600))

st_tmp = st1.copy() + st2.copy()

### Rotating seismic data into ZNE coordinate
# use get_zne_data. This will provide ZNE coordinate data. 

st1_zne = st1.copy()

try:
    st1_zne = get_zne_data (st1_zne, inv1, starttime)
except:
    print("error! st1 zne")
    st1_zne = st1.copy()

if plotOPT:
    _plot = st1_zne.plot(size=(1000,600))

st2_zne = st2.copy()

try:
    st2_zne = get_zne_data (st2_zne, inv2, starttime)
except:
    print("error! st2 zne")
    st2_zne = st2.copy()

if plotOPT:   
    _plot = st2_zne.plot(size=(1000,600))

# think about gappy data case. here we assume no gap with three component data
rotateOPT_1 = 0
if len(st1) == 3:
    roateOPT_1 = 1
rotateOPT_2 = 0
if len(st2) == 3:
    roateOPT_2 = 1

## Rotating seismic data into ZRT coordinate 
# get_baz to esimate the back azimuth and then use obspy 
# rotate to covert ZNE data into ZRT data

baz1 = get_baz(st1, inv1, evla, evlo)
st1_zrt = st1_zne.copy()

if rotateOPT_1:
    st1_zrt.rotate(method='NE->RT',back_azimuth=baz1[2])
    
if plotOPT:
    _plot = st1_zrt.plot(size=(1000,600))

baz2 = get_baz(st2, inv2, evla, evlo)
st2_zrt = st2_zne.copy()

if rotateOPT_2:
    st2_zrt.rotate(method='NE->RT',back_azimuth=baz2[2])
if plotOPT:
    _plot = st2_zrt.plot(size=(1000,600))

### Combining all streams
if plotOPT:
    _plot = st1_zrt.plot(size=(1000,600))

if rotateOPT_1 == 1 and rotateOPT_2 == 1:
    # def
    st_all = st2_zrt.copy() + st2_zne.select(component="E")  + st2_zne.select(component="N")  + st1_zrt.copy() + st1_zne.select(component="E") + st1_zne.select(component="N")
else:
    st_all = st1_zrt.copy() + st2_zrt.copy()

print(st_all)
if plotOPT:
    _plot = st_all.plot(size=(1000,600))

### Computing arrival time of Rayleigh wave
# event ditance / Rvel (This example uses 4.2 km/s)

Rwave_arrival = baz2[0]/1000.0/Rvel

if sta1 == "TRINX":
    Rwave_arrival  = 6
    
print("# Rwave_arrival = ",Rwave_arrival)

### Trim seismic data
#tw_start = origin_time + Rwave_arrival + tw_pre_tw - 60
#tw_end = tw_start + tw_trim - 60 -60 -30

tw_start = origin_time + Rwave_arrival + tw_pre_tw 
tw_end = tw_start + tw_trim

print(tw_start, tw_end)

notrimOPT = 0

if notrimOPT:
    print("# no trime")
else:
    if manualOPT:
        print("# no trime")
    else:
        st_all.trim(tw_start, tw_end)  

st_all.detrend('linear') # remove linear trend
st_all.detrend("demean") # demean
st_all.taper(0.05)
if plotOPT:
    _plot = st_all.plot(size=(1200,800))

if plotOPT:
    st_tmp =  st2_zne.select(component="Z")  + st2_zne.select(component="E")  + st2_zne.select(component="N")  + st1_zne.select(component="Z") + st1_zne.select(component="E") + st1_zne.select(component="N")
    
    
    if notrimOPT:
        print("# no trime")
    else:
        if manualOPT:
            print("# no trime")
        else:
            st_tmp.trim(tw_start, tw_end)  

    st_tmp.detrend('linear') # remove linear trend
    st_tmp.detrend("demean") # demean
    st_tmp.taper(0.05)
    _plot = st_tmp.plot(size=(1500,800))

print(st_all)

### Waveform similarity
select_channel = ["Z", "N", "E"]

slideOPT = 1

for component in select_channel:
    st_select = st_all.select(component=component) 

    try:
        waveform_comparison(st_select, 1, slideOPT, logOPT, manualOPT, fl, fh)

    except Exception as e:
        print("# error {0}".format(e))

