from obspy import read, read_inventory
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal import PPSD
from obspy.imaging.cm import pqlx
from obspy.core import Stream
from obspy import read, read_inventory

import obspy as ob
print("# obspy version = ",ob.__version__)

from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from obspy.clients.nrl import NRL

# color code
from obspy.imaging.cm import pqlx
from obspy.imaging.cm import viridis_white_r

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 15,15

import re
import yaml
import os
import numpy as np
import argparse

# font size
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

#Import parameters from config

# Get file directory
file_path = os.path.realpath(__file__)
directory = os.path.dirname(file_path)

flags = {"config": ("-c", str),
         "start_day": ("-st", str), "end_day": ("-ed", str), "ppsd_length": ("-psd", str), "net": ("-n", str), "loc_loop": ("-loc", str), "com_loop": ("-com", str), "sta_loop": ("-sta", str),
         "pngOPT": ("-png", bool), "epsOPT": ("-eps", bool), "debugOPT": ("-debug", bool), "plotOPT": ("-plot", bool), "waveformplotOPT": ("-waveformplot", bool), "showEQOPT": ("-eq", bool),
         "transformOPT": ("-transform", bool), "savenpzOPT": ("-savenpz", bool), "loopOPT": ("-loop", bool), "SISOPT": ("-sis", bool), "localdataOPT": ("-local", bool)}

parser = argparse.ArgumentParser(description="Flags for waveform_similarity.py")

for key in flags.keys():
    var = key
    flag, vartype = flags[key]
    parser.add_argument(flag, f"--{var}", metavar='', type=vartype)

args, unknown = parser.parse_known_args()
args_dict = args.__dict__

config_name = "cal_PSD.yml"

if args_dict["config"]:
    config_name = args_dict["config"]

with open(f"{directory}/{config_name}") as c:
    config = yaml.safe_load(c)

for var in flags.keys():
    if args_dict[var]:
        if var == "ppsd_length":
            config[var] = eval(args_dict[var])
        elif var == "loc_loop" or var == "com_loop" or var == "sta_loop":
            config[var] = args_dict[var].split(",")
        else:
            config[var] = args_dict[var]

start_day = config["start_day"]
end_day = config["end_day"]
starttime = UTCDateTime(start_day)
endtime = UTCDateTime(end_day)
print(endtime)

psd_data_length = eval(config["psd_data_length"])

min_db = config["min_db"]
max_db = config["max_db"] # max db
ddb = config["ddb"] # 1 db increment

period_smoothing_width_octaves = config["period_smoothing_width_octaves"]
period_step_octaves = config["period_step_octaves"]

period_low = config["period_low"]
period_max = config["period_max"]

cmap = viridis_white_r
print(cmap)

ppsd_length = eval(config["ppsd_length"])
print("# ppsd_length = ", ppsd_length)

net = config["net"]
loc_loop = config["loc_loop"]
com_loop = config["com_loop"]
sta_loop = config["sta_loop"]

pngOPT = config["pngOPT"]
epsOPT = config["epsOPT"]
debugOPT = config["debugOPT"]
plotOPT = config["plotOPT"]
waveformplotOPT = config["waveformplotOPT"]
showEQOPT = config["showEQOPT"]
transformOPT = config["transformOPT"]
savenpzOPT = config["savenpzOPT"]
loopOPT = config["loopOPT"]
SISOPT = config["SISOPT"]
localdataOPT = config["localdataOPT"]

for sta in sta_loop:
    print("# sta = ", sta)
    for com in com_loop:
        print("# com = ", com)
        for loc in loc_loop:
            print("# loc = ", loc)

client = Client("https://service.ncedc.org")

def get_seedid(tr):
    seedid=tr.stats.network+"."+tr.stats.station+"."+tr.stats.location+"."+tr.stats.channel
    return seedid

def cal_PSD(sta, net, com, loc, start_day, end_day, pngOPT, epsOPT, debugOPT, client, 
            bvOPT, SISOPT, testHTOPT, localIROPT, waveformplotOPT, showEQOPT, savenpzOPT, transformOPT, localdataOPT):
    if transformOPT:
        client.base_url = 'http://transform.geo.berkeley.edu:8080'

    # image files    
    png_fi = "./psd_plot/"+net+"."+sta+"."+com+"."+loc+"."+start_day+"."+end_day+".png"
    eps_fi = "./psd_plot/"+net+"."+sta+"."+com+"."+loc+"."+start_day+"."+end_day+".eps"
    # PSD result file
    npz_fi = "./psd_plot/"+net+"."+sta+"."+com+"."+loc+"."+start_day+"."+end_day+".npz"
    # image file
    WFplot_fi = "./psd_plot/"+net+"."+sta+"."+com+"."+loc+"."+start_day+"."+end_day+".WF.png"

    if debugOPT:
        print("# bvOPT = ", bvOPT)
        print("# testHTOPT = ", testHTOPT)
        print("# png_fi = ", png_fi)
        print("# eps_fi = ", eps_fi)
        print("# npz_fi = ", npz_fi)
        print("# WFplot_fi = ", WFplot_fi)

    if SISOPT:
        sis_dir = "https://files.anss-sis.scsn.org/preview"

        SISTestOPT = 0
        sis_key = "prod"
        if SISTestOPT:
            sis_key = "test"

        inv_fi = sis_dir+"/preview_"+sis_key+"_"+net+"_"+sta+".xml"
        print("# SIS inv_fi = ", inv_fi)
        
        inv = read_inventory(inv_fi)
        
    else:
        if testHTOPT:
            ir_dir = "http://ncedc.org/ftp/outgoing/taira/WQC"
            inv_fi = ir_dir+"/"+net+"."+sta+".xml"
            inv = read_inventory(inv_fi)

        if localIROPT:
            inv_fi = "station_BVtest.xml"
            
            if sta == "MBARI":
                inv_fi = "station_MBARI.xml"
                
            inv = read_inventory(inv_fi)

        if testHTOPT != 1 and localIROPT != 1:
            inv = client.get_stations(network=net, station=sta, starttime=starttime, endtime=endtime, level="response")

    if debugOPT:
        print("# net  = ", net)
        print("# sta  = ", sta)
        print("# loc  = ", loc)
        print("# com  = ", com)

        print(client)
        
    if loopOPT == 1:

        starttime_cut = starttime

        st = Stream()

        while (starttime_cut <= endtime):
            endtime_cut = starttime_cut+psd_data_length
            print("# startime_cut = ", starttime_cut)
            print("#  endtime_cut = ", endtime_cut)
            if bvOPT:
                st += client.get_waveforms_testdata(net,  sta, loc, com,
                                                    starttime_cut,  endtime_cut, debugOPT=0)        
            else:
                st += client.get_waveforms(net,  sta, loc, com, starttime_cut,  endtime_cut)

            if debugOPT:
                print(st)

            starttime_cut += 60*60*24
            print("# new starttime_cut = ", starttime_cut)
    else:
        if localdataOPT:
            localdata_dir = "/work/ftp/outgoing/taira/WQC/bktest_data"
            localdata_time = "2024.163.000000"
            if(sta == "BK02"):
                localdata_dir = "/home/bsl/taira/hydro/Instrumentation/python_work"
                localdata_time = "2024.170.000000"
            localdata_fi= localdata_dir+"/"+sta+"."+net+"."+com+"."+loc+".D."+localdata_time
            st = read(localdata_fi) 
        else:
            if bvOPT:
                if debugOPT:
                    print("# get data start")
                st = client.get_waveforms_testdata(net,  sta, loc, com, starttime,  endtime, debugOPT=0)        
                if debugOPT:
                    print("# get data done")
            else:
                st = client.get_waveforms(net,  sta, loc, com, starttime,  endtime)

    print(st)

    if plotOPT:

        fl = 5
        fh = 10

        fl = 0.1
        fh = 1.0
        fl = 0.01
        fh = 0.1
        st_fil = st.copy().taper(0.05).filter("bandpass", freqmin=fl, freqmax=fh)

        _plot = st_fil.plot(size=(1000,300))

        #_plot = st_fil.plot_rev(size=(1000,300), fix_scale=True, fix_ymin=-20, fix_ymax=20)
        _plot = st_fil.plot(size=(1000,300), fix_scale=True, fix_ymin=-20, fix_ymax=20)

        for tr in st:
            _plot = tr.plot(size=(1000,200))
            
        for tr in st:
            tr.plot()

    if debugOPT:            
        vars(inv[0][0][0])
        inv[0][0][0].latitude
        st_lat = inv[0][0][0].latitude
        st_lon = inv[0][0][0].longitude
        st_ele = inv[0][0][0].elevation
        st_dep = inv[0][0][0].depth
        print("# st_lat = ", st_lat, " st_lon = ", st_lon)           

    if loopOPT == 0:
        if debugOPT:
            print("# merge start")

        st_merge = st.merge()
        print(st_merge)


        if waveformplotOPT:
            _plot = st_merge.plot()
            _plot = st_merge.plot(outfile=WFplot_fi)

        # def
        if debugOPT:
            print("# PPSD start")

        ppsd = PPSD(stats=st_merge[0].stats, metadata=inv, db_bins=(min_db, max_db, ddb), ppsd_length=ppsd_length, 
                    period_smoothing_width_octaves=period_smoothing_width_octaves, period_step_octaves=period_step_octaves,
                    )

        if debugOPT:
            print("# PPSD end")

        # inv_36dB_9K
        #ppsd = PPSD(stats=st_merge[0].stats, metadata=inv_36dB_9K, db_bins=(min_db, max_db, ddb), ppsd_length=ppsd_length, 
        #            period_smoothing_width_octaves=period_smoothing_width_octaves, period_step_octaves=period_step_octaves,
        #            )
        #ppsd = PPSD(stats=st_merge[1].stats, metadata=inv, db_bins=(min_db, max_db, ddb), ppsd_length=ppsd_length)

        # use inv1
        #ppsd = PPSD(stats=st_merge[0].stats, metadata=inv1, db_bins=(min_db, max_db, ddb), ppsd_length=ppsd_length)
        if debugOPT:
            print("# PPSD merge start")
            
        ppsd.add(st_merge[0])
        #ppsd.add(st_merge[1])
        if debugOPT:
            print("# PPSD merge end")
    
        print(ppsd)

    if loopOPT == 1:
        st_merge = st.merge()

        # def
        ppsd_loop = PPSD(stats=st_merge[0].stats, metadata=inv, db_bins=(min_db, max_db, ddb), ppsd_length=ppsd_length, 
                    period_smoothing_width_octaves=period_smoothing_width_octaves, period_step_octaves=period_step_octaves,
                    )

        print("# st_merge[0].stats.starttime) = ", st_merge[0].stats.starttime)
        print("# st_merge[0].stats.endtine) = ", st_merge[0].stats.endtime)

        starttime_cut = (st_merge[0].stats.starttime)

        if starttime_cut <= st_merge[0].stats.endtime:
            print("# test")

        while (starttime_cut <= st_merge[0].stats.endtime):
            endtime_cut = starttime_cut+psd_data_length
            print("# startime_cut = ", starttime_cut)
            print("#  endtime_cut = ", endtime_cut)
            st_select = st_merge.copy().trim(starttime=starttime_cut, endtime=endtime_cut)
            print(st_select)
            ppsd_loop.add(st_select)

            starttime_cut += 60*60*24
            print("# new starttime_cut = ", starttime_cut)

    if loopOPT == 1:
        ppsd_loop.plot(cmap=cmap, period_lim=(period_low, period_max), show_mean = True, )
    else:
        if debugOPT:
            print("# ppsd.plot start")
        ppsd.plot(cmap=cmap, period_lim=(period_low, period_max), show_mean = True, )
        if debugOPT:
            print("# ppsd.plot end")
    
        if showEQOPT:
            ppsd.plot(cmap=cmap, period_lim=(period_low, period_max), show_mean = True, show_earthquakes=(0, 5, 0, 10) )
        
    if pngOPT:
        if loopOPT == 1:
            ppsd_loop.plot(png_fi, cmap=cmap, period_lim=(period_low, period_max), show_mean = True, )

        else:
            if debugOPT:
                print("# ppsd.plot png start")
    
            ppsd.plot(png_fi, cmap=cmap, period_lim=(period_low, period_max), show_mean = True, )
        #ppsd.plot(png_fi, cmap=cmap, period_lim=(period_low, period_max), show_mean = True,  show_earthquakes=(-1, 2, 0, 10) ) 
            #ppsd.plot_rev(png_plot_rev_fi,cmap=cmap, period_lim=(period_low, period_max), show_percentiles=True, percentiles=[10])

            if debugOPT:
                print("# ppsd.plot png end")

    if epsOPT:
        if loopOPT == 1:
            ppsd_loop.plot(eps_fi, cmap=cmap, period_lim=(period_low, period_max), show_mean = True, )
        else:
            if debugOPT:
                print("# ppsd.plot eps start")
            ppsd.plot(eps_fi, cmap=cmap, period_lim=(period_low, period_max), show_mean = True, )
            if debugOPT:
                print("# ppsd.plot eps end")
    if savenpzOPT:
        if loopOPT == 1:                
            ppsd_loop.save_npz(npz_fi)
        else:
            ppsd.save_npz(npz_fi)

    return inv

for sta in sta_loop:
    print("# sta = ", sta)
    bvOPT = 0
    if re.findall('BK+[0-9]', sta) != []:
        #print("test")
        bvOPT = 1
                
    if sta == "MBARI":
        bvOPT = 1

        print("# bvOPT = ", bvOPT)

    
    if sta == "BK90x":
        localdataOPT = 1
    if sta == "BK91x":
        localdataOPT = 1
    if sta == "BK92x":
        localdataOPT = 1
    if sta == "BK93x":
        localdataOPT = 1
    if sta == "BK80x":
        localdataOPT = 1
    if sta == "BK02x":
        localdataOPT = 1

   
    if bvOPT:
        # BK02
        testHTOPT = 0 # ir_dir = "http://ncedc.org/ftp/outgoing/taira/WQC"
        localIROPT = 1  # station BV
        transformOPT = 1 # has to be
        SISOPT = 0 # has to be zero

    else:    
        testHTOPT = 0 # ir_dir = "http://ncedc.org/ftp/outgoing/taira/WQC"
        localIROPT = 0  # station BV



    for com in com_loop:
        print("# com = ", com)
        for loc in loc_loop:
            print("# loc = ", loc)
        

            try:
                inv = cal_PSD(sta, net, com, loc, start_day, end_day, pngOPT, epsOPT, debugOPT, client, bvOPT, SISOPT, testHTOPT, localIROPT, waveformplotOPT, showEQOPT, savenpzOPT, transformOPT, localdataOPT)

                #print("no rev now")
                #cal_PSD_rev(sta, net, com, loc, start_day, end_day, pngOPT, epsOPT, debugOPT, client, bvOPT, SISOPT, testHTOPT, localIROPT, waveformplotOPT, showEQOPT)
                
            except Exception as e:
                print(e)
                print("# no data?")