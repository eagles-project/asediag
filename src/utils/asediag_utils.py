import logging
import numpy as np
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT

import warnings

# Mute all warnings
warnings.filterwarnings("ignore")

# Class to generate color bar ranges
class gen_colbar_range(object):
    
    def __init__(self,**kwargs):
        self.v1 = kwargs.get('v1',None)
        self.v2 = kwargs.get('v2',None)
        self.diff = kwargs.get('diff',None)
        self.factor = kwargs.get('factor',None)
    
    def hmap(self):
        if self.factor != None:
            factor = self.factor
            mstd1h = (np.mean(self.v1.values)+factor*np.std(self.v1.values))
            mstd1l = (np.mean(self.v1.values)-factor*np.std(self.v1.values))
            v1 = self.v1[(self.v1<=mstd1h) & (self.v1>=mstd1l)]
            mstd1h = (np.mean(self.v2.values)+factor*np.std(self.v2.values))
            mstd1l = (np.mean(self.v2.values)-factor*np.std(self.v2.values))
            v2 = self.v2[(self.v2<=mstd1h) & (self.v2>=mstd1l)]
            aagg = (np.max(v1.values)+np.max(v2.values))/2
        else:
            aagg = (np.max(self.v1.values)+np.max(self.v2.values))/2
        
        aagg = np.log10(aagg)
        expo = np.floor(aagg)
        bbgg = aagg - expo
        if 10**(bbgg)<2.:
            s1 = [5*10**(expo-4),1*10**(expo-3),2*10**(expo-3), \
                    5*10**(expo-3),1*10**(expo-2),2*10**(expo-2), \
                    5*10**(expo-2),1*10**(expo-1),2*10**(expo-1), \
                    5*10**(expo-1),10**expo,      2.*10**expo]
        elif 10**(bbgg)<5.:
            s1 = [1*10**(expo-3),2*10**(expo-3),5*10**(expo-3), \
                    1*10**(expo-2),2*10**(expo-2),5*10**(expo-2), \
                    1*10**(expo-1),2*10**(expo-1),5*10**(expo-1), \
                    10**expo,      2.*10**expo,   5.*10**expo]
        else:
            s1 = [2*10**(expo-3),5*10**(expo-3),1*10**(expo-2), \
                    2*10**(expo-2),5*10**(expo-2),1*10**(expo-1), \
                    2*10**(expo-1),5*10**(expo-1),10**expo,       \
                    2.*10**expo,   5.*10**expo,   10**(expo+1)]
        return s1
    
    def hdiff(self):
        aagg = np.max(abs(self.diff).values)
        aagg = np.log10(aagg)
        expo = np.ceil(aagg)
        s1 = np.array([-100,-70,-50,-20,-10,-5,-2,-1,1,2,5,10,20,50,70,100])*(10**(expo)/1e3)
        return list(s1)
    
    def vmap(self):
        s1=[0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000]
        aagg=(np.max(self.v1).values+np.max(self.v2).values)/2
        if aagg == 0:
            s1 = -1*np.array(s1[::-1])
            aagg = 0.25*(abs(np.max(self.v1).values)+abs(np.min(self.v1).values))/2
        aagg=np.log10(aagg)
        s1=np.array(s1)*(10**(np.round(aagg-2.7)))
        return list(s1)   
 
    def vdiff(self):
        s2=[-100,-50.,-20,-10,-5,-2,2,5,10,20,50,100]
        if (abs(np.max(self.v1).values)/abs(np.max(self.diff).values))<10:
            aagg=0.25*0.1*(abs(np.max(self.diff).values)+abs(np.min(self.diff).values))/2
        else:                           
            aagg=0.25*(abs(np.max(self.diff).values)+abs(np.min(self.diff).values))/2
        aagg=np.log10(aagg)
        s1 = np.array(s2)*(10**(np.round(aagg-1.7)))*10
        return list(s1)

# Function to get vertical integral of data
def get_vertint(vdata,ha,p0,hb,ps,grav,fact):
    ## calc. dp
    delp = 0*vdata
    p = ha*p0+hb*ps
    if 'ncol' in p.dims:
        p = p.transpose('ilev','ncol')
    else:
        p = p.transpose('ilev','lat','lon')
    delp = p[1:,:].values-p[:-1,:].values
    delp = delp + 0*vdata
    ## unit conversion and vertical integration
    vdata = vdata*(delp/grav) # p/g = Pa/ms^-2 = Nm^-2/ms^-2 = Kg.ms^-2/m^2/ms^-2
    vdata = vdata*fact
    vdata = vdata.sum('lev')
    return vdata
    
# Function to execute shell commands
def exec_shell(cmd):
    '''func to execute shell commands'''
    cmd_split = cmd.split(' ')
    p = Popen(cmd_split, stdout=PIPE, stdin=PIPE, stderr=STDOUT, universal_newlines=True)
    op, _ = p.communicate()
    logger = logging.getLogger('log.asediag')
    logger.info('\n[cmd]: ' + cmd+ '\n')

# Function to set up output directory
def setup_output_directory(out_directory, case1, case2, region, child = ''):
    path = Path(out_directory) / f'{case2}_minus_{case1}_{region}' / f'{child}'
    if path.exists():
        logging.info(f'Output directory already exists: {path}')
    else:
        path.mkdir(parents=True)
        logging.info(f'Selected output directory: {path}')
        logging.info('All shell scripts and log files will be stored here.')
    return path

# Function to get the absolute path of a directory
def get_dir_path(path):
    if (path == ''):
        p=Path('.')
        dir_path = p.absolute()
    else:
        dir_path = Path(path)
    return dir_path

# Function to round numbers
def rounding(n):
    if (type(n)==str) or (np.isnan(n)) or (np.isinf(n)):
        return str('-')
    elif ((abs(n)>1e-4) and (abs(n)<1e4)):
        try:
            sgn = '-' if n<0 else ''
            num = format(abs(n)-int(abs(n)),'f')
            if int(num[2:6])==0:
                frac = abs(n)-int(abs(n))
                rounded = np.round(frac,4)
                d = str(int(abs(n))+rounded)
                return sgn + d
            else:
                for i,e in enumerate(num[2:]):
                    if e!= '0':
                        if i==0:
                            d = str(int(abs(n))) + (num[1:i+5])
                        else:
                            d = str(int(abs(n))) + (num[1:i+4])
                        return sgn+d
        except:
            return '-'
    else:
        d = '{:.0e}'.format(n)
        d = d.replace('e+00','.0')
        return d

# Function to get latitude and longitude for a region
def get_latlon(reg):
    regions = {'CONUS':'24.74 49.34 -124.78 -66.95',\
              'NA':'15 72 -167 -50',\
              'EUS':'24.74 49.34 -97 -66.95',\
              'ECN':'18 45 90 130',\
              'IND':'6 40 66 98',\
              'CAF':'-5 20 -18 50', \
              'SH_pole':'-90 -60 -180 180',\
              'SH_midlat':'-60 -30 -180 180',\
              'Tropics':'-30 30 -180 180',\
              'NH_midlat':'30 60 -180 180',\
              'NH':'0 90 -180 180',\
              'SH':'-90 0 -180 180',\
              'NH_pole':'60 90 -180 180',\
              'Global':'-90 90 -180 180',\
              'CUS':'31 41 -104 -91',\
              'ENA':'32 46 -33 -21',\
              'NEP':'30 50 -160 -120',\
              'SO':'-60 -40 130 165'}
    lat1 = float(regions[reg].split(' ')[0])
    lat2 = float(regions[reg].split(' ')[1])
    lon1 = float(regions[reg].split(' ')[2])
    lon2 = float(regions[reg].split(' ')[3])
    return lat1,lat2,lon1,lon2

# Function to get local latitude and longitude
def get_local(reg):
    loclatlon = {'SGP':'36.605 -97.485',\
               'ENA':'39.091 -28.026',\
               'NSA':'71.322 -156.615',\
               'TCAP':'42.5 -72',\
               'TWP':'-2.06 147.425'}
    lat1 = float(loclatlon[reg].split(' ')[0])
    lon1 = float(loclatlon[reg].split(' ')[1])
    return lat1,lon1

# Function to parse and get latitude and longitude for multiple local regions
def get_plocal(loc):
    try:
        bb = loc.split(',')
        names = []
        lats = []
        lons = []
        for local in bb:
            ll = local.strip().split(':')
            names.append(ll[0])
            lats.append(float(ll[1]))
            lons.append(float(ll[2]))
    except:
        lats = None
        lons = None
        names = []
    return names,lats,lons

# Function to rearrange variables
def rearrange_variables(variables):
    no_underscore_var = [var for var in variables if '_' not in var]
    a_vars = [var for var in variables if '_a' in var and len(var.split('_')[1]) == 2]
    c_vars = [var for var in variables if '_c' in var and len(var.split('_')[1]) == 2]
    other_vars = [var for var in variables if var not in no_underscore_var and var not in a_vars and var not in c_vars]

    rearranged_list = no_underscore_var + other_vars + a_vars + c_vars
    return rearranged_list

# Function to get unique list
def unique_list(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

# Function to get nearest latitude and longitude
def get_nearestlatlon(lon1,lat1,lon,lat):
    try:
        ind=np.argmin([(lon-lon1)**2+(lat-lat1)**2])
        lat1,lat2,lon1,lon2 = lat[ind],lat[ind],lon[ind],lon[ind]
    except:
        RLLlon = lon.sel(lon=lon1, method='nearest')
        RLLlat = lat.sel(lat=lat1, method='nearest')
        lat1,lat2,lon1,lon2 = RLLlat,RLLlat,RLLlon,RLLlon
    return lat1,lat2,lon1,lon2

# Function to group duplicate indices in a dataframe
def group_duplicate_index(df):
    a = df.values
    sidx = np.lexsort(a.T)
    b = a[sidx]

    m = np.concatenate(([False], (b[1:] == b[:-1]).all(1), [False] ))
    idx = np.flatnonzero(m[1:] != m[:-1])
    I = df.index[sidx].tolist()
    return [I[i:j] for i,j in zip(idx[::2],idx[1::2]+1)]

# Function to get rounded latitude and longitude
def get_rounded_latlon(val1,val2):
    lg1 = val1 % 5
    lg2 = val2 % 5
    if lg1 in [0,5]:
        alg1 = val1
    else:
        alg1 = val1 - lg1 + 5
    if lg2 in [0,5]:
        alg2 = val2
    else:
        alg2 = val2 - lg2 + 5
        
    diff = alg2 - alg1
    
    step = diff // 5
    
    return alg1, alg2, step
