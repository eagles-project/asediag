import logging
import numpy as np
import pandas as pd
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT

def exec_shell(cmd):
    '''func to execute shell commands'''
    cmd_split = cmd.split(' ')
    p = Popen(cmd_split, stdout=PIPE, stdin=PIPE, stderr=STDOUT, universal_newlines=True)
    op, _ = p.communicate()
    logger = logging.getLogger('log.asediag')
    logger.info('\n[cmd]: ' + cmd+ '\n')

def setup_output_directory(out_directory, case1, case2, region, child = ''):
    path = Path(out_directory) / f'{case2}_minus_{case1}_{region}' / f'{child}'
    if path.exists():
        logging.info(f'Output directory already exists: {path}')
    else:
        path.mkdir(parents=True)
        logging.info(f'Selected output directory: {path}')
        logging.info('All shell scripts and log files will be stored here.')
    return path

def get_dir_path(path):
    if (path == ''):
        p=Path('.')
        dir_path = p.absolute()
    else:
        dir_path = Path(path)
    return dir_path

def rounding(n):
    if (type(n)==str) or (np.isnan(n)):
        return str('-')
    elif ((abs(n)>1e-4) and (abs(n)<1e4)):
        try:
            sgn = '-' if n<0 else ''
            num = format(abs(n)-int(abs(n)),'f')
            if int(num[2:])<1:
                d = str((abs(n)))
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
        return '{:.0e}'.format(n)

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

def get_local(reg):
    loclatlon = {'SGP':'36.605 -97.485',\
               'ENA':'39.091 -28.026',\
               'NSA':'71.322 -156.615',\
               'TCAP':'42.5 -72',\
               'TWP':'-2.06 147.425'}
    lat1 = float(loclatlon[reg].split(' ')[0])
    lon1 = float(loclatlon[reg].split(' ')[1])
    return lat1,lon1

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

def get_nearestlatlon(lon1,lat1,lon,lat):
    try:
        ind=np.argmin([(lon-lon1)**2+(lat-lat1)**2])
        lat1,lat2,lon1,lon2 = lat[ind],lat[ind],lon[ind],lon[ind]
    except:
        RLLlon = lon.sel(lon=lon1, method='nearest')
        RLLlat = lat.sel(lat=lat1, method='nearest')
        lat1,lat2,lon1,lon2 = RLLlat,RLLlat,RLLlon,RLLlon
    return lat1,lat2,lon1,lon2

def group_duplicate_index(df):
    a = df.values
    sidx = np.lexsort(a.T)
    b = a[sidx]

    m = np.concatenate(([False], (b[1:] == b[:-1]).all(1), [False] ))
    idx = np.flatnonzero(m[1:] != m[:-1])
    I = df.index[sidx].tolist()
    return [I[i:j] for i,j in zip(idx[::2],idx[1::2]+1)]

def get_html(form,title,extra=[],locations=[],fmt=None,listofvs=None,spfull_vars=None):     
    df = pd.DataFrame()
    if listofvs == None:
        listofvs = ['bc','bc_accum','bc_coarse','bc_pcarbon','bc_a1', 'bc_a3', 'bc_a4', 'bc_c1', 'bc_c3', 'bc_c4',\
                   'so4','so4_accum','so4_aitken','so4_coarse','so4_a1', 'so4_a2', 'so4_a3', 'so4_c1', 'so4_c2', 'so4_c3',\
                   'dst','dst_accum','dst_coarse','dst_a1', 'dst_a3', 'dst_c1', 'dst_c3',\
                   'mom','mom_accum','mom_aitken','mom_coarse','mom_pcarbon','mom_a1', 'mom_a2', 'mom_a3', 'mom_a4', 'mom_c1', 'mom_c2', 'mom_c3', 'mom_c4',\
                   'pom','pom_accum','pom_coarse','pom_pcarbon','pom_a1', 'pom_a3', 'pom_a4', 'pom_c1', 'pom_c3', 'pom_c4',\
                   'ncl','ncl_accum','ncl_aitken','ncl_coarse','ncl_a1', 'ncl_a2', 'ncl_a3', 'ncl_c1', 'ncl_c2', 'ncl_c3',\
                   'soa','soa_accum','soa_aitken','soa_coarse','soa_a1', 'soa_a2', 'soa_a3', 'soa_c1', 'soa_c2', 'soa_c3',\
                   'num','num_accum','num_aitken','num_coarse','num_pcarbon','num_a1', 'num_a2', 'num_a3', 'num_a4', 'num_c1', 'num_c2', 'num_c3', 'num_c4',\
                   'SO2','DMS','H2SO4','SOAG']+['']+extra
    else:
        listofvs = listofvs + extra

    if spfull_vars == None:
        spfull = {
        'bc': '<div style="position:relative;"><a id="BlackCarbon" style="position:absolute; top:-90px;"></a><span style="color: red;"><strong>Black Carbon</strong></span></div>',
        'so4': '<div style="position:relative;"><a id="Sulfate" style="position:absolute; top:-90px;"></a><span style="color: red;"><strong>Sulfate</strong></span></div>',
        'dst': '<div style="position:relative;"><a id="Dust" style="position:absolute; top:-90px;"></a><span style="color: red;"><strong>Dust</strong></span></div>',
        'mom': '<div style="position:relative;"><a id="mom" style="position:absolute; top:-90px;"></a><span style="color: red;"><strong>Marine organic matter</strong></span></div>',
        'pom': '<div style="position:relative;"><a id="pom" style="position:absolute; top:-90px;"></a><span style="color: red;"><strong>Primary organic matter</strong></span></div>',
        'ncl': '<div style="position:relative;"><a id="Seasalt" style="position:absolute; top:-90px;"></a><span style="color: red;"><strong>Sea salt</strong></span></div>',
        'soa': '<div style="position:relative;"><a id="soa" style="position:absolute; top:-90px;"></a><span style="color: red;"><strong>Secondary organic aerosol</strong></span></div>',
        'num': '<div style="position:relative;"><a id="num" style="position:absolute; top:-90px;"></a><span style="color: red;"><strong>Aerosol number</strong></span></div>',
        'SO2': '<div style="position:relative;"><a id="so2" style="position:absolute; top:-90px;"></a><span style="color: red;"><strong>SO2</strong></span></div>',
        'DMS': '<div style="position:relative;"><a id="dms" style="position:absolute; top:-90px;"></a><span style="color: red;"><strong>DMS</strong></span></div>',
        'H2SO4': '<div style="position:relative;"><a id="h2so4" style="position:absolute; top:-90px;"></a><span style="color: red;"><strong>H2SO4</strong></span></div>',
        'SOAG': '<div style="position:relative;"><a id="soag" style="position:absolute; top:-90px;"></a><span style="color: red;"><strong>SOAG</strong></span></div>',
        'accum': '<div style="position:relative;"><a id="ACCUM" style="position:absolute; top:-90px;"></a><span style="color: black;"><strong>accum</strong></span></div>',
        'aitken': '<div style="position:relative;"><a id="AITKEN" style="position:absolute; top:-90px;"></a><span style="color: black;"><strong>aitken</strong></span></div>',
        'coarse': '<div style="position:relative;"><a id="COARSE" style="position:absolute; top:-90px;"></a><span style="color: black;"><strong>coarse</strong></span></div>',
        'pcarbon': '<div style="position:relative;"><a id="PCARBON" style="position:absolute; top:-90px;"></a><span style="color: black;"><strong>pcarbon</strong></span></div>',
        }
    else:
        spfull = {}
        for var in spfull_vars:
            new_var = {var: str('<div style="position:relative;"><a id='+var.replace(' ','')+' style="position:absolute; top:-90px;"></a><span style="color: red;"><strong>'+var+'</strong></span></div>')}
            spfull.update(new_var)

            
    
    tmp = ' '*16
    allKeys = list(spfull.keys())
    mode_list = {'accum', 'aitken', 'coarse', 'pcarbon'}
    filtered_key_list = [item for item in allKeys if item not in mode_list]
    for key in filtered_key_list:
        name = spfull[key].split('<strong>')[1].split('</strong>')[0]
        idval = spfull[key].split('id=')[1].split(' ')[0].strip('"')
        tmp = tmp + '<li><a href="#'+idval+'">'+name+'</a></li>\n'+' '*16
        
    df['Variable']=listofvs
    df['DJF']=df['Variable'].apply(lambda x: '<a href="{}_{}">DJF</a>'.format(x,form.replace('season','DJF')))
    df['JJA']=df['Variable'].apply(lambda x: '<a href="{}_{}">JJA</a>'.format(x,form.replace('season','JJA')))
    df['ANN']=df['Variable'].apply(lambda x: '<a href="{}_{}">ANN</a>'.format(x,form.replace('season','ANN')))

    if fmt == None:
        fmt = form.split('.')[1]
    for loc in locations:
        df[loc]=df['Variable'].apply(lambda x: '<a href="{}_{}">{}</a>'.format(x,form.split('.')[0].replace('season',loc)+'.'+fmt,loc))
    
    df['Variable'] = df['Variable'].replace({
                                            r'.*_accum$': 'accum',
                                            r'.*_aitken$': 'aitken',
                                            r'.*_coarse$': 'coarse',
                                            r'.*_pcarbon$': 'pcarbon'
                                            }, regex=True)
    
    df['Variable']=df['Variable'].map(spfull).fillna(df['Variable'])
    df.columns = ['Variable','','Seasons',' ']+locations

    # Table styling
    styles = [
        {
            'selector': 'caption',
            'props': [
                ('font-weight', 'bold'),
                ('font-size', '2em'),
                ('padding', '10px'),
                ('text-align', 'center'),
                ('color', '#333')
            ]
        },
        {
            'selector': 'th',
            'props': [
                ('font-size', '1.2em'),
                ('text-align', 'center'),
                ('background-color', '#eee'),
                ('color', '#555'),
                ('border-bottom', '2px solid #aaa')
            ]
        },
        {
            'selector': 'td',
            'props': [
                ('text-align', 'center'),
                ('font-family', 'calibri'),
                ('font-size', '12pt'),
                ('padding', '10px'),
                ('border-bottom', '1px solid #eee')
            ]
        },
        {
            'selector': 'a',
            'props': [
                ('color', '#337ab7'),
                ('text-decoration', 'none')
            ]
        },
        {
            'selector': 'a:hover',
            'props': [
                ('color', '#23527c'),
                ('text-decoration', 'underline')
            ]
        },
        {
            'selector': 'tbody tr:hover',
            'props': [
                ('background-color', '#f5f5f5')
            ]
        }
    ]

    
    html = (
        df.style.set_table_styles(styles)
        .set_properties(**{
            'font-family': 'calibri',
            'width': '12em',
            'padding': '10px'
        })
        .hide(axis="index")
        .to_html()
    )

    return html,title,tmp


def get_html_table(df):

    styles = [
        dict(selector=" ",
             props=[("margin", "0"),
                    ("font-family", "sans-serif"),
                    ("font-size", "medium"),
                    ("text-align", "right"),
                    ("width", "auto"),
                    ("border", "0"),
                       ]),

        dict(selector="tbody tr:nth-child(even)",
             props=[("background-color", "white")]),

        dict(selector="tbody tr:nth-child(odd)",
             props=[("background-color", "#EDEDED")]),

        # Adding hover effect for even rows
        dict(selector="tbody tr:nth-child(even):hover",
             props=[("background-color", "#D3D3D3")]),  # or any other color you prefer for hover

        # Adding hover effect for odd rows
        dict(selector="tbody tr:nth-child(odd):hover",
             props=[("background-color", "#BEBEBE")]),  # or any other color you prefer for hover

        dict(selector="td",
             props=[("padding", "5px")]),

        dict(selector="thead th",
             props=[("background-color", "#FFFFFF"),
                    ("border-bottom", "2px solid #808080"),
                    ("color", "#808080"),
                    ("text-align", "right"),
                    ("font-family", "sans-serif"),
                    ("font-size", "medium")]),
    ]
    return (df.style.set_table_styles(styles)).to_html()

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

def html_template(title,html,tmp):
    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PNNL Aerosol Diagnostics</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
            }}
            
            header {{
                background-color: #333;
                color: white;
                text-align: center;
                padding: 0;
                position: fixed;
                top: 0;
                width: 100%;
                z-index: 1000;
                box-shadow: 0px 3px 10px rgba(0, 0, 0, 0.2);
            }}
    
            #container {{
                display: flex;
            }}
    
            #side-nav {{
                position: fixed;
                top: 80px;
                height: calc(100% - 80px);
                overflow-y: auto;
                width: 20%;
                border-right: 1px solid #ccc;
                padding: 1rem;
                z-index: 500;
                background-color: #fff;
                box-shadow: 3px 0px 10px rgba(0, 0, 0, 0.1);
                padding-top: 5px;
            }}
    
            #side-nav ul {{
                list-style-type: none;
                padding: 0;
            }}
    
            #side-nav li {{
                margin-bottom: 1rem;
            }}
    
            #side-nav h2 {{
                color: #1976D2;
            }}
        
            #side-nav a {{
                text-decoration: none;
                color: #555;
                font-weight: bold;
                padding: 0.2rem;
                display: block;
                border-radius: 5px;
                transition: background-color 0.3s;
            }}
    
            #side-nav a:hover {{
                background-color: #eee;
            }}
    
            #content {{
                margin-top: 80px;
                margin-left: 24%;
                width: 80%;
                padding: 1rem;
            }}
        
            footer {{
                text-align: center;
                padding: 0.5rem 0;
                background-color: #333;
                color: #fff;
                font-size: 0.8em;
                position: fixed;
                bottom: 0;
                width: 100%;
                z-index: 999;
                box-shadow: 0px -3px 10px rgba(0, 0, 0, 0.2);
            }}
    
        
        </style>
    </head>
    
    <body>
    
        <header>
            <h1>PNNL Aerosol Diagnostics</h1>
        </header>
    
        <div id="container">
            <!-- Side Navigation -->
            <aside id="side-nav">
                <h2>{title}</h2>
                <ul>
                    {tmp}
                    <li style="margin-top: 5rem;"><a href="../aerosol.html" style="color: #1976D2;font-size: 20px;"><< Main Menu</a></li>
                </ul>
            </aside>
    
            <!-- Main Content -->
            <section id="content">
                {html}
            </section>
        </div>
    
        <footer>
            <p>&#169; 2023 Pacific Northwest National Laboratory. All rights reserved.</p>
        </footer>
    
    </body>
    
    </html>
    """
    
    return html_code

