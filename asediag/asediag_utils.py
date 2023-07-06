import numpy as np
import pandas as pd
from pathlib import Path

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

def group_duplicate_index(df):
    a = df.values
    sidx = np.lexsort(a.T)
    b = a[sidx]

    m = np.concatenate(([False], (b[1:] == b[:-1]).all(1), [False] ))
    idx = np.flatnonzero(m[1:] != m[:-1])
    I = df.index[sidx].tolist()
    return [I[i:j] for i,j in zip(idx[::2],idx[1::2]+1)]

def get_html(form,title):
    df = pd.DataFrame()
    listofvs = ['bc','bc_a1', 'bc_a3', 'bc_a4', 'bc_c1', 'bc_c3', 'bc_c4',\
               'so4','so4_a1', 'so4_a2', 'so4_a3', 'so4_c1', 'so4_c2', 'so4_c3',\
               'dst','dst_a1', 'dst_a3', 'dst_c1', 'dst_c3',\
               'mom','mom_a1', 'mom_a2', 'mom_a3', 'mom_a4', 'mom_c1', 'mom_c2', 'mom_c3', 'mom_c4',\
               'pom','pom_a1', 'pom_a3', 'pom_a4', 'pom_c1', 'pom_c3', 'pom_c4',\
               'ncl','ncl_a1', 'ncl_a2', 'ncl_a3', 'ncl_c1', 'ncl_c2', 'ncl_c3',\
               'soa','soa_a1', 'soa_a2', 'soa_a3', 'soa_c1', 'soa_c2', 'soa_c3',\
               'num','num_a1', 'num_a2', 'num_a3', 'num_a4', 'num_c1', 'num_c2', 'num_c3', 'num_c4',\
               'SO2','DMS','H2SO4']
    spfull = {'bc':'<a id="Black Carbon"><font color="red"><strong>Black Carbon</string></font>','so4':'<a id="Sulfate"><font color="red"><strong>Sulfate</string></font>','dst':'<font color="red"><strong>Dust</string></font>','mom':'<font color="red"><strong>Marine organic matter</string></font>',\
             'pom':'<font color="red"><strong>Primary organic matter</string></font>','ncl':'<font color="red"><strong>Sea salt</string></font>','soa':'<font color="red"><strong>Secondary organic aerosol</string></font>',\
             'num':'<a id="Aerosol number"><font color="red"><strong>Aerosol number</string></font>',\
             'SO2':'<font color="red"><strong>SO2</string></font>',\
             'DMS':'<font color="red"><strong>DMS</string></font>',\
             'H2SO4':'<font color="red"><strong>H2SO4</string></font>'}
    df['Variable']=listofvs
    df['DJF']=df['Variable'].apply(lambda x: '<a href="{}_{}">DJF</a>'.format(x,form.replace('season','DJF')))
    df['JJA']=df['Variable'].apply(lambda x: '<a href="{}_{}">JJA</a>'.format(x,form.replace('season','JJA')))
    df['ANN']=df['Variable'].apply(lambda x: '<a href="{}_{}">ANN</a>'.format(x,form.replace('season','ANN')))
    df['Variable']=df['Variable'].map(spfull).fillna(df['Variable'])
    df.columns = ['Variable','','Seasons',' ']
    styler = df.style
    styler=styler.set_caption(title).set_table_styles([
        {'selector':'caption',
        'props':[
            ('font-weight','bold'),
            ('font-size','2.5em'),
            ('padding-bottom','1em'),
            ('text-align','center'),
            ('border-width','0.5em')]},
        {'selector':'th.col_heading',
        'props':[
            ('font-size','1.5em'),
            ('padding-bottom','1em')]}
    ])

    html = (
        styler.set_properties(**{'font-size':'12pt','font-family':'calibri','width':'12em','text-align':'center','padding-bottom':'1em'}).hide(axis="index").to_html()
    )
    html=html.replace('</caption>','</caption>  <caption style = "font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding-left: 2.5em"></caption>')
    return html

def get_html_table(df):

    styles = [
        dict(selector=" ", 
             props=[("margin","0"),
                    ("font-family","sans-serif"),
                    {"font-size","medium"},
                    {"text-align","right"},
                    {"width","auto"},
                    ("border","0"),
                       ]),

        dict(selector="tbody tr:nth-child(even)",
             props=[("background-color", "white")]),
        dict(selector="tbody tr:nth-child(odd)",
             props=[("background-color", "#EDEDED")]),

        dict(selector="td", 
             props=[("padding", "5px")]),

        dict(selector="thead th",
             props=[("background-color", "#FFFFFF"),
                    {"border-bottom","2px solid #808080"},
                    {"color","#808080"},
                    {"text-align","right"},
                    ("font-family","sans-serif"),
                    {"font-size","medium"}]),
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
        return s1
    
    def vmap(self):
        s1=[0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000]
        aagg=(np.max(self.v1).values+np.max(self.v2).values)/2
        aagg=np.log10(aagg)
        s1=np.array(s1)*(10**(np.round(aagg-2.7)))
        return s1
    
    def vdiff(self):
        s2=[-100,-50.,-20,-10,-5,-2,2,5,10,20,50,100]
        if (abs(np.max(self.v1).values)/abs(np.max(self.diff).values))<10:
            aagg=0.25*0.1*(abs(np.max(self.diff).values)+abs(np.min(self.diff).values))/2
        else:                           
            aagg=0.25*(abs(np.max(self.diff).values)+abs(np.min(self.diff).values))/2
        aagg=np.log10(aagg)
        s1 = np.array(s2)*(10**(np.round(aagg-1.7)))*10
        return s1

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
