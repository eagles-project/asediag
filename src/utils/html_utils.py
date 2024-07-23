import pandas as pd

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
        listofvs = listofvs + [''] + extra

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
    mode_list = {'accum', 'aitken', 'coarse', 'pcarbon', 'mode'}
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

