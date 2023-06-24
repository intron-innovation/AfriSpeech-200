import seaborn as sns
import pandas as pd
sns.set_theme(style="whitegrid")



#https://bsouthga.dev/posts/color-gradients-with-python

def hex_to_RGB(hex):
  ''' "#FFFFFF" -> [255,255,255] '''
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]


def RGB_to_hex(RGB):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  RGB = [int(x) for x in RGB]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])

def color_dict(gradient):
  ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
  return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
      "r":[RGB[0] for RGB in gradient],
      "g":[RGB[1] for RGB in gradient],
      "b":[RGB[2] for RGB in gradient]}


def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
  ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
  # Starting and ending colors in RGB form
  s = hex_to_RGB(start_hex)
  f = hex_to_RGB(finish_hex)
  # Initilize a list of the output colors with the starting color
  RGB_list = [s]
  # Calcuate a color at each evenly spaced value of t from 1 to n
  for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
    curr_vector = [
      int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
      for j in range(3)
    ]
    # Add it to our list of output colors
    RGB_list.append(curr_vector)

  return color_dict(RGB_list)



import matplotlib.pyplot as plt
from colour import Color

import matplotlib.cbook as cbook
import matplotlib.image as image




def plot_table(langs,counts,n=21):

    #langs = df['language'].values
    #counts = df['count'].values

    color_gradient_count = max(counts)+10
    start_color = Color("#FFFFFF")
    
    colors_ =  linear_gradient(start_hex="#FFFFFF",finish_hex="#32527B",n=color_gradient_count)['hex']
    color_for_side = '#138a07'
    #colors_ = list(start_color.range_to(Color("#32527B"),color_gradient_count))
    #colors_ = [c.hex for c in colors_]
    colors_.append(color_for_side)

    all_ = []
    counts_use = []
    for l,c in zip(langs,counts):
        if '_MAINK' in l:
            l=l.replace('_MAINK','')
            all_.append(f"{l}")
            counts_use.append(-1) # special c 
        else:
            all_.append(f"{l}\n({c})")
            counts_use.append(c)


    cell_text = [all_[i:i + n] for i in range(0, len(all_), n)]
    counts_split = [counts_use[i:i + n] for i in range(0, len(counts_use), n)]

    # Add a table at the bottom of the axes
    colors = [[colors_[a] for a in c] for c in counts_split] 
    #colors = [[colors_[int(a.split('\n')[1].strip('()'))] for a in all_[i:i + n]] for i in range(0, len(all_), n)]

    #cell_text[-1] = cell_text[-1] + ['' for l in range(46-len(cell_text[-1]))]
    #colors[-1] = colors[-1] + [colors_[0] for l in range(46-len(colors[-1]))]

    #colors = [["#56b5fd","w","w","w","w"],[ "#1ac3f5","w","w","w","w"]]

    # fig, ax = plt.subplots(figsize=(50,50))

    # plt.figure(figsize=(50,50))
    # plt.axis('tight')
    # plt.axis('off')
    # the_table = ax.table(cellText=cell_text,cellColours=colors,loc='center')


    # original method

    #fig, ax = plt.subplots()
    plt.figure(figsize=(22,41))
    plt.axis('tight')
    plt.axis('off')
    the_table = plt.table(cellText=cell_text,cellColours=colors,loc='center',cellLoc='center')

    the_table.set_fontsize(200)
    #the_table.scale(1,1)
    #plt.title('Lanfrica Records per language')
    #plt.figimage(im, 2500, 10, zorder=3, alpha=.5)
    #plt.figimage(im2, 2000, 10, zorder=3, alpha=.5)

    #the_table.figure.set_figwidth(20)
    #the_table.figure.set_figheight(11.25)

    #the_table.figure.tight_layout()
    #plt.tight_layout()
    #the_table.figure.savefig('table_accent_chosen_heatmap.png')
    plt.savefig('table_accent_chosen_heatmap.png')

    breakpoint()
    return fig,the_table,colors_



def get_color_shade(colors_):

    # Make a figure and axes with dimensions as desired.
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    a = mpl.colors.LinearSegmentedColormap.from_list(name='myc',colors =colors_ )
    print(a)
    ticks=[0,1]
    vals=['Less','More']
    cmap = mpl.cm.cool
    bounds=[1,2]
    norm= mpl.colors.NoNorm
    #norm = mpl.colors.Normalize(vmin=5, vmax=10)
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=a,                           
                                    orientation='horizontal')
    cb1.set_ticks(ticks=ticks)
    cb1.set_ticklabels(ticklabels=vals)
    #cb1.set_label('Some Units')
    return cb1