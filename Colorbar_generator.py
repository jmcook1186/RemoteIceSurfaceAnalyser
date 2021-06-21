import matplotlib as mpl
import matplotlib.pyplot as plt



def generate_colorbar(min_val, max_val, cmap, ticks, savepath):

    fig, ax = plt.subplots(1, 1)

    fraction = 1

    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    cbar = ax.figure.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax, pad=.05, extend='both', fraction=fraction,ticks=ticks)

    ax.axis('off')
    plt.savefig(savepath,dpi=300)
    
    return



# DEFINE OPTIONS

min_val = 500 # lowest value on colorbar
max_val = 1000 # highest value on colorbar
cmap = 'cividis' # color scheme
ticks =[500, 600, 700, 800, 900, 1000] # labels to include on colorbar
savepath = '/home/joe/Code/Remote_Ice_Surface_Analyser/Manuscript/Figures/Colorbar_reff.jpg'
generate_colorbar(min_val, max_val, cmap, ticks, savepath)