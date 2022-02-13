import matplotlib.pyplot as plt

def save_img(outfile,dpi):
    pl.savefig(outfile, dpi=dpi, bbox_inches='tight')
    pl.close()